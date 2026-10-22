import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        token_embedding,
        max_positions: int,
        dropout: float,
    ):
        super().__init__()
        self.max_positions = max_positions
        self.position_embedding = nn.Embedding(
            max_positions, token_embedding.embedding_dim
        )
        self.token_embedding = token_embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, position_offset=0):
        seq_len = input_ids.shape[1]
        if position_offset + seq_len > self.max_positions:
            raise ValueError("Sequence exceeds max_positions")

        positions = torch.arange(
            position_offset, position_offset + seq_len, device=input_ids.device
        )
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_heads <= 0 or d_model <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x = x.view(b, l, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, num_heads, l, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, num_heads * head_dim)

    def project_kv(self, key_value):
        return (
            self._split_heads(self.k_proj(key_value)),
            self._split_heads(self.v_proj(key_value)),
        )

    def attend(self, query, k, v, allowed_mask=None):
        q = self._split_heads(self.q_proj(query))
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, allowed_mask, dropout_p=dropout_p)
        out = self._merge_heads(out)

        out = self.out_proj(out)
        return out

    def forward(self, query, key_value, allowed_mask=None):
        k, v = self.project_kv(key_value)
        return self.attend(query, k, v, allowed_mask)

    def forward_step(self, query, key_value, allowed_mask, cache=None, static=False):
        if static and cache is not None:  # Encoder K/V do not change.
            k, v = cache
        else:
            k, v = self.project_kv(key_value)
            if cache is not None:
                k = torch.cat((cache[0], k), dim=2)
                v = torch.cat((cache[1], v), dim=2)
        return self.attend(query, k, v, allowed_mask), (k, v)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, allowed_mask=None):
        normalized = self.ln1(x)
        y = self.sa(normalized, normalized, allowed_mask)
        x = x + self.residual_dropout(y)

        x = x + self.ff(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ca = MultiHeadAttention(d_model, num_heads, dropout=dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_hidden,
        self_allowed_mask=None,
        cross_allowed_mask=None,
    ):
        # x:              [B, T, D]
        # encoder_hidden: [B, S, D]

        normalized = self.ln1(x)
        y = self.sa(normalized, normalized, self_allowed_mask)
        x = x + self.residual_dropout(y)

        normalized = self.ln2(x)
        y = self.ca(normalized, encoder_hidden, cross_allowed_mask)
        x = x + self.residual_dropout(y)

        normalized = self.ln3(x)
        x = x + self.ff(normalized)
        return x

    def forward_step(self, x, encoder_hidden, self_mask, source_mask, cache=None):
        self_kv, cross_kv = (None, None) if cache is None else cache
        normalized = self.ln1(x)
        y, self_kv = self.sa.forward_step(normalized, normalized, self_mask, self_kv)
        x = x + self.residual_dropout(y)
        y, cross_kv = self.ca.forward_step(
            self.ln2(x), encoder_hidden, source_mask, cross_kv, static=True
        )
        x = x + self.residual_dropout(y)
        return x + self.ff(self.ln3(x)), (self_kv, cross_kv)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, allowed_mask=None):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, allowed_mask, use_reentrant=False)
            else:
                x = block(x, allowed_mask)
        return self.ln(x)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        num_layers: int,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        encoder_hidden,
        self_allowed_mask=None,
        cross_allowed_mask=None,
    ):
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block,
                    x,
                    encoder_hidden,
                    self_allowed_mask,
                    cross_allowed_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, encoder_hidden, self_allowed_mask, cross_allowed_mask)
        return self.ln(x)

    def forward_step(self, x, encoder_hidden, self_mask, source_mask, cache=None):
        if cache is None:
            cache = [None] * len(self.blocks)
        next_cache = []
        for block, layer_cache in zip(self.blocks, cache):
            x, layer_cache = block.forward_step(
                x, encoder_hidden, self_mask, source_mask, layer_cache
            )
            next_cache.append(layer_cache)
        return self.ln(x), next_cache


@dataclass
class Seq2SeqTransformerConfig:
    vocab_size: int
    max_src_len: int
    max_tgt_len: int
    pad_id: int
    decoder_start_token_id: int
    eos_token_id: int
    d_model: int = 256
    n_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    d_ff: int = 1024
    dropout: float = 0.1
    gradient_checkpointing: bool = False


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: Seq2SeqTransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )

        self.src_pos_embedding = TokenPositionEmbedding(
            self.token_embedding, config.max_src_len, dropout=config.dropout
        )

        self.tgt_pos_embedding = TokenPositionEmbedding(
            self.token_embedding, config.max_tgt_len, dropout=config.dropout
        )

        self.encoder = Encoder(
            d_model=config.d_model,
            num_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            num_layers=config.encoder_layers,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        self.decoder = Decoder(
            d_model=config.d_model,
            num_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            num_layers=config.decoder_layers,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        decoder_input_ids = torch.full_like(labels, self.config.pad_id)

        decoder_input_ids[:, 1:] = labels[:, :-1]
        decoder_input_ids[:, 0] = self.config.decoder_start_token_id
        return decoder_input_ids

    def make_source_allowed_mask(self, attention_mask: torch.Tensor):
        B, S = attention_mask.shape
        return (attention_mask == 1).view(B, 1, 1, S)

    def make_decoder_allowed_mask(self, decoder_input_ids):
        B, T = decoder_input_ids.shape
        key_padding_mask = (decoder_input_ids != self.config.pad_id).view(B, 1, 1, T)

        causal_mask = torch.ones(
            T,
            T,
            dtype=torch.bool,
            device=decoder_input_ids.device,
        ).tril()

        return key_padding_mask & causal_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        **batch,
    ):
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_id

        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Provide decoder_input_ids or labels")

            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)

        source_allowed_mask = self.make_source_allowed_mask(attention_mask)
        decoder_allowed_mask = self.make_decoder_allowed_mask(decoder_input_ids)

        embeddings = self.src_pos_embedding(input_ids)
        encoder_hidden = self.encoder(embeddings, source_allowed_mask)

        embeddings = self.tgt_pos_embedding(decoder_input_ids)
        decoder_hidden = self.decoder(
            embeddings, encoder_hidden, decoder_allowed_mask, source_allowed_mask
        )

        logits = self.lm_head(decoder_hidden)

        return {"logits": logits}

    @torch.inference_mode()
    def generate_greedy(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=256,
        use_cache=True,
    ):
        return self.generate(
            input_ids, attention_mask, max_new_tokens, use_cache=use_cache
        )

    def _decode_next(self, tokens, encoder_hidden, source_mask, cache, use_cache):
        if use_cache:
            x = self.tgt_pos_embedding(
                tokens[:, -1:], position_offset=tokens.shape[1] - 1
            )
            # One query can see all previous non-padding keys; no causal mask needed.
            self_mask = (tokens != self.config.pad_id)[:, None, None, :]
            hidden, cache = self.decoder.forward_step(
                x, encoder_hidden, self_mask, source_mask, cache
            )
        else:
            hidden = self.decoder(
                self.tgt_pos_embedding(tokens),
                encoder_hidden,
                self.make_decoder_allowed_mask(tokens),
                source_mask,
            )
        return self.lm_head(hidden[:, -1]), cache

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=256,
        num_beams=1,
        length_penalty=1.0,
        use_cache=True,
    ):
        """Return one best sequence per input, without BOS. Finished beams append PAD."""
        if (
            not isinstance(num_beams, int)
            or not 1 <= num_beams <= self.config.vocab_size
        ):
            raise ValueError("num_beams must be between 1 and vocab_size")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError("length_penalty must be finite and non-negative")
        self.eval()

        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        source_mask = self.make_source_allowed_mask(attention_mask)
        encoder_hidden = self.encoder(self.src_pos_embedding(input_ids), source_mask)
        if num_beams > 1:
            encoder_hidden = encoder_hidden.repeat_interleave(num_beams, dim=0)
            source_mask = source_mask.repeat_interleave(num_beams, dim=0)
        tokens = torch.full(
            (batch_size * num_beams, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        scores = torch.full((batch_size, num_beams), -torch.inf, device=device)
        scores[:, 0] = 0  # Start with one live hypothesis, not duplicated beams.
        lengths = torch.zeros_like(scores, dtype=torch.long)
        finished = torch.zeros_like(scores, dtype=torch.bool)
        best_finished_score = torch.full((batch_size,), -torch.inf, device=device)
        best_finished_tokens = tokens.new_empty(batch_size, 0)
        batch_indices = torch.arange(batch_size, device=device)
        beam_offsets = batch_indices[:, None] * num_beams
        cache = None
        steps = min(max_new_tokens, self.config.max_tgt_len)
        for _ in range(steps):
            logits, cache = self._decode_next(
                tokens, encoder_hidden, source_mask, cache, use_cache
            )
            if num_beams == 1:
                next_tokens = torch.where(
                    finished, self.config.pad_id, logits.argmax(-1)[:, None]
                )
                tokens = torch.cat((tokens, next_tokens), dim=1)
                finished |= next_tokens == self.config.eos_token_id
                if finished.all():
                    break
                continue
            log_probs = F.log_softmax(logits.float(), dim=-1).view(
                batch_size, num_beams, -1
            )
            log_probs = log_probs.masked_fill(finished[:, :, None], -torch.inf)
            log_probs[:, :, self.config.pad_id] = torch.where(
                finished, 0.0, log_probs[:, :, self.config.pad_id]
            )
            candidates = scores[:, :, None] + log_probs
            next_lengths = lengths + (~finished).long()
            ranks = candidates / next_lengths.clamp_min(1)[:, :, None].pow(
                length_penalty
            )
            selected = ranks.flatten(1).topk(num_beams, dim=-1).indices
            parents = selected // self.config.vocab_size
            next_tokens = selected % self.config.vocab_size
            scores = candidates.flatten(1).gather(1, selected)
            lengths = next_lengths.gather(1, parents)
            finished = finished.gather(1, parents) | (
                next_tokens == self.config.eos_token_id
            )
            parent_indices = (parents + beam_offsets).flatten()
            tokens = torch.cat(
                (tokens.index_select(0, parent_indices), next_tokens.reshape(-1, 1)),
                dim=1,
            )
            if cache is not None:
                # Cross K/V are identical within one input's beams; only self K/V move.
                cache = [
                    (
                        tuple(t.index_select(0, parent_indices) for t in self_kv),
                        cross_kv,
                    )
                    for self_kv, cross_kv in cache
                ]
            normalized_scores = scores / lengths.clamp_min(1).pow(length_penalty)
            completed_scores, completed_beams = normalized_scores.masked_fill(
                ~finished, -torch.inf
            ).max(dim=1)
            completed_tokens = tokens.view(batch_size, num_beams, -1)[
                batch_indices, completed_beams, 1:
            ]
            best_finished_tokens = F.pad(
                best_finished_tokens, (0, 1), value=self.config.pad_id
            )
            improved = completed_scores > best_finished_score
            best_finished_tokens = torch.where(
                improved[:, None], completed_tokens, best_finished_tokens
            )
            best_finished_score = torch.maximum(best_finished_score, completed_scores)
            if finished.all():
                break
        if num_beams == 1:
            return tokens[:, 1:]
        normalized_scores = scores / lengths.clamp_min(1).pow(length_penalty)
        best_scores, best_beams = normalized_scores.max(dim=1)
        best_tokens = tokens.view(batch_size, num_beams, -1)[
            batch_indices, best_beams, 1:
        ]
        # Do not lose a completed result if later hypotheses displaced it from the pool.
        return torch.where(
            (best_finished_score > best_scores)[:, None],
            best_finished_tokens,
            best_tokens,
        )
