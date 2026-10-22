"""Find nearest train questions using masked-mean encoder embeddings."""

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from datasets import load_from_disk
from hydra.utils import instantiate
from tokenizers import Tokenizer
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.utils.io_utils import ROOT_PATH


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).float()
    pooled = (hidden.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
    return F.normalize(pooled, dim=-1)


@torch.inference_mode()
def encode(model, rows, device):
    if any(not row for row in rows):
        raise ValueError("Cannot embed an empty question")
    inputs = np.full(
        (len(rows), max(map(len, rows))), model.config.pad_id, dtype=np.int64
    )
    for index, row in enumerate(rows):
        inputs[index, : len(row)] = row
    inputs = torch.from_numpy(inputs).to(device)
    mask = inputs != model.config.pad_id
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        hidden = model.encoder(
            model.src_pos_embedding(inputs), model.make_source_allowed_mask(mask)
        )
    return masked_mean(hidden, mask)


def merge_nearest(queries, vectors, row_ids, best_scores, best_ids):
    """Exact search within a block, then merge it with the previous top-k."""
    k = best_scores.shape[1]
    scores, local_ids = (queries @ vectors.T).topk(min(k, len(vectors)), dim=1)
    ids = row_ids[local_ids]
    scores = torch.cat((best_scores, scores), dim=1)
    ids = torch.cat((best_ids, ids), dim=1)
    scores, order = scores.topk(k, dim=1)
    return scores, ids.gather(1, order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="saved/strong_body_full_clean_cosine_16k_len128_b128_200ep/model_best.pth",
    )
    parser.add_argument("--output", default="artifacts/train_test_cosine")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--train-limit",
        type=int,
        help="Only for a quick check; omit to search all train rows",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.chunk_size, args.top_k) < 1 or (
        args.train_limit is not None and args.train_limit < 1
    ):
        parser.error("Sizes must be positive")

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    checkpoint_path = ROOT_PATH / args.checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = checkpoint["config"]
    model = instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    checkpoint_epoch = checkpoint["epoch"]
    del checkpoint

    data_path = ROOT_PATH / config.data.dataset_path
    train = load_from_disk(str(data_path / "train"))
    test = load_from_disk(str(data_path / "test")).select_columns(["input_ids"])
    indices_path = ROOT_PATH / config.datasets.train.indices_path
    manifest = json.loads(indices_path.with_suffix(".json").read_text())
    if manifest["source_fingerprint"] != train._fingerprint or manifest[
        "source_rows"
    ] != len(train):
        raise ValueError("Length-filter indices belong to a different dataset")
    indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
    if len(indices) != manifest["kept_rows"]:
        raise ValueError("Length-filter indices do not match the manifest")
    if args.train_limit is not None:
        indices = indices[: args.train_limit]
    if len(indices) < args.top_k:
        raise ValueError("Train is smaller than top-k")
    train = train.select_columns(["input_ids"])
    test_rows = test[:]["input_ids"]
    output = ROOT_PATH / args.output
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT_PATH / config.data.tokenizer_path
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    preparation = data_path / "preparation.json"
    if preparation.exists():
        expected_hash = json.loads(preparation.read_text()).get("tokenizer_sha256")
        if (
            expected_hash
            and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("Tokenizer has changed since dataset preparation")
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer and model vocabularies differ")

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    queries = torch.cat(
        [
            encode(model, test_rows[start : start + args.batch_size], device)
            for start in range(0, len(test_rows), args.batch_size)
        ]
    )
    np.save(output / "test_embeddings.npy", queries.cpu().numpy())
    best_scores = torch.full((len(test), args.top_k), -torch.inf, device=device)
    best_ids = torch.full((len(test), args.top_k), -1, dtype=torch.long, device=device)
    train_sum = torch.zeros(queries.shape[1], dtype=torch.float64, device=device)

    with torch.inference_mode(), tqdm(
        total=len(indices), desc="Encoder + nearest train", mininterval=20
    ) as progress:
        for start in range(0, len(indices), args.chunk_size):
            source_ids = np.asarray(indices[start : start + args.chunk_size])
            rows = train[source_ids.tolist()]["input_ids"]
            # Local length sorting reduces padding; source row IDs follow the same order.
            order = np.argsort(list(map(len, rows)), kind="stable")
            rows = [rows[index] for index in order]
            row_ids = torch.as_tensor(source_ids[order], device=device)
            vectors = torch.cat(
                [
                    encode(model, rows[offset : offset + args.batch_size], device)
                    for offset in range(0, len(rows), args.batch_size)
                ]
            )
            train_sum += vectors.sum(0, dtype=torch.float64)
            best_scores, best_ids = merge_nearest(
                queries, vectors, row_ids, best_scores, best_ids
            )
            progress.update(len(rows))

    # For normalized vectors, the mean of ALL pairwise cosines equals this dot product.
    mean_pairwise = float(
        (queries.double().mean(0) * (train_sum / len(indices))).sum().item()
    )
    scores, nearest_ids = best_scores.cpu().numpy(), best_ids.cpu().numpy()
    elapsed = time.perf_counter() - started
    np.savez(output / "nearest.npz", scores=scores, train_indices=nearest_ids)
    unique_ids = np.unique(nearest_ids)
    nearest_texts = tokenizer.decode_batch(train[unique_ids.tolist()]["input_ids"])
    texts = dict(zip(map(int, unique_ids), nearest_texts))
    questions = tokenizer.decode_batch(test_rows)
    with (output / "nearest.jsonl").open("w", encoding="utf-8") as file:
        for index, question in enumerate(questions):
            neighbors = [
                {
                    "train_index": int(row_id),
                    "cosine_similarity": float(score),
                    "question": texts[int(row_id)],
                }
                for row_id, score in zip(nearest_ids[index], scores[index])
            ]
            file.write(
                json.dumps(
                    {"test_index": index, "question": question, "neighbors": neighbors},
                    ensure_ascii=False,
                )
                + "\n"
            )
    top1 = scores[:, 0]
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "dataset": str(data_path),
        "train_indices": str(indices_path),
        "train_index_meaning": "Row in the tokenized train split before length filtering",
        "train_rows": len(indices),
        "test_rows": len(test),
        "embedding": "Last encoder layer, masked mean over non-PAD input tokens, L2 normalized",
        "embedding_dim": int(queries.shape[1]),
        "encoder_precision": (
            "bfloat16 autocast" if device.type == "cuda" else "float32"
        ),
        "similarity_precision": "float32, exhaustive block search",
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "mean_all_pairwise_cosine": mean_pairwise,
        "nearest_cosine": {
            "mean": float(top1.mean()),
            "min": float(top1.min()),
            "p05": float(np.quantile(top1, 0.05)),
            "median": float(np.median(top1)),
            "p95": float(np.quantile(top1, 0.95)),
            "max": float(top1.max()),
        },
        "nearest_above": {
            str(threshold): int((top1 >= threshold).sum())
            for threshold in (0.8, 0.9, 0.95, 0.99, 0.999)
        },
        "identical_question_in_top_k": sum(
            any(questions[i] == texts[int(row_id)] for row_id in nearest_ids[i])
            for i in range(len(test))
        ),
        "encoder_and_search_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
