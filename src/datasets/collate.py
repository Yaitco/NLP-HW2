import torch
from transformers import DataCollatorForSeq2Seq

from src.tokenizer import load_tokenizer


class Seq2SeqCollator:
    """Dynamically pad source/target IDs; support old targets without EOS."""

    def __init__(self, tokenizer_path, pad_id, eos_token_id, pad_to_multiple_of=None):
        tokenizer = load_tokenizer(tokenizer_path)
        if tokenizer.pad_token_id != pad_id or tokenizer.eos_token_id != eos_token_id:
            raise ValueError("Tokenizer PAD/EOS IDs do not match the model config")
        self.eos_token_id = eos_token_id
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=pad_id,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, dataset_items):
        features = []
        for item in dataset_items:
            feature = dict(item)
            labels = feature.get("labels")
            if labels is None:
                feature.pop("labels", None)
            else:
                labels = list(labels)
                if not labels or labels[-1] != self.eos_token_id:
                    labels.append(self.eos_token_id)
                feature["labels"] = labels
            features.append(feature)
        return self.collator(features)


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
