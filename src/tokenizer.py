import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

import datasets
from src.utils.io_utils import ROOT_PATH


def load_tokenizer(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_PATH / path
    raw_tokenizer = Tokenizer.from_file(str(path))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    return tokenizer


def build_tokenizer(config):
    return load_tokenizer(config["tokenizer"]["output_path"])


def load_preparation_dataset(config, normalized=True):
    settings = config.get("preprocessing", {}).get("code_normalization")
    if settings and normalized:
        path = ROOT_PATH / settings["output_path"]
        report = json.loads((path / "preparation.json").read_text())
        if (
            report.get("normalizer") != "ast_unparse"
            or report["code_normalization"] != settings
            or report["source_dataset"] != config["dataset"]
            or report["cleaning"] != config["preprocessing"].get("cleaning")
        ):
            raise ValueError("Normalized dataset belongs to a different configuration")
        return datasets.load_from_disk(str(path)), report

    data = datasets.load_dataset(
        config["dataset"]["path"],
        download_config=datasets.DownloadConfig(
            local_files_only=config["dataset"].get("local_files_only", False)
        ),
    )
    report = (
        select_numeric_train(data, config)
        if config.get("preprocessing", {}).get("cleaning")
        else None
    )
    return data, report


def prepare_code_dataset(config):
    from src.code_preprocessing import normalize_batch

    preprocessing = config["preprocessing"]
    settings = preprocessing["code_normalization"]
    output = ROOT_PATH / settings["output_path"]
    if output.exists():
        # Only reuse a completed dataset made with the same configuration.
        load_preparation_dataset(config)
        print(f"Reusing normalized dataset: {output}", flush=True)
        return
    data, report = load_preparation_dataset(config, normalized=False)
    for split, dataset in data.items():
        data[split] = dataset.map(
            normalize_batch,
            batched=True,
            with_indices=True,
            batch_size=preprocessing["batch_size"],
            num_proc=preprocessing["num_proc"],
            fn_kwargs={
                "code_column": config["dataset"]["code_column"],
                "strip_imports": settings.get("strip_imports", False),
                "question_column": config["dataset"]["question_column"] if settings.get("clean_questions", False) else None,
            },
            desc=f"Normalize {split} code",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(output))
    report = {
        **(report or {}),
        "source_dataset": config["dataset"],
        "cleaning": preprocessing.get("cleaning"),
        "code_normalization": settings,
        "normalizer": "ast_unparse",
        "code_format": "solve_body_result",
        "rows": {split: len(dataset) for split, dataset in data.items()},
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def batch_iterator(
    dataset, batch_size=10000, question_column="question", code_column="code"
):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        texts = []

        for question, code in zip(batch[question_column], batch[code_column]):
            texts.append(question)
            texts.append(code)

        yield texts


def train_tokenizer(config):
    output_path = ROOT_PATH / config["tokenizer"]["output_path"]
    if output_path.exists():
        raise FileExistsError(f"Tokenizer already exists: {output_path}")
    dataset, _ = load_preparation_dataset(config)
    tokenizer_train_dataset = (
        dataset["train"]
        .select_columns(
            [config["dataset"]["question_column"], config["dataset"]["code_column"]]
        )
        .shuffle(seed=42)
    )
    size = config["tokenizer"]["training_examples"]
    if size is not None:
        tokenizer_train_dataset = tokenizer_train_dataset.select(
            range(min(size, len(tokenizer_train_dataset)))
        )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config["tokenizer"]["vocab_size"],
        min_frequency=config["tokenizer"]["min_frequency"],
        special_tokens=config["tokenizer"]["special_tokens"],
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        batch_iterator(
            tokenizer_train_dataset,
            question_column=config["dataset"]["question_column"],
            code_column=config["dataset"]["code_column"],
        ),
        trainer=trainer,
        length=len(tokenizer_train_dataset) * 2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))


def tokenize_dataset(config):
    output_path = ROOT_PATH / config["preprocessing"]["output_path"]
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}; choose a new output_path"
        )
    dataset, report = load_preparation_dataset(config)

    for split, settings in config["preprocessing"]["splits"].items():
        size = settings["max_examples"]
        if size is not None:
            dataset[split] = dataset[split].select(
                range(min(size, len(dataset[split])))
            )

    tokenizer = build_tokenizer(config).backend_tokenizer
    question_column = config["dataset"]["question_column"]
    code_column = config["dataset"]["code_column"]
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain <eos>")

    def tokenizer_func(batch):
        inputs = tokenizer.encode_batch(
            batch[question_column], add_special_tokens=False
        )
        targets = tokenizer.encode_batch(batch[code_column], add_special_tokens=False)

        return {
            "input_ids": [e.ids for e in inputs],
            "attention_mask": [e.attention_mask for e in inputs],
            "labels": [e.ids + [eos_id] for e in targets],
        }

    def test_tokenizer_func(batch):
        inputs = tokenizer.encode_batch(
            batch[question_column], add_special_tokens=False
        )

        return {
            "input_ids": [e.ids for e in inputs],
            "attention_mask": [e.attention_mask for e in inputs],
            "labels": [None for _ in inputs],
        }

    tokenized_dataset = datasets.DatasetDict()
    for split, data in dataset.items():
        data = data.select_columns([question_column, code_column])
        tokenized_dataset[split] = data.map(
            test_tokenizer_func if split == "test" else tokenizer_func,
            batched=True,
            batch_size=config["preprocessing"]["batch_size"],
            remove_columns=data.column_names,
            num_proc=config["preprocessing"]["num_proc"],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(str(output_path))
    if report is not None:
        report["tokenizer_sha256"] = hashlib.sha256(
            (ROOT_PATH / config["tokenizer"]["output_path"]).read_bytes()
        ).hexdigest()
        lengths = pc.list_value_length(
            tokenized_dataset["train"].data.column("labels")
        ).to_numpy()
        report["code_tokens_with_eos"] = {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
            "percent_at_most_60": float((lengths <= 60).mean() * 100),
        }
        (output_path / "preparation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)


def select_numeric_train(dataset, config):
    """Reuse the full reference audit and hold out the original validation rows."""
    settings = config["preprocessing"]["cleaning"]
    audit_dir = ROOT_PATH / settings["audit_dir"]
    audit = json.loads((audit_dir / "manifest.json").read_text())
    signature = audit["signature"]
    train = dataset["train"]
    # Execution statuses describe raw code. Token-length arrays in the audit
    # depend on its tokenizer, but the numeric selection does not.
    if not audit.get("complete") or any(
        (
            signature["dataset"] != config["dataset"]["path"],
            signature["fingerprint"] != train._fingerprint,
            signature["rows"] != len(train),
        )
    ):
        raise ValueError("Full audit and raw dataset must match")
    statuses = np.load(audit_dir / "status.npy", mmap_mode="r", allow_pickle=False)
    if statuses.shape != (len(train),):
        raise ValueError("Audit status file length mismatch")
    keep = statuses == 0  # numeric_finite; no program execution during tokenization.
    # Retain numeric rows already recovered by the earlier per-program recheck.
    if settings.get("rechecked_manifest"):
        checked = json.loads((ROOT_PATH / settings["rechecked_manifest"]).read_text())
        if (
            checked["raw_fingerprint"] != train._fingerprint
            or checked["tokenizer_sha256"] != signature["tokenizer_sha256"]
        ):
            raise ValueError("Rechecked rows belong to a different dataset/tokenizer")
        keep[checked["recovered_rows"]] = True
    start, size = settings["validation_start"], settings["validation_size"]
    if start < 0 or size < 1 or start + size > len(train):
        raise ValueError("Validation range is outside the raw train")
    dataset["validation"] = train.select(range(start, start + size))
    keep[start : start + size] = False
    dataset["train"] = train.select(np.flatnonzero(keep))
    report = {
        "policy": "finite_real_solve_only_from_audit",
        "raw_rows": len(train),
        "raw_fingerprint": train._fingerprint,
        "audit_tokenizer_sha256": signature["tokenizer_sha256"],
        "train_rows": len(dataset["train"]),
        "validation_start": start,
        "validation_rows": size,
        "test_rows": len(dataset["test"]),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report
