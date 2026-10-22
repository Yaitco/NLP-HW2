"""Export an inference bundle; optionally upload it to a Hugging Face model repo."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from huggingface_hub import HfApi
from omegaconf import OmegaConf
from tokenizers import Tokenizer
import torch

from src.model import Seq2SeqTransformer, Seq2SeqTransformerConfig
from src.utils.io_utils import ROOT_PATH

BUNDLE_FILES = ["model.pt", "config.json", "tokenizer.json", "README.md"]


def export_model(checkpoint_path, output):
    # Project training checkpoints contain OmegaConf objects: use trusted local files.
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config = checkpoint["config"]
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    model_config = Seq2SeqTransformerConfig(
        **{
            key: value
            for key, value in config["model"]["config"].items()
            if not key.startswith("_")
        }
    )
    tokenizer_path = ROOT_PATH / config["data"]["tokenizer_path"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != model_config.vocab_size:
        raise ValueError("Checkpoint and tokenizer vocabulary sizes differ")
    for token, token_id in [
        ("<pad>", model_config.pad_id),
        ("<bos>", model_config.decoder_start_token_id),
        ("<eos>", model_config.eos_token_id),
    ]:
        if tokenizer.token_to_id(token) != token_id:
            raise ValueError(f"Checkpoint and tokenizer disagree on {token}")
    code_format = config["data"].get("code_format", "full")
    if code_format not in ("full", "solve_body_result"):
        raise ValueError(f"Unknown code format: {code_format}")

    model = Seq2SeqTransformer(model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    shutil.copyfile(tokenizer_path, output / "tokenizer.json")
    (output / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "code_format": code_format,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "README.md").write_text(
        "---\npipeline_tag: text-generation\nlanguage: en\n"
        "datasets:\n- SlavaYus/tinysms_qa_12m\ntags:\n- pytorch\n- code-generation\n---\n\n"
        "# TinySMS Transformer\n\n"
        "Custom encoder-decoder Transformer: arithmetic word problems to Python code.\n\n"
        "Use `notebooks/inference.ipynb` from the project repository. "
        "The bundle contains model weights, architecture config, and its matching tokenizer. "
        "Optimizer state and training configuration are excluded.\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="saved/strong_full_clean_sqrt_16k_len128_b64/model_best.pth",
    )
    parser.add_argument("--output", default="artifacts/hf_model")
    parser.add_argument(
        "--repo-id", help="Upload after export, e.g. YaItco/NLP-2-transformer"
    )
    parser.add_argument(
        "--private", action="store_true", help="Create a private repo (default: public)"
    )
    args = parser.parse_args()
    output = export_model(ROOT_PATH / args.checkpoint, ROOT_PATH / args.output)
    print(f"Exported model: {output}")
    if args.repo_id:
        api = HfApi()
        api.create_repo(
            repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private
        )
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(output),
            allow_patterns=BUNDLE_FILES,
        )
        print(result)


if __name__ == "__main__":
    main()
