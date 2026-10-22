"""Train the tokenizer and/or tokenize TinySMS using a YAML config."""

import argparse
from pathlib import Path

import yaml

from src.tokenizer import prepare_code_dataset, tokenize_dataset, train_tokenizer
from src.utils.io_utils import ROOT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/configs/tokenizer.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_PATH / config_path
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config.get("prepare_code", False):
        prepare_code_dataset(config)
    if config["train_tokenizer"]:
        train_tokenizer(config)
    if config["tokenize_dataset"]:
        tokenize_dataset(config)


if __name__ == "__main__":
    main()
