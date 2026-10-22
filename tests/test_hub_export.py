"""Tiny local export and notebook checks; no network, user checkpoints, or GPU."""

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
import torch

from export_model import BUNDLE_FILES, export_model, main
from src.model import Seq2SeqTransformer, Seq2SeqTransformerConfig
from src.utils.io_utils import ROOT_PATH


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer = Tokenizer(
            WordLevel(
                {
                    "<pad>": 0,
                    "<unk>": 1,
                    "<bos>": 2,
                    "<eos>": 3,
                    "a": 4,
                    "b": 5,
                    "c": 6,
                },
                unk_token="<unk>",
            )
        )
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.add_special_tokens(["<pad>", "<unk>", "<bos>", "<eos>"])
        self.tokenizer.save(str(self.tokenizer_path))
        self.config = Seq2SeqTransformerConfig(
            vocab_size=7,
            max_src_len=8,
            max_tgt_len=8,
            pad_id=0,
            decoder_start_token_id=2,
            eos_token_id=3,
            d_model=8,
            n_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            d_ff=16,
            dropout=0,
        )
        self.model = Seq2SeqTransformer(self.config).eval()
        self.checkpoint = self.root / "checkpoint.pth"
        torch.save(
            {
                "config": OmegaConf.create(
                    {
                        "model": {
                            "config": {
                                "_target_": "src.model.Seq2SeqTransformerConfig",
                                **asdict(self.config),
                            }
                        },
                        "data": {"tokenizer_path": str(self.tokenizer_path)},
                        "writer": {"api_key": "must-not-be-exported"},
                    }
                ),
                "state_dict": self.model.state_dict(),
                "optimizer": {"private_state": "must-not-be-exported"},
            },
            self.checkpoint,
        )
        self.output = self.root / "bundle"

    def test_bundle_loads_weights_only_and_excludes_training_state(self):
        export_model(self.checkpoint, self.output)
        self.assertEqual({p.name for p in self.output.iterdir()}, set(BUNDLE_FILES))
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(set(config), {"model", "code_format"})
        self.assertEqual(config["code_format"], "full")
        self.assertNotIn(
            "must-not-be-exported", (self.output / "config.json").read_text()
        )
        state = torch.load(
            self.output / "model.pt", map_location="cpu", weights_only=True
        )
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in state.values()))
        model = Seq2SeqTransformer(Seq2SeqTransformerConfig(**config["model"])).eval()
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            kwargs = {
                "input_ids": torch.tensor([[4, 5]]),
                "decoder_input_ids": torch.tensor([[2, 6]]),
            }
            torch.testing.assert_close(
                self.model(**kwargs)["logits"], model(**kwargs)["logits"]
            )
        self.assertEqual(
            (self.output / "tokenizer.json").read_bytes(),
            self.tokenizer_path.read_bytes(),
        )

    def test_wrong_tokenizer_is_rejected_before_export(self):
        self.tokenizer.add_tokens(["extra"])
        self.tokenizer.save(str(self.tokenizer_path))
        with self.assertRaisesRegex(ValueError, "vocabulary sizes differ"):
            export_model(self.checkpoint, self.output)
        self.assertFalse(self.output.exists())

    def test_upload_only_sends_the_bundle(self):
        with patch(
            "sys.argv",
            [
                "export_model.py",
                "--checkpoint",
                str(self.checkpoint),
                "--output",
                str(self.output),
                "--repo-id",
                "YaItco/NLP-2-transformer",
            ],
        ), patch("export_model.HfApi") as api:
            main()
        api.return_value.create_repo.assert_called_once_with(
            repo_id="YaItco/NLP-2-transformer",
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        self.assertEqual(
            set(api.return_value.upload_folder.call_args.kwargs["allow_patterns"]),
            set(BUNDLE_FILES),
        )

    def test_notebook_preserves_test_order_with_sorted_batches(self):
        export_model(self.checkpoint, self.output)
        notebook = nbformat.read(ROOT_PATH / "notebooks/inference.ipynb", as_version=4)
        nbformat.validate(notebook)
        scope = {}
        questions = [{"question": "b a"}, {"question": "a"}, {"question": "c a a"}]
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(self.output)
        ), patch("datasets.load_dataset", return_value=questions), patch(
            "torch.cuda.synchronize"
        ), patch(
            "torch.autocast", return_value=nullcontext()
        ), patch.object(
            Seq2SeqTransformer,
            "generate",
            lambda self, input_ids, **kwargs: input_ids[:, :1],
        ):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                exec(compile(cell.source, "inference.ipynb", "exec"), scope)
                scope.update(
                    DEVICE="cpu",
                    BATCH_SIZE=2,
                    OUTPUT=str(self.root / "predictions.json"),
                )
        self.assertEqual(
            json.loads((self.root / "predictions.json").read_text()), ["b", "a", "c"]
        )


if __name__ == "__main__":
    unittest.main()
