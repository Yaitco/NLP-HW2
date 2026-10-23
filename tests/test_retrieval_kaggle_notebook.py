"""Validate Kaggle setup without secrets, network, GPU or real training."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from IPython.core.interactiveshell import InteractiveShell
import nbformat
from omegaconf import OmegaConf
import torch

from src.utils.io_utils import ROOT_PATH


class KaggleRetrievalNotebookTests(unittest.TestCase):
    def notebook(self):
        return nbformat.read(
            ROOT_PATH / "notebooks/retrieval_kaggle.ipynb", as_version=4
        )

    def test_cells_compile_after_ipython_transformation_and_have_no_outputs(self):
        notebook = self.notebook()
        nbformat.validate(notebook)
        transformer = InteractiveShell.instance().input_transformer_manager
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(transformer.transform_cell(cell.source), cell.id, "exec")
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
        self.assertIn(
            "from kaggle_secrets import UserSecretsClient", notebook.cells[0].source
        )
        self.assertIn(
            "!gh repo clone https://github.com/Yaitco/NLP-HW2.git",
            notebook.cells[1].source,
        )

    def test_setup_links_input_and_generates_matching_portable_config(self):
        notebook = self.notebook()
        sources = {cell.id: cell.source for cell in notebook.cells}
        with initialize_config_dir(
            config_dir=str(ROOT_PATH / "src/configs"), version_base=None
        ):
            source_config = compose(
                config_name="tinysms_noimports_640_6x6_b64_acc2_1000ep"
            )
        source_config.model.config.update(
            d_model=16, n_heads=2, d_ff=32, encoder_layers=1, decoder_layers=1
        )
        source_config.data.add_imports = False
        source_config.data.clean_questions = False
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            repo, input_root = folder / "repo", folder / "input"
            shutil.copytree(ROOT_PATH / "src/configs", repo / "src/configs")
            (repo / "train.py").touch()
            (input_root / source_config.data.dataset_path).mkdir(parents=True)
            tokenizer_path = input_root / source_config.data.tokenizer_path
            tokenizer_path.write_text("{}")
            (input_root / Path(source_config.datasets.train.indices_path).parent).mkdir(
                parents=True
            )
            checkpoint = input_root / "model_best.pth"
            torch.save(
                dict(
                    config=source_config,
                    epoch=100,
                    state_dict={"weight": torch.zeros(1)},
                ),
                checkpoint,
            )
            scope = {}
            with patch("pathlib.Path.cwd", return_value=repo):
                exec(sources["kaggle-settings"], scope)
            scope.update(INPUT_ROOT=input_root, CHECKPOINT_PATH=checkpoint)
            exec(sources["kaggle-prepare"], scope)
            for relative in (
                source_config.data.dataset_path,
                source_config.data.tokenizer_path,
                str(Path(source_config.datasets.train.indices_path).parent),
            ):
                self.assertTrue((repo / relative).is_symlink())
                self.assertEqual(
                    (repo / relative).resolve(), (input_root / relative).resolve()
                )
            with initialize_config_dir(
                config_dir=str(repo / "src/configs"), version_base=None
            ):
                cfg = compose(config_name="tinysms_retrieval_kaggle")
            OmegaConf.to_container(cfg, resolve=True)
            self.assertEqual(cfg.model.config.d_model, 16)
            self.assertFalse(cfg.data.add_imports)
            self.assertFalse(cfg.data.clean_questions)
            self.assertEqual(cfg.trainer.amp.dtype, "auto")
            self.assertFalse(cfg.code_execution.sandbox)
            self.assertTrue(cfg.trainer.compare_greedy)
            self.assertEqual(cfg.trainer.generation.num_beams, 16)
            self.assertTrue(cfg.trainer.validate_before_training)
            self.assertEqual(cfg.trainer.n_epochs, 20)
            self.assertIsNone(cfg.trainer.epoch_len)
            self.assertIsNone(cfg.lr_scheduler)
            self.assertEqual(cfg.optimizer.lr, 1e-5)
            self.assertEqual(
                Path(cfg.trainer.from_pretrained),
                repo / "artifacts/retrieval_val64/initial_model.pth",
            )
            self.assertEqual(
                Path(cfg.datasets.train.indices_path),
                repo / "artifacts/retrieval_val64/train_indices.npy",
            )
            # The setup is idempotent, while unrelated existing files are preserved.
            exec(sources["kaggle-prepare"], scope)
            tokenizer_link = repo / source_config.data.tokenizer_path
            tokenizer_link.unlink()
            tokenizer_link.write_text("keep this local file")
            with self.assertRaises(FileExistsError):
                exec(sources["kaggle-prepare"], scope)
            self.assertEqual(tokenizer_link.read_text(), "keep this local file")


if __name__ == "__main__":
    unittest.main()
