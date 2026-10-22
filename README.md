# PyTorch Template for DL projects

> TinySMS adaptation: see [MIGRATION.md](MIGRATION.md) for the transferred code,
> dataset paths, and the one-batch training command.

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## About

This repository contains a template for [PyTorch](https://pytorch.org/)-based Deep Learning projects.

The template utilizes different python-dev techniques to improve code readability. Configuration methods enhance reproducibility and experiments control.

The repository is released as a part of the [HSE DLA course](https://github.com/markovka17/dla), however, can easily be adopted for any DL-task.

This template is the official recommended template for the [EPFL CS-433 ML Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/).

**New:** we added a [HF Main](https://github.com/Blinorot/pytorch_project_template/tree/hf_main) variant of the template with [HuggingFace](https://huggingface.co/) Integration for multi-GPU and multi-node training, automatic mixed precision, gradient accumulation, and seamless HuggingFace Ecosystem Compatibility.

> 📖 **If you use this template in your work, please cite this repository or include a reference. Attribution supports the project and encourages continued development.**

## Tutorials

This template utilizes experiment tracking techniques, such as [WandB](https://docs.wandb.ai/) and [Comet ML](https://www.comet.com/docs/v2/), and [Hydra](https://hydra.cc/docs/intro/) for the configuration. It also automatically reformats code and conducts several checks via [pre-commit](https://pre-commit.com/). If you are not familiar with these tools, we advise you to look at the tutorials below:

- [Python Dev Tips](https://github.com/ebezzam/python-dev-tips): information about [Git](https://git-scm.com/doc), [pre-commit](https://pre-commit.com/), [Hydra](https://hydra.cc/docs/intro/), and other stuff for better Python code development. The YouTube recording of the workshop is available [here](https://youtu.be/okxaTuBdDuY).

- [Seminar on R&D Coding 2025](https://youtu.be/PE1zaW5it_A): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with discussion on logging, project-based coding, configuration, and reproducibility. The materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/tree/summer25/day05).

- [Seminar on R&D Coding 2024](https://youtu.be/sEA-Js5ZHxU): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with template discussion and reasoning. It also explains how to work with [WandB](https://docs.wandb.ai/). The seminar materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/blob/main/day03/Seminar_WandB_and_Coding.ipynb).

- [HSE DLA Course Introduction Week](https://github.com/markovka17/dla/tree/2024/week01): combines the two seminars above into one with some updates, including an extra example for [Comet ML](https://www.comet.com/docs/v2/).

- [PyTorch Basics](https://github.com/markovka17/dla/tree/2024/week01/intro_to_pytorch): several notebooks with [PyTorch](https://pytorch.org/docs/stable/index.html) basics and corresponding seminar recordings from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/).

To start working with a template, just click on the `use this template` button.

<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>

You can choose any of the branches as a starting point. [Set your choice as the default branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch) in the repository settings. You can also [delete unnecessary branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

## Examples

> [!IMPORTANT]
> The main branch leaves some of the code parts empty or fills them with dummy examples, showing just the base structure. The final users can add code required for their own tasks.

You can find examples of this template completed for different tasks in other branches:

- [HF Main](https://github.com/Blinorot/pytorch_project_template/tree/hf_main): the variant of the `main` branch with [HuggingFace](https://huggingface.co/) Integration. Supports multi-GPU and multi-node training, automatic mixed precision, gradient accumulation, and seamless HuggingFace Ecosystem Compatibility.

- [Image classification](https://github.com/Blinorot/pytorch_project_template/tree/example/image-classification): simple classification problem on [MNIST](https://yann.lecun.com/exdb/mnist/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets.

- [ASR](https://github.com/Blinorot/pytorch_project_template/tree/example/asr): template for the automatic speech recognition (ASR) task. Some of the parts (for example, `collate_fn` and beam search for `text_encoder`) are missing for studying purposes of [HSE DLA course](https://github.com/markovka17/dla).

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

### Export to Hugging Face and run inference

```bash
venv/bin/hf auth login
venv/bin/python export_model.py --repo-id YaItco/NLP-2-transformer
```

The default checkpoint is `saved/strong_full_clean_sqrt_16k_len128_b64/model_best.pth`.
Override it with `--checkpoint PATH`. The script exports `model.pt`, `config.json`,
`tokenizer.json`, and a model card to `artifacts/hf_model`, then uploads only these
files. It takes architecture and tokenizer path from the checkpoint. Optimizer
state, credentials, and training configuration are excluded. New repositories are
public by default; add `--private` to create a private repository. Omit `--repo-id`
to export locally without uploading.

Open `notebooks/inference.ipynb` inside the downloaded project with dependencies
installed and a GPU enabled. It downloads the model from Hugging Face, reads only
the public test split in streaming mode, generates in FP16 with KV-cache, and saves
`prediction.jsonl`, one `{"code": "..."}` object per test example in original order.
Model code is imported from this project. Private HF repositories require authentication in that environment.
The printed time includes tokenization/generation but excludes model/data downloads.

### Compact code targets and a new 16k BPE

```bash
python prepare_data.py --config src/configs/tokenizer_body_16k.yaml
python filter_lengths.py --config src/configs/filter_length_128_body_16k.yaml
```

The preparation command normalizes the full numerically cleaned train (before
the old token-length filter) and the original validation split. It saves a separate
dataset with `original_code` and the new `code`, trains a fresh 16,384-token BPE on
100,000 random train examples, and tokenizes every split. Public test rows remain
without reference code. Existing datasets/tokenizers are preserved.

Code is parsed through AST to remove comments safely, normalized with `ast.unparse`,
and stored without `def solve()` and the final `return result`. Other final return
expressions become `result = expression`; earlier returns inside branches remain.
Body indentation is retained so multiline string values do not change. Unused
functions outside `solve()` are removed. Required module-level helpers or
other unsupported module structure stop preparation with the row index rather
than silently deleting dependencies or dropping rows.

New artifacts: `artifacts/code_body_full_clean`, `artifacts/tokenizer_body_16k.json`,
`artifacts/tokenized_body_full_clean_16k`, and `artifacts/length128_body_16k`.
The last command recomputes the 128-token filter for train only, including EOS in
target lengths. Validation and all 1,919 public test examples remain unfiltered.

For subsequent model training, use the new tokenizer/data/indices, train from
scratch, and set `data.code_format: solve_body_result` so Pass@1 restores the
function wrapper inside the existing sandbox. For exported predictions use
`src.code_preprocessing.restore_code(body)`.

To reuse the normalized dataset after preparation, set `prepare_code: false`.
To reuse an already trained tokenizer, also set `train_tokenizer: false`.

Train from scratch for 100 virtual epochs with batch 128 and no accumulation:

```bash
venv/bin/python train.py -cn=tinysms_strong_body_16k_b128_100ep
```

This is 100,000 microbatches/optimizer updates. The learning rate warms up for
1,000 updates to `3e-4`, then follows cosine decay to `3e-5`. Validation loss runs
every 1,000 microbatches and Pass@1 every 20,000 plus the final validation.
The time limit is disabled so the schedule completes; runtime depends on throughput.

For 200 virtual epochs (200,000 updates), use:

```bash
venv/bin/python train.py -cn=tinysms_strong_body_16k_b128_200ep
```

Warmup remains 1,000 updates; cosine spans the remaining 199,000 updates.

The body-code training configs use `dataloader.eval_batch_size: 32` for validation
(both loss and Pass@1) and test. Training batch size is independent. Override it
with `dataloader.eval_batch_size=16`, for example; `null` uses the training batch size.
When resuming a checkpoint, `eval_batch_size` can change; other dataloader settings
must match. This also works with older checkpoints without `eval_batch_size`.

To compare test questions with the checkpoint's filtered training set:

```bash
venv/bin/python analyze_similarity.py --batch-size 512 --chunk-size 8192
```

This uses masked-mean, L2-normalized last-layer encoder vectors and exhaustive
blockwise cosine search. It saves top-5 train neighbors for every test question
to `artifacts/train_test_cosine/nearest.jsonl`, numeric results to `nearest.npz`,
and aggregate statistics to `summary.json`. Train indices refer to rows before
the length filter. `--train-limit` restricts the search for a quick check.

Validation generation in `tinysms_train` and `tinysms_strong_*` uses KV-cache and
beam search. Configure it under `trainer.generation`:

```yaml
max_new_tokens: 512
num_beams: 4       # 1 selects greedy decoding.
length_penalty: 1.0
use_cache: true
```

Beam scores are `sum(log probabilities) / length ** length_penalty`; length
includes EOS and excludes BOS and trailing PAD. Finished beams only append PAD.
The best completed candidate is retained if it leaves the active beam pool.
KV-cache stores decoder self-attention keys/values and the encoder projections
for cross-attention. `use_cache: false` recomputes complete prefixes for comparison.
These options affect Pass@1 generation; teacher-forced validation loss is unchanged.

For numeric voting during validation, use `trainer.generation.num_beams=4`
and `trainer.beam_voting=true`. In `notebooks/inference.ipynb`, set `NUM_BEAMS = 4`
and `BEAM_VOTING = True` (start with `BATCH_SIZE = 16` for four beams).
Voting is off by default. It runs each distinct completed program in the existing
Bubblewrap sandbox, groups finite numeric results with `rtol=1e-5, atol=1e-6`,
and selects the largest group. Ties use the best length-normalized beam score;
the output is the best-scoring code in the winning group. If all programs fail,
the original best beam is returned. References are never used to select a beam.
The inference timer includes voting; the 180-second T4 budget must be measured.
By default, voting uses Bubblewrap and permitted user namespaces, with no automatic
fallback. For Colab experiments without namespaces, the notebook has `SANDBOX = False`:
code runs in a fresh process with time/memory limits and a temporary working directory,
but can access the runtime's files and network. Set `SANDBOX = True` to use Bubblewrap.
For training/validation the equivalent explicit override is `code_execution.sandbox=false`;
this applies to both voting and the Pass@1 metric. Local training defaults to `true`.
With `SANDBOX = False`, voting groups up to `EXECUTION_BATCH_SIZE = 16` programs per
process. Simple arithmetic functions are compiled together as one AST module,
then called with separate globals and individual timeouts. Imports and other
unsupported constructs retain individual-process execution. Set
`EXECUTION_BATCH_SIZE = 1` to compare with the previous implementation; training
uses `code_execution.batch_size` for this setting. Sandboxed execution stays per program.
No weights change or Hugging Face re-upload is needed; use the updated project code.

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Useful Links:

You may find the following links useful:

- [Report branch](https://github.com/Blinorot/pytorch_project_template/tree/report): Guidelines for writing a scientific report/paper (with an emphasis on DL projects).

- [CLAIRE Template](https://github.com/CLAIRE-Labo/python-ml-research-template): additional template by [EPFL CLAIRE Laboratory](https://www.epfl.ch/labs/claire/) that can be combined with ours to enhance experiments reproducibility via [Docker](https://www.docker.com/).

- [Mamba](https://github.com/mamba-org/mamba) and [Poetry](https://python-poetry.org/): alternatives to [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and [pip](https://pip.pypa.io/en/stable/installation/) package managers given above.

- [Awesome README](https://github.com/matiassingers/awesome-readme): a list of awesome README files for inspiration. Check the basics [here](https://github.com/PurpleBooth/a-good-readme-template).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
