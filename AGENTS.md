# Repository Guidelines

## Project Structure & Module Organization
This repository aggregates four spiking-transformer research codebases as Git submodules: `MaxFormer/`, `QKFormer/`, `spikformer/`, and `Spikingformer/`. Treat each top-level directory as an independent experiment family with its own README, model files, dataset folders, and launch scripts.

Common layout patterns:
- `*/imagenet/`: ImageNet training, evaluation, configs, logs, and checkpoint helpers.
- `*/cifar10`, `*/cifar100`, `*/cifar10-100`: CIFAR training code and YAML configs.
- `*/cifar10-dvs`, `*/cifar10dvs`, `*/dvs128-gesture`, `*/event`: neuromorphic dataset pipelines.
- `imgs/`, `images/`, `origin_logs/`, `logs/`, and `checkpoints/`: paper assets, reference logs, and model artifacts.

## Build, Test, and Development Commands
Create an isolated Python environment per subproject because dependency versions differ.

```bash
cd MaxFormer
pip install -r requirements.txt
```

For projects without `requirements.txt`, install the versions listed in that project README, especially `torch`, `timm`, `cupy`, `spikingjelly`, and `pyyaml`.

Typical runs:

```bash
cd QKFormer/cifar10 && python train.py
cd spikformer/imagenet && python -m torch.distributed.launch --nproc_per_node=8 train.py
cd Spikingformer/imagenet && python test.py
cd MaxFormer/imagenet && bash train.sh
```

Prefer the local `run.sh`, `train.sh`, or `test.sh` files when present; they capture expected arguments and paths.

## Coding Style & Naming Conventions
The code is Python/PyTorch research code. Use 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for `torch.nn.Module` classes. Keep dataset-specific entry points named plainly (`train.py`, `test.py`, `model.py`, `utils.py`) to match existing folders. Store experiment parameters in YAML files near the matching training script, for example `MaxFormer/imagenet/conf/10_512_t4.yml`.

## Testing Guidelines
There is no unified test suite in the root. Validate changes by running the smallest affected script first, such as a CIFAR or DVS `python train.py` dry run, before launching ImageNet distributed jobs. For model edits, verify tensor shapes through one forward pass and compare accuracy or energy outputs against the checked-in `logs/` or `origin_logs/` where available.

## Commit & Pull Request Guidelines
The visible root history only contains `first init`, so use clear, imperative commit messages such as `Fix QKFormer CIFAR100 config path` or `Add MaxFormer evaluation notes`. Pull requests should state the affected subproject and dataset, list exact commands run, mention required checkpoint/data paths, and include metric changes or log snippets for training, evaluation, or energy-calculation changes.

## Security & Configuration Tips
Do not commit datasets, large checkpoints, private download credentials, or WandB API keys. Keep local paths and credentials in ignored shell wrappers or environment variables, and document reproducible public setup steps in the relevant README.
