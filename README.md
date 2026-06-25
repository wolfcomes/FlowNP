# FlowNP

FlowNP is a research codebase for molecular generation with flow-matching style models. The repository contains training entry points, model code, data-processing scripts, configuration files, and evaluation utilities for molecule and protein-pocket conditioned experiments.

This GitHub repository is intended to track source code and lightweight project metadata only. Local datasets, processed tensors, checkpoints, generated molecules, docking outputs, and evaluation artifacts are intentionally excluded.

## Repository Structure

```text
configs/                 Training configuration files
src/                     Core model, data-processing, analysis, and utility modules
scripts/                 Evaluation, docking, plotting, scoring, and data utility scripts
process_*.py             Dataset preprocessing entry points
train.py                 Molecule-only training entry point
train_pocket.py          Protein-pocket conditioned training entry point
test.py                  Molecule generation / sampling entry point
test_pocket.py           Protein-pocket conditioned generation entry point
docs/plans/              Repository preparation and cleanup notes
```

## Data And Artifacts

The repository does not include:

- raw datasets
- processed `.pt` tensors
- model checkpoints
- generated `.sdf` files
- docking outputs
- evaluation result tables and figures

Expected local artifact directories include:

```text
data/
checkpoints_coconut/
checkpoints_crossdock/
results/
evaluation_results/
figures/
```

Keep these files locally or host them separately if they need to be shared.

## Environment

The project depends on a scientific Python and molecular modeling stack. The exact package versions should match the training environment used for the experiments.

Core dependencies include:

- Python
- PyTorch
- DGL
- PyTorch Lightning
- RDKit
- torch-scatter
- NumPy
- pandas
- SciPy
- scikit-learn
- PyYAML
- tqdm
- einops
- matplotlib
- seaborn

Some evaluation and docking scripts also use external tools or optional packages such as `smina`, Open Babel, ODDT, Biopython, and MDAnalysis.

## Training

Run molecule-only training with a config file:

```bash
python train.py --config configs/coconut_ctmc.yaml
python train.py --config configs/qm9_ctmc.yaml
```

Run protein-pocket conditioned training:

```bash
python train_pocket.py --config configs/crossdock_ctmc.yaml
```

The config files reference local processed data directories under `data/` and write checkpoints under `checkpoints_*`. Those paths are ignored by git.

## Sampling

Sample from a trained molecule-only checkpoint:

```bash
python test.py --model_dir checkpoints_coconut/<run-name> --output_file results/sample.sdf
```

Sample with the protein-pocket conditioned entry point:

```bash
python test_pocket.py --checkpoint checkpoints_crossdock/<run-name>/checkpoints/last.ckpt
```

Adjust paths and command-line options for the local experiment setup.

## Evaluation Utilities

The `scripts/` directory contains utilities for docking, molecular property analysis, NP/SA scoring, scaffold analysis, similarity analysis, and loss-curve plotting. For example:

```bash
python scripts/plot_loss_curves.py --knn-root checkpoints_coconut
python scripts/docking_multi.py
```

Docking scripts expect local receptor/ligand files and an available `smina` executable.

## Repository Hygiene

Before committing new work, check:

```bash
git status --short
git diff --stat
```

Large experiment artifacts should stay out of git. If model weights or datasets need to be shared later, use Git LFS, GitHub Releases, Hugging Face, or another external storage service.
