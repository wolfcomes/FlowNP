# FlowNP

FlowNP is a research codebase for molecular generation with flow-based neural processes. It includes molecule-only generation, protein-pocket conditioned generation, preprocessing utilities, docking/evaluation scripts, and experiment configuration files.

## Model Overview

![FlowNP model architecture](assets/model_architecture.png)

## Repository Layout

```text
configs/                 Training configuration files
src/                     Core model, data processing, analysis, and utility modules
scripts/                 Evaluation, docking, scoring, plotting, and analysis scripts
process_*.py             Dataset preprocessing entry points
train.py                 Molecule-only training entry point
train_pocket.py          Protein-pocket conditioned training entry point
test.py                  Molecule-only sampling entry point
test_pocket.py           Protein-pocket conditioned sampling entry point
environment.yml          Conda environment exported from the local flowmol environment
assets/                  README assets
```

Local datasets, processed tensors, checkpoints, generated molecules, docking outputs, and evaluation artifacts are excluded from git.

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate flownp
```

The environment was exported from the local `flowmol` conda environment and renamed to `flownp` for this repository. It includes the CUDA-enabled PyTorch/DGL stack used by the current experiments.

Some evaluation scripts require external command-line tools or optional packages, including `smina`, Open Babel, ODDT, Biopython, and MDAnalysis. Install these according to the needs of the specific evaluation workflow.

## Data And Checkpoints

Large artifacts are not tracked in this repository. This includes:

- raw datasets
- processed `.pt` tensors
- trained model checkpoints
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

Datasets and pretrained checkpoints are available from Zenodo:

https://zenodo.org/records/20893299

After downloading the archive, place files under the local paths expected by the config files, for example:

```text
data/coconut_kekulized/
data/crossdock_kekulized/
checkpoints_coconut/
checkpoints_crossdock/
```

## Training

Molecule-only training:

```bash
python train.py --config configs/coconut_ctmc.yaml
python train.py --config configs/qm9_ctmc.yaml
```

Protein-pocket conditioned training:

```bash
python train_pocket.py --config configs/crossdock_ctmc.yaml
```

The config files reference local processed data directories under `data/` and write checkpoints under `checkpoints_*`. These paths are ignored by git.

## Sampling

Sample from a molecule-only checkpoint:

```bash
python test.py \
  --model_dir checkpoints_coconut/<run-name> \
  --output_file results/sample.sdf \
  --n_mols 100 \
  --n_timesteps 250
```

Sample from a protein-pocket conditioned checkpoint:

```bash
python test_pocket.py \
  --checkpoint checkpoints_crossdock/<run-name>/checkpoints/last.ckpt \
  --pdb data/crossdock_example_pockets/<protein-pocket>.pdb \
  --output_dir results/pocket_samples \
  --n_mols_per_protein 10 \
  --n_timesteps 250
```

Use `--pdb` for a single protein-pocket PDB file, or `--pdb_dir` for a directory of PDB files. Optional reference ligands can be provided with `--ref_ligands_dir` when ligand-centered pocket alignment is needed.

Adjust paths and command-line options for the local experiment setup.

## Evaluation

The `scripts/` directory contains utilities for docking, molecular property analysis, NP/SA scoring, scaffold analysis, similarity analysis, and t-SNE visualization.

Examples:

```bash
python scripts/analyze_molecule_properities.py \
  --input_files results/sample.csv results/reference.csv \
  --smiles_cols SMILES smiles \
  --output_dir evaluation_results/molecule_properties \
  --n_molecules 1000

python scripts/docking_multi.py
```

`analyze_molecule_properities.py` computes molecular property tables and distribution plots from one or more CSV files. Docking scripts expect local receptor/ligand files and an available `smina` executable.

## Repository Hygiene

Before committing new work, check:

```bash
git status --short
git diff --stat
```

Do not commit local datasets, processed tensors, checkpoints, generated molecules, docking outputs, or evaluation artifacts. Use an external archive such as Zenodo for those files.

## Citation

Citation information will be added when the associated manuscript or archive is available.
