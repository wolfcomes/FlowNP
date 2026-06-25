# FlowNP GitHub Upload Design

**Date:** 2026-06-25

## Goal

Prepare `FlowNP` for its first GitHub upload as a private repository under a personal account, with a clean source-only history.

## Scope

The repository should include source code, scripts, configs, lightweight tests, and project documentation.

The repository should exclude:
- local datasets
- processed tensors
- checkpoints
- generated results
- evaluation outputs
- caches and temporary files

## Constraints

- The current project directory is not yet a git repository.
- The project contains many large files, including `.pt`, `.ckpt`, `.pkl`, and generated outputs.
- We should not delete any local research assets; we only need to exclude them from version control.
- The repository will be created under the user's personal GitHub account and should start as `private`.

## Recommended Approach

Use a source-only repository bootstrap:

1. Add a `.gitignore` tailored to the current project layout and artifact types.
2. Add a minimal `README.md` that explains what the project is, what is included, and how to run the main entry points.
3. Initialize git locally on the `main` branch.
4. Verify that ignored data and checkpoints are not staged.
5. Create the first local commit.
6. Leave remote creation and push as the final manual or assisted step after local verification.

## Why This Approach

- It avoids GitHub file size and repository bloat problems on the first push.
- It preserves the user's local experimental assets.
- It creates a repository that is easy to maintain, share, and later make public if desired.
- It keeps future options open for sharing model weights through Git LFS or external hosting without polluting the main code history.

## Repository Shape

Expected tracked content:
- `src/`
- `scripts/`
- `configs/`
- `train.py`
- `train_pocket.py`
- `process_*.py`
- lightweight test files
- `README.md`
- `docs/plans/`

Expected ignored content:
- `data/`
- `checkpoints_coconut/`
- `checkpoints_crossdock/`
- `results/`
- `evaluation_results/`
- cache folders and generated binary artifacts

## Success Criteria

The work is successful when:

1. The project has a valid `.gitignore` for current local artifacts.
2. The project has a usable top-level `README.md`.
3. A local git repository exists on branch `main`.
4. `git status` shows only source and documentation files staged or committed.
5. No large local datasets or checkpoints are included in tracked files.
