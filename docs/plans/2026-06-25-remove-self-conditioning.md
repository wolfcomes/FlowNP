# Remove Self-Conditioning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove self-conditioning code paths and configuration entry points from `FlowNP`.

**Architecture:** Add a focused regression test that scans the source/config surface for removed self-conditioning tokens, then remove the feature implementation, parameter plumbing, and checked-in config keys. Historical local checkpoint metadata stays untouched because it is excluded from the intended GitHub repository.

**Tech Stack:** Python, unittest, PyYAML-style configuration files, git hygiene checks

---

### Task 1: Add cleanup regression test

**Files:**
- Create: `test_repository_cleanup.py`

**Step 1: Write the failing test**

Create a `unittest` test that scans:
- `src/`
- `configs/`
- top-level Python files

The test should fail if it finds:
- `self_conditioning`
- `self-conditioning`
- `SelfConditioning`
- `prev_dst_dict`

Exclude generated artifact directories and `docs/plans/`, because the design docs intentionally mention removed terms.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_repository_cleanup.py -v`

Expected: FAIL because the current source/config surface still contains self-conditioning references.

### Task 2: Remove source code paths

**Files:**
- Modify: `src/models/vector_field.py`
- Delete: `src/models/self_conditioning.py`

**Step 1: Remove imports and constructor plumbing**

Remove:
- `from src.models.self_conditioning import SelfConditioningResidualLayer`
- the `self_conditioning` constructor argument
- `self.self_conditioning`
- the conditional residual layer initialization block

**Step 2: Remove forward-pass self-conditioning logic**

Remove:
- `prev_dst_dict` from `CTMCVectorField.forward`
- the two self-conditioning conditional blocks
- `prev_dst_dict` from `step`
- the matching call arguments in sampling and stepping

**Step 3: Delete the residual layer module**

Delete `src/models/self_conditioning.py`.

### Task 3: Remove config and CLI entry points

**Files:**
- Modify: `src/model_utils/sweep_config.py`
- Modify: `configs/qm9_ctmc.yaml`
- Modify: `configs/coconut_ctmc.yaml`
- Modify: `configs/crossdock_ctmc.yaml`

**Step 1: Remove CLI argument**

Delete `p.add_argument('--self_conditioning', ...)`.

**Step 2: Remove config merge entry**

Delete `self_conditioning` from the vector-field override list.

**Step 3: Remove YAML keys**

Delete `self_conditioning: false` from each checked-in config file.

### Task 4: Verify cleanup

**Files:**
- Test: `test_repository_cleanup.py`

**Step 1: Run cleanup test**

Run: `python -m unittest test_repository_cleanup.py -v`

Expected: PASS.

**Step 2: Run syntax checks**

Run:

```bash
python -m py_compile src/models/vector_field.py src/model_utils/sweep_config.py
```

Expected: exit code 0.

**Step 3: Scan final references**

Run:

```bash
rg -n "self[_-]?condition|SelfConditioning|prev_dst_dict" src configs *.py
```

Expected: no matches.
