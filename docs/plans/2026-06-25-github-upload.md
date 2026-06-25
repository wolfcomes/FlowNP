# FlowNP GitHub Upload Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prepare `FlowNP` for an initial private GitHub upload with only source code, configs, scripts, and lightweight documentation tracked by git.

**Architecture:** Add repository hygiene files first, then initialize version control and verify that large local artifacts remain excluded. Keep all local research outputs in place but ignored, so the first repository history stays clean and pushable.

**Tech Stack:** Git, Markdown, Python project layout, shell verification commands

---

### Task 1: Add repository ignore rules

**Files:**
- Create: `.gitignore`

**Step 1: Write the ignore file**

Add ignore rules for:
- Python caches and editor metadata
- virtual environments
- local datasets and processed data
- checkpoint directories
- generated results and evaluation outputs
- binary artifact types such as `.pt`, `.ckpt`, `.pkl`, `.sdf`, logs, and temporary files

**Step 2: Verify the ignore file content**

Run: `sed -n '1,220p' .gitignore`
Expected: ignore patterns cover the current project artifact directories and file types

**Step 3: Commit checkpoint**

Run:

```bash
git add .gitignore
git diff --cached -- .gitignore
```

Expected: only `.gitignore` is staged at this point if git is already initialized

### Task 2: Add a minimal repository README

**Files:**
- Create: `README.md`

**Step 1: Write the README**

Include:
- project name and short description
- key directories
- environment and dependency notes
- training entry points
- docking/evaluation script note
- note that data, checkpoints, and results are not included in the repository

**Step 2: Verify the README content**

Run: `sed -n '1,260p' README.md`
Expected: the README is clear, concise, and aligned with the current repository shape

**Step 3: Commit checkpoint**

Run:

```bash
git add README.md
git diff --cached -- README.md
```

Expected: only README changes are visible in the staged diff

### Task 3: Initialize local git metadata

**Files:**
- Modify: repository metadata only

**Step 1: Initialize the repository**

Run:

```bash
git init
git branch -M main
```

Expected: a new local git repository exists with `main` as the current branch

**Step 2: Verify repository status**

Run:

```bash
git status --short
git branch --show-current
```

Expected:
- current branch is `main`
- only source files, docs, `.gitignore`, and `README.md` are visible as untracked or staged
- ignored large asset directories do not appear in `git status`

### Task 4: Create the initial commit

**Files:**
- Track: source files, configs, scripts, docs, tests, `.gitignore`, `README.md`

**Step 1: Stage the intended repository content**

Run: `git add .`

**Step 2: Verify staged files before commit**

Run:

```bash
git status --short
git diff --cached --stat
```

Expected:
- no dataset directories or checkpoint directories are staged
- staged content is limited to source, configs, scripts, and docs

**Step 3: Create the initial commit**

Run:

```bash
git commit -m "chore: initialize repository for GitHub"
```

Expected: initial commit succeeds locally

### Task 5: Prepare push instructions

**Files:**
- No file changes required

**Step 1: Verify local history**

Run:

```bash
git log --oneline -n 3
git status --short
```

Expected:
- the initial commit is visible
- working tree is clean

**Step 2: Provide remote setup commands**

Document the commands for either SSH or HTTPS remote setup, for example:

```bash
git remote add origin git@github.com:<username>/FlowNP.git
git push -u origin main
```

Expected: the user can create a private GitHub repository and push this local history safely
