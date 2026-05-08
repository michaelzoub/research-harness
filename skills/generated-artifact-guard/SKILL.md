---
name: generated-artifact-guard
description: Use before staging, committing, or summarizing git changes in this repo, especially after running the harness, evals, benchmarks, or tests. Prevents generated per-run artifacts under outputs, eval_outputs, or evals from being committed.
---

# Generated Artifact Guard

Generated run and eval artifacts are local evidence, not source.

## Never Commit

Do not stage or commit per-run files or folders under:

- `outputs/`
- `eval_outputs/`
- `evals/`
- `examples/outputs/`
- `logs/`
- `benchmarks/`

Keep only placeholder files such as `.gitkeep` if the repo intentionally tracks
an empty artifact directory.

## Required Check

Before summarizing changes for a commit or staging files, run:

```bash
git status --short --ignored outputs eval_outputs evals examples/outputs logs benchmarks
git ls-files outputs eval_outputs evals examples/outputs logs benchmarks
```

If generated artifact files appear as tracked, remove them from the index without
deleting local files:

```bash
git rm --cached -r eval_outputs outputs evals examples/outputs logs benchmarks
```

Then restore any intended placeholders if needed:

```bash
git add outputs/.gitkeep eval_outputs/.gitkeep evals/.gitkeep
```

Only stage source code, tests, skills, docs, prompts, examples, and small static
fixtures that are intentionally part of the project.
