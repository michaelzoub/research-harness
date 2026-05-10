---
name: update-architecture
description: Update architecture documentation in docs/architecture.md to reflect changes to the three-loop structure, product agents, PRD format, loop modes, or artifact flow. Use whenever the orchestrator, loops, schemas, or store change in a way that affects the documented architecture.
---

# Update Architecture Docs

Use this skill when the project architecture changes: new loop modes, new product agents, PRD format changes, store schema changes, or flow changes to how stories are picked and executed.

## Files to Update

- `docs/architecture.md` — primary architecture reference (mermaid diagrams + prose)
- `docs/gen_overview_diagram.py` — sequence diagram generator (re-run to emit SVG)
- `skills/research-agent-architecture/SKILL.md` — skill-level invariants

## Three-Loop Architecture

Keep this mental model current in all docs:

```text
Outer loop  — Session (sessions.py)
              Manages context isolation and parallel agent runs.
              Resets state between runs so each agent starts clean.

Middle loop — EvolutionaryOuterLoop (loops.py)
              Proposes and evaluates variants across N outer iterations.
              Drives research (query variants → retrieve → score) or
              optimize (code variants → evaluator → score).

Inner loop  — Ralph loop (orchestrator._run_loop + agent harness)
              The agent harness: model + loop policy + tools + store.
              Picks next story (passes: false) from prd.json.
              Executes it via research or optimization agent.
              Updates prd.json (passes: true) and appends progress.txt.
              Repeats until all stories pass or iteration budget exhausted.
```

## PRD User Story Format

Every task in `organized_tasks` must follow this schema:

```json
{
  "id": "US-001",
  "title": "Descriptive story title",
  "acceptanceCriteria": [
    "Concrete criterion A",
    "Concrete criterion B"
  ],
  "passes": false
}
```

Key rules:
- `id` uses `US-NNN` format (not `prd_task_NNN`)
- `acceptanceCriteria` is camelCase
- `passes` starts `false`; the loop sets it `true` when the story completes
- prd.json is refreshed after **each** story completes (not only at run end)

## Checklist When Updating Architecture

1. Update the mermaid diagram in `docs/architecture.md` to match the new flow.
2. Update the prose sections (Product Agent Details, Optimization Output Contract) if the behavior changed.
3. Re-run `python docs/gen_overview_diagram.py` to regenerate the SVG.
4. Update `skills/research-agent-architecture/SKILL.md` if any invariants changed.
5. Verify `organized_tasks` in a real `prd.json` matches the US-NNN / acceptanceCriteria format.
