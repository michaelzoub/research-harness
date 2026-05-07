---
name: agent-skill-authoring
description: Create, update, or validate Agent Skills in this repo according to the Agent Skills specification. Use when editing skills, SKILL.md frontmatter, descriptions, references, scripts, assets, or repo-local skill validation tests.
---

# Agent Skill Authoring

Use this skill when adding or editing repo-local skills.

## Source Of Truth

Fetch the documentation index before exploring Agent Skills docs:

```text
https://agentskills.io/llms.txt
```

The key spec page is:

```text
https://agentskills.io/specification
```

## Required Structure

Each skill is a directory whose name matches its `name` field:

```text
skill-name/
  SKILL.md
  scripts/       optional
  references/    optional
  assets/        optional
```

`SKILL.md` must start with YAML frontmatter containing at least:

```yaml
---
name: skill-name
description: What the skill does and when to use it.
---
```

## Frontmatter Rules

- `name` is required, 1-64 characters.
- `name` uses lowercase letters, numbers, and hyphens only.
- `name` must not start/end with a hyphen or contain `--`.
- `name` must match the parent directory.
- `description` is required, non-empty, and at most 1024 characters.
- Descriptions should include trigger keywords and when to use the skill.

## Writing Rules

- Keep `SKILL.md` concise and project-specific.
- Put always-needed gotchas in `SKILL.md`.
- Put large optional details in `references/` and mention exactly when to read
  them.
- Do not add auxiliary README or changelog files inside a skill.
- Prefer concrete contracts, commands, and examples over generic advice.

## Validation

If `skills-ref` is available, run:

```bash
skills-ref validate ./skills/<skill-name>
```

If not, run the repo tests that validate local skill frontmatter.

