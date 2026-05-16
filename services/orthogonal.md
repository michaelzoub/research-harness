# Orthogonal

Use Orthogonal when curated skills or API marketplace entries may provide a
better tested workflow than a generic paid API call.

## Setup

Upstream skill: `https://orthogonal.com/skill.md`

Current local status:

- CLI installed globally as `orth`.
- Verified version: `0.2.0`.
- Authentication still required: run `orth login` with an API key from
  `https://orthogonal.com/dashboard/settings/api-keys`.
- The upstream skill mentions `orth account`; the installed CLI exposes
  `orth balance` instead.

Useful commands:

```bash
orth login
orth whoami
orth balance
orth skills search "<task>"
orth skills add <owner/slug>
orth api search "<task>"
orth api show <api-slug> <path>
orth run <api-slug> <path> --dry-run -b '{"field":"value"}'
orth run <api-slug> <path> -b '{"field":"value"}'
```

## Research Triggers

Consider Orthogonal when:

- A curated skill may exist for the task.
- The task involves enrichment, brand assets, email/contact discovery, data
  extraction, or a third-party API where parameters are easy to get wrong.
- The user explicitly asks to use Orthogonal.

## Decision Flow

1. Check installed skills first.
2. Search Orthogonal skills for the specific task.
3. Install a dedicated skill when one fits.
4. Fall back to `orth api search` only if no skill covers the use case.
5. Run `orth api show <slug> <path>` before the first call.
6. Use dry run before paid calls when supported.

## Rules

- Do not stretch a generic skill over a task that has a dedicated skill.
- Do not guess parameter names.
- If an integration returns an OAuth/connect-account error, report the required
  connection and continue with other sources when possible.

