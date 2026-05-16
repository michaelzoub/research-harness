# AgentCash

Use AgentCash for x402 and MPP pay-per-call APIs, especially when the task maps
to Stable* services or requires paid search, enrichment, scraping, social data,
uploads, email, or other protected APIs.

## Setup

Upstream skill: `https://agentcash.dev/skill.md`

Onboarding command used:

```bash
npx agentcash@latest onboard AC-PF5B-L28V-P8ET-KJY3
```

Current local status:

- AgentCash skill installed at `/Users/michaelzoubkoff/.agents/skills/agentcash`.
- MCP configured for Codex, Cursor, Claude Code, and Claude Desktop.
- Invite redemption returned `conflict`: the invite code was already redeemed.
- Balance observed after setup: `5.604264`.

Useful commands:

```bash
npx agentcash@latest balance
npx agentcash@latest accounts
npx agentcash@latest search <query>
npx agentcash@latest discover <origin>
npx agentcash@latest fetch <url>
npx agentcash@latest fund
```

## Research Triggers

Consider AgentCash when research needs:

- People, organization, contact, LinkedIn, or company enrichment.
- Paid web search, scraping, or structured extraction.
- Social/platform data from TikTok, Instagram, Facebook, Reddit, LinkedIn, or
  similar sources.
- Recent startup launches from Product Hunt, Hacker News, or adjacent launch
  platforms.
- Recent startup funding and financing events.
- File upload/hosting, email, phone, travel, or other paid API workflows.

## Useful Searches

```bash
npx agentcash@latest search "product hunt hacker news"
npx agentcash@latest search "startup funding"
npx agentcash@latest search "recent funding rounds"
npx agentcash@latest search "prediction market odds"
```

Recent discoveries include:

- Product Hunt launch endpoints such as `/api/producthunt` and `/scout/ph`.
- Hacker News search/top-story endpoints such as `/scout/hn` and
  `/api/hn/search`.
- Funding endpoints such as `/v1/market/recent-funding`,
  `/funding/v1/rounds`, and related funding signal endpoints.

## Rules

- Search or discover before fetch; do not guess paid endpoint schemas.
- Check balance before paid workflows.
- Prefer the core AgentCash skill instructions when available because endpoint
  surfaces change.
- Record paid request outputs as research sources with timestamps and price
  metadata when available.

