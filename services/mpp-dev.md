# mpp.dev / Tempo

Use Tempo for MPP service discovery and paid HTTP requests when a research task
needs live or premium structured data.

## Setup

Upstream skill: `https://tempo.xyz/SKILL.md`

Current local status:

- Tempo installed at `$HOME/.tempo/bin/tempo`.
- Verified version: `1.6.0`.
- Wallet status: `ready: true`.
- Wallet address: `0xf54c88f469480c9faf53fcc81a5c4a36232379d2`.
- Tempo network key address:
  `0xbc69b9d4e8393415cdaa3c2be7d0dd0a4c8d1c20`.
- Spending limit observed: `100.000000 USDC.e`, remaining `100.000000`.

Use the full path in commands:

```bash
"$HOME/.tempo/bin/tempo" wallet -t whoami
"$HOME/.tempo/bin/tempo" wallet -t services --search <query>
"$HOME/.tempo/bin/tempo" wallet -t services <SERVICE_ID>
"$HOME/.tempo/bin/tempo" request -t --dry-run -X POST --json '{"input":"..."}' <SERVICE_URL>/<ENDPOINT_PATH>
"$HOME/.tempo/bin/tempo" request -t -X POST --json '{"input":"..."}' <SERVICE_URL>/<ENDPOINT_PATH>
```

## Research Triggers

Consider Tempo when research needs:

- Live prediction-market or onchain market data.
- Tako data search, visualization, charts, or research reports.
- Paid web scraping, structured extraction, maps, finance, crypto, travel, or
  other marketplace APIs.
- A service marketplace search before choosing a paid source.

## Useful Discovered Services

- `tako`: data visualization and research platform for dataset search, charts,
  and AI research reports.
- `codex`: onchain data API with token and prediction-market data.
- `nansen`: blockchain analytics, smart-money intelligence, and prediction
  market tagged data.
- `stableenrich`: people, company, web search, scraping, places, social media,
  and contact enrichment.
- `firecrawl`: web scraping, crawling, and structured extraction.

## Rules

- Always run service discovery before requests.
- Build URLs only from discovered `service_url` plus endpoint path.
- Use `--dry-run` before calls with unclear cost.
- If an endpoint returns schema errors, inspect service details or linked docs
  before retrying.

