---
name: research-service-directory
description: Use optional advanced or paid service providers during research mode. Use when modifying or running research that may need mpp.dev/Tempo, AgentCash, Orthogonal, Tako, live prediction-market odds, Product Hunt/Hacker News launch discovery, or recent startup funding data.
---

# Research Service Directory

Use this skill when a research-mode task may need advanced, live, structured, or
paid data sources. The provider directory lives in `services/`.

## Core Rule

Advanced services are discretionary research tools. They should not always be
used. Prefer local corpus, deterministic fixtures, and ordinary retrieval unless
fresh or premium data would materially improve the answer.

## When To Use

Consider the service directory when the user asks for:

- `mpp.dev`, Tempo, AgentCash, Orthogonal, paid APIs, x402, MPP, or wallet-backed
  API calls.
- Live prediction-market odds, onchain market data, or current market state.
- Tako-style data search, charts, insights, or embeddable knowledge cards.
- Recent startup launches, Product Hunt activity, Hacker News activity, or new
  startup discovery.
- Recent startup funding, financing events, investment activity, funding rounds,
  or location/funding-type-filtered funding research.
- Enrichment, scraping, social data, company/person/contact intelligence, or
  other sources that ordinary retrieval is unlikely to structure well.

## Provider Order

1. Check `services/README.md` for policy and provider selection.
2. Use `services/mpp-dev.md` for Tempo and MPP service discovery.
3. Use `services/agentcash.md` for AgentCash and x402/MPP pay-per-call APIs.
4. Use `services/orthogonal.md` when a curated Orthogonal skill or API
   marketplace entry may fit.

## Research Artifact Requirements

When a paid or advanced service is used, preserve:

- Provider, service ID or origin, endpoint path, method, timestamp, and price
  when available.
- Request parameters needed to reproduce the result, excluding secrets.
- Response-derived source records in `sources.json`.
- Claims tied to those source IDs in `claims.json`.
- Caveats when the data is live, paid, sampled, delayed, or otherwise not fully
  reproducible.

## Cost Discipline

- Check readiness and balance before paid calls.
- Discover service details and endpoint schemas before calling.
- Use dry run for ambiguous or potentially expensive requests.
- Stop and report clearly on auth, funding, balance, or spending-limit blockers.

