---
name: no-predefined-agent-trajectories
description: Prevent hardcoded agent paths. Use whenever modifying agent loops, optimization/recovery policies, source-query generation, challenge adapters, proposal templates, fallback variants, or skills that could add fixed strategies, fixed literature lenses, or predefined trajectories.
---

# No Predefined Agent Trajectories

The harness must not smuggle in a fixed path the agent will always take.

## Core Rule

Never add predefined trajectories, named strategies, literature lenses, or recovery paths unless they are explicitly present in one of:

- the user's prompt
- retrieved sources or stored claims
- evaluator output, score history, or failure traces
- a public challenge interface contract needed only for syntactic compatibility

## Forbidden Patterns

Do not hardcode:

- Always-use literature terms, methods, or named theories.
- Domain-specific strategies that every run tries by default.
- Fallback variants with fixed strategic meaning.
- Recovery policies that inject a canned solution direction.
- Query expansion that assumes a method not requested or retrieved.

## Required Pattern

When a loop needs a query, variant, mutation, or recovery action:

1. Start from the user goal.
2. Add only terms from retrieved evidence, parent variants, score history, or failure traces.
3. If offline fallback needs numeric diversity, derive parameters from context hashes or parent mutation, and label it as context-derived.
4. Record provenance in metadata so the reason for the proposal is auditable.
5. Add a regression test showing unrelated predefined terms do not appear when the prompt/evidence did not contain them.

## Challenge Adapters

Challenge adapters may encode interface requirements, file formats, imports, and valid action schemas. They must not encode a preferred strategy trajectory unless the challenge spec, prompt, retrieved evidence, or evaluator feedback justifies it.
