---
name: branding-generator
description: >
  Creates branding kits, article covers, headers, social graphics, Twitter/X
  images, thumbnails, and visual identity assets in this repo. Use when
  generating or editing cover images, social graphics, header images, branding,
  or other visual identity work.
---

# Branding Generator

## What this skill does

Creates a small branding kit for a research artifact:
- article cover (square)
- social graphic (portrait or 16:9)
- header image (wide)

Supports text overlay with configurable typography and layout rules. When
generation is needed, prefer MPP image-gen endpoints (Tempo Wallet) before any
scraping or custom adapters.

## Visual constraints (house style)

- Primary: clean white background (`#FFFFFF`) by default; avoid cream/off-white
  unless the user explicitly asks for it.
- Accent: warm rust/burnt orange around `#D97843` sparingly. Approved variants
  are a more futuristic vivid orange and a dark red accent, still used sparingly
  and never as a dominant field unless requested.
- Common DNA: technical drawing precision, generous negative space, muted
  neutrals, mixed media (geometry + photo/texture + annotation), avoid AI glossy
  look.

## Reference set

This skill includes 8 reference images under `references/` when available
(originally generated as internal style anchors; study aesthetic logic, do not
copy).

## Workflow

1. Choose format(s): cover / header / social.
2. Choose motif from references:
   - geometric grayscale composition
   - architectural site plan
   - cartographic abstraction
   - linear network diagram
   - collage / cut-paper
   - modernist scrapbook
   - lunar/celestial (target accent color)
   - generative line art (instrument feel)
3. If generation required:
   - Discover MPP services from `https://mpp.dev/services/llms.txt`.
   - Prefer `fal.mpp.tempo.xyz` (Flux) or `stablestudio.dev` via Tempo.
   - Use dry-run first; set `--max-spend` if user gave a budget.
4. Apply typography overlay:
   - Title large, subtitle small, optional issue/date.
   - Keep text inside safe margins (10-12% padding).
   - Use accent only for 1-2 elements (line, dot, small tag).

## Typography parameters

- `font_family`: `system-sans` | `serif` | `mono`
- `title_case`: `as-is` | `upper` | `sentence`
- `weight`: 400-800
- `tracking`: -2 to +80 (css-style)
- `alignment`: left | center
