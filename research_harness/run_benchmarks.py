from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import struct
import tempfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .store import ArtifactStore

# ── Role colour + label maps ──────────────────────────────────────────────────
_ROLE_COLORS: dict[str, str] = {
    "search_literature":         "#3b82f6",
    "hypothesis_generation":     "#8b5cf6",
    "critic_reviewer":           "#f59e0b",
    "synthesis_agent":           "#10b981",
    "harness_debugger":          "#6b7280",
    "task_router":               "#ec4899",
    "plateau_recovery_policy":   "#f97316",
    "literature_grounding_policy": "#06b6d4",
    "research_variant_agent":    "#2563eb",
    "optimize_evaluator":        "#dc2626",
    "llm_thinking":              "#7c3aed",
    "loop_controller":           "#14b8a6",
    "orchestration":             "#64748b",
    "memory":                    "#0f766e",
}
_DEFAULT_ROLE_COLOR = "#94a3b8"

# Harness bookkeeping traces — omitted from agent_timeline.png so the chart shows
# model-backed agents and evaluators (wall clock until output is ready).
_TIMELINE_CHART_EXCLUDED_ROLES: frozenset[str] = frozenset(
    {"orchestration", "loop_controller", "memory"}
)

_ROLE_SHORT: dict[str, str] = {
    "search_literature":         "Search",
    "hypothesis_generation":     "Hyp",
    "critic_reviewer":           "Critic",
    "synthesis_agent":           "Synth",
    "harness_debugger":          "Debug",
    "task_router":               "Router",
    "plateau_recovery_policy":   "Plateau",
    "literature_grounding_policy": "Ground",
    "research_variant_agent":    "Research",
    "optimize_evaluator":        "Eval",
    "llm_thinking":              "LLM",
    "loop_controller":           "Loop",
    "orchestration":             "Orch",
    "memory":                    "Memory",
}

_FONT: dict[str, tuple[str, ...]] = {
    " ": ("000","000","000","000","000","000","000"),
    "0": ("111","101","101","101","101","101","111"), "1": ("010","110","010","010","010","010","111"),
    "2": ("111","001","001","111","100","100","111"), "3": ("111","001","001","111","001","001","111"),
    "4": ("101","101","101","111","001","001","001"), "5": ("111","100","100","111","001","001","111"),
    "6": ("111","100","100","111","101","101","111"), "7": ("111","001","001","010","010","010","010"),
    "8": ("111","101","101","111","101","101","111"), "9": ("111","101","101","111","001","001","111"),
    "A": ("010","101","101","111","101","101","101"), "B": ("110","101","101","110","101","101","110"),
    "C": ("111","100","100","100","100","100","111"), "D": ("110","101","101","101","101","101","110"),
    "E": ("111","100","100","110","100","100","111"), "F": ("111","100","100","110","100","100","100"),
    "G": ("111","100","100","101","101","101","111"), "H": ("101","101","101","111","101","101","101"),
    "I": ("111","010","010","010","010","010","111"), "J": ("001","001","001","001","101","101","111"),
    "K": ("101","101","110","100","110","101","101"), "L": ("100","100","100","100","100","100","111"),
    "M": ("101","111","111","101","101","101","101"), "N": ("101","111","111","111","101","101","101"),
    "O": ("111","101","101","101","101","101","111"), "P": ("111","101","101","111","100","100","100"),
    "Q": ("111","101","101","101","111","001","001"), "R": ("110","101","101","110","101","101","101"),
    "S": ("111","100","100","111","001","001","111"), "T": ("111","010","010","010","010","010","010"),
    "U": ("101","101","101","101","101","101","111"), "V": ("101","101","101","101","101","101","010"),
    "W": ("101","101","101","101","111","111","101"), "X": ("101","101","101","010","101","101","101"),
    "Y": ("101","101","101","010","010","010","010"), "Z": ("111","001","001","010","100","100","111"),
    "-": ("000","000","000","111","000","000","000"), "_": ("000","000","000","000","000","000","111"),
    ".": ("000","000","000","000","000","110","110"), ":": ("000","110","110","000","110","110","000"),
    "/": ("001","001","001","010","100","100","100"), "?": ("111","001","001","010","010","000","010"),
    "(": ("001","010","100","100","100","010","001"), ")": ("100","010","001","001","001","010","100"),
    "+": ("000","010","010","111","010","010","000"), "$": ("010","111","100","111","001","111","010"),
    "%": ("101","001","010","010","010","100","101"), ",": ("000","000","000","000","000","010","100"),
}


class _PngCanvas:
    def __init__(self, width: int, height: int, bg: str = "#ffffff"):
        self.width = width
        self.height = height
        self.pixels = bytearray(_rgb(bg) * width * height)

    def rect(self, x: int, y: int, w: int, h: int, color: str) -> None:
        r, g, b = _rgb_tuple(color)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        for yy in range(y0, y1):
            offset = (yy * self.width + x0) * 3
            self.pixels[offset : offset + (x1 - x0) * 3] = bytes([r, g, b]) * (x1 - x0)

    def outline(self, x: int, y: int, w: int, h: int, color: str) -> None:
        self.rect(x, y, w, 1, color)
        self.rect(x, y + h - 1, w, 1, color)
        self.rect(x, y, 1, h, color)
        self.rect(x + w - 1, y, 1, h, color)

    def text(self, x: int, y: int, text: str, color: str = "#0f172a", scale: int = 2, max_chars: Optional[int] = None) -> None:
        if max_chars is not None and len(text) > max_chars:
            text = text[: max(0, max_chars - 3)] + "..."
        cx = x
        for char in text:
            glyph = _FONT.get(char) or _FONT.get(char.upper()) or _FONT.get("?")
            if glyph is None:
                cx += 4 * scale
                continue
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cx + gx * scale, y + gy * scale, scale, scale, color)
            cx += 4 * scale

    def png(self) -> bytes:
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            rows.append(b"\x00" + bytes(self.pixels[y * stride : (y + 1) * stride]))
        raw = b"".join(rows)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(raw, 9))
            + _png_chunk(b"IEND", b"")
        )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _rgb(color: str) -> bytes:
    return bytes(_rgb_tuple(color))


def _rgb_tuple(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _role_color(role: str) -> str:
    return _ROLE_COLORS.get(role, _DEFAULT_ROLE_COLOR)


def _role_short(role: str) -> str:
    return _ROLE_SHORT.get(role, role.replace("_", " ").title()[:8])


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _fmt_tick(ms: int) -> str:
    s = ms // 1000
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    return f"{m}:{rem:02d}"


def _nice_tick_ms(total_ms: int, target_ticks: int = 6) -> int:
    if total_ms <= 0:
        return 1000
    approx = total_ms / target_ticks
    for step in [500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000, 600_000]:
        if approx <= step:
            return step
    return 600_000


def _shorten(text: str, max_len: int = 22) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _human_span_label(raw_name: str, role: str, counts: Counter[str]) -> str:
    counts[role or "unknown"] += 1
    index = counts[role or "unknown"]
    if role == "research_variant_agent":
        return f"Research variant {index}"
    if role == "optimize_evaluator":
        return f"Optimizer eval {index}"
    if role == "llm_thinking":
        round_match = re.search(r"round[_-](\d+)", raw_name)
        if "query" in raw_name:
            return f"LLM query proposal r{round_match.group(1)}" if round_match else f"LLM query proposal {index}"
        if "prediction_market" in raw_name:
            return f"LLM PM code proposal r{round_match.group(1)}" if round_match else f"LLM PM code proposal {index}"
        if "code" in raw_name:
            return f"LLM code proposal r{round_match.group(1)}" if round_match else f"LLM code proposal {index}"
        return f"LLM thinking {index}"
    if role == "loop_controller":
        round_match = re.search(r"round[_-](\d+)", raw_name)
        return f"Continue? r{round_match.group(1)}" if round_match else f"Loop decision {index}"
    if role == "orchestration":
        round_match = re.search(r"round[_-](\d+)", raw_name)
        if "persist" in raw_name:
            return f"Persist variants r{round_match.group(1)}" if round_match else f"Persist variants {index}"
        if "propose" in raw_name:
            return f"Propose variants r{round_match.group(1)}" if round_match else f"Propose variants {index}"
        if "rank" in raw_name:
            return f"Rank/select r{round_match.group(1)}" if round_match else f"Rank/select {index}"
        if "seed" in raw_name:
            return "Build seed context"
        return f"Orchestration {index}"
    if role == "memory":
        return "Memory / PRD"
    if role == "search_literature":
        number = _trailing_number(raw_name)
        return f"Literature search {number}" if number else "Literature search"
    if role == "hypothesis_generation":
        number = _trailing_number(raw_name)
        return f"Hypothesis agent {number}" if number else "Hypothesis agent"
    if role == "critic_reviewer":
        return "Critic review"
    if role == "synthesis_agent":
        return "Synthesis"
    if role == "harness_debugger":
        return "Harness debugger"
    if role == "task_router":
        return "Task router"
    if role == "literature_grounding_policy":
        return "Literature grounding"
    if role == "plateau_recovery_policy":
        return "Plateau recovery"
    return raw_name.replace("_", " ").replace(":", " ").title()


def _trailing_number(text: str) -> Optional[str]:
    match = re.search(r"(?:_|-)(\d+)$", text)
    return match.group(1) if match else None


def _timeline_chart_lane(role: str, human_label: str) -> Optional[str]:
    """Map a trace to a Gantt row lane for the agent-focused chart, or None to omit."""
    if role in _TIMELINE_CHART_EXCLUDED_ROLES:
        return None
    if role == "research_variant_agent":
        return human_label
    if role == "llm_thinking":
        return "LLM"
    if role == "optimize_evaluator":
        return "Optimizer evaluation"
    if role == "search_literature":
        return "Literature search"
    if role == "hypothesis_generation":
        return "Hypothesis"
    if role == "critic_reviewer":
        return "Critic"
    if role == "synthesis_agent":
        return "Synthesis"
    if role == "harness_debugger":
        return "Harness debugger"
    if role == "task_router":
        return "Task router"
    if role == "literature_grounding_policy":
        return "Literature grounding"
    if role == "plateau_recovery_policy":
        return "Plateau recovery"
    return human_label


def _gantt_row_label(span: dict[str, Any]) -> str:
    """Left-axis label: consolidated lane when present, else full span label."""
    return str(span.get("row_label") or span["label"])


def _build_timeline_spans(
    summary: dict[str, Any],
    *,
    for_agent_chart: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Parse agent trace summaries into Gantt spans.

    Returns (spans, num_rows, total_ms).  Each span dict has:
      label, role, status, offset_ms, runtime_ms, end_ms, token_usage, summary, row.
    Rows are assigned by unique agent_name in order of first appearance so
    parallel agents land on separate rows and the chart reads top-to-bottom.

    When for_agent_chart is True, orchestration / loop / memory traces are dropped
    and several roles share one row (e.g. all LLM calls on \"LLM\") so the PNG
    highlights agents and wall-clock time until model output is ready.
    """
    run = summary.get("run") or {}
    run_start = _parse_iso(str(run.get("started_at", "")))
    traces = summary.get("trace_summaries") or []

    spans: list[dict[str, Any]] = []
    cursor_ms = 0

    for trace in traces:
        runtime_ms = max(int(trace.get("runtime_ms") or 0), 0)
        started_dt = _parse_iso(str(trace.get("started_at") or ""))

        if started_dt and run_start:
            offset_ms = max(0, int((started_dt - run_start).total_seconds() * 1000))
        else:
            # Sequential fallback when no wall-clock start is stored.
            offset_ms = cursor_ms

        end_ms = offset_ms + runtime_ms
        cursor_ms = max(cursor_ms, end_ms)

        raw_label = str(trace.get("agent_name") or "unknown")
        spans.append({
            "label":       raw_label,
            "raw_label":   raw_label,
            "role":        str(trace.get("role") or ""),
            "status":      str(trace.get("status") or ""),
            "model":       str(trace.get("model") or ""),
            "offset_ms":   offset_ms,
            "runtime_ms":  runtime_ms,
            "end_ms":      end_ms,
            "token_usage": int(trace.get("token_usage") or 0),
            "summary":     str(trace.get("summary") or "")[:120],
            "row":         0,
        })

    spans.sort(key=lambda s: (s["offset_ms"], s["label"]))
    label_counts: Counter[str] = Counter()
    for span in spans:
        span["label"] = _human_span_label(str(span["raw_label"]), str(span["role"]), label_counts)

    if for_agent_chart:
        chart_spans: list[dict[str, Any]] = []
        for span in spans:
            lane = _timeline_chart_lane(str(span["role"]), str(span["label"]))
            if lane is None:
                continue
            span["row_label"] = lane
            chart_spans.append(span)
        spans = chart_spans

    # Dedicate one row per unique label (or consolidated row_label for agent chart).
    agent_to_row: dict[str, int] = {}
    for span in spans:
        name = str(span["row_label"]) if for_agent_chart else span["label"]
        if name not in agent_to_row:
            agent_to_row[name] = len(agent_to_row)
        span["row"] = agent_to_row[name]

    num_rows = max(len(agent_to_row), 1)

    # Total wall-clock duration.
    started = _parse_iso(str(run.get("started_at", "")))
    completed = _parse_iso(str(run.get("completed_at", "")))
    if started and completed:
        total_ms = max(int((completed - started).total_seconds() * 1000), 1)
    else:
        total_ms = max((s["end_ms"] for s in spans), default=5000)

    return spans, num_rows, total_ms


def _gantt_svg(spans: list[dict[str, Any]], num_rows: int, total_ms: int) -> str:
    if not spans or total_ms <= 0:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 80" width="100%" '
            'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;display:block;">'
            '<rect width="960" height="80" fill="#fff"/>'
            '<text x="24" y="44" font-size="12" fill="#94a3b8">No agent timing data available.</text>'
            '</svg>'
        )

    SVG_W     = 960
    LEFT_PAD  = 220   # label column (room for "Optimizer evaluation", etc.)
    RIGHT_PAD = 16
    CHART_W   = SVG_W - LEFT_PAD - RIGHT_PAD
    BAR_H     = 18
    ROW_H     = 26
    AXIS_H    = 34
    MAX_ROWS  = 40
    BOT_PAD   = 38 if num_rows > MAX_ROWS else 28  # caption (+ optional overflow line)

    display_rows  = min(num_rows, MAX_ROWS)
    display_spans = [s for s in spans if s["row"] < display_rows]
    svg_h         = AXIS_H + display_rows * ROW_H + BOT_PAD

    p: list[str] = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {svg_h}" '
        f'width="100%" style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;display:block;">'
    )
    p.append(f'<rect width="{SVG_W}" height="{svg_h}" fill="#fff"/>')

    # Alternating row stripes.
    for row in range(display_rows):
        ry = AXIS_H + row * ROW_H
        if row % 2 == 1:
            p.append(f'<rect x="{LEFT_PAD}" y="{ry}" width="{CHART_W + RIGHT_PAD}" height="{ROW_H}" fill="#f8fafc"/>')

    # Time axis ticks.
    tick_ms = _nice_tick_ms(total_ms)
    tick = 0
    while True:
        tx = LEFT_PAD + int(tick / total_ms * CHART_W)
        if tx > SVG_W - RIGHT_PAD + 2:
            break
        label = _fmt_tick(tick)
        p.append(
            f'<line x1="{tx}" y1="{AXIS_H - 4}" x2="{tx}" y2="{AXIS_H + display_rows * ROW_H}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        p.append(f'<text x="{tx}" y="{AXIS_H - 8}" text-anchor="middle" font-size="10" fill="#94a3b8">{html.escape(label)}</text>')
        tick += tick_ms
        if tick > total_ms + tick_ms:
            break

    # Axis baseline.
    p.append(f'<line x1="{LEFT_PAD}" y1="{AXIS_H}" x2="{SVG_W - RIGHT_PAD}" y2="{AXIS_H}" stroke="#e2e8f0" stroke-width="1"/>')

    labeled_rows: set[int] = set()

    for span in display_spans:
        offset  = span["offset_ms"]
        runtime = max(span["runtime_ms"], 50)   # min 50 ms so 0-duration spans show
        row     = span["row"]
        color   = _role_color(span["role"])

        bx = LEFT_PAD + int(offset / total_ms * CHART_W)
        bw = max(int(runtime / total_ms * CHART_W), 3)
        bw = min(bw, LEFT_PAD + CHART_W - bx)   # clamp to chart area
        by = AXIS_H + row * ROW_H + (ROW_H - BAR_H) // 2

        opacity = "0.35" if span["status"] == "failed" else "1"

        tok = span["token_usage"]
        tip = "\n".join(filter(None, [
            span["label"],
            f"Trace: {span['raw_label']}" if span.get("raw_label") != span["label"] else None,
            f"Role: {span['role']}",
            f"Status: {span['status']}",
            f"Duration: {_fmt_duration(span['runtime_ms'] / 1000)}",
            f"Tokens: {tok:,}" if tok else None,
            f"Start: +{_fmt_duration(offset / 1000)}",
            f"Model: {span['model']}" if span["model"] else None,
            span["summary"][:90] if span["summary"] else None,
        ]))

        p.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{BAR_H}" rx="4" '
            f'fill="{color}" opacity="{opacity}">'
            f'<title>{html.escape(tip)}</title>'
            f'</rect>'
        )

        # Label inside bar when wide enough.
        if bw > 50:
            chars = max(4, bw // 6)
            short = _shorten(span["label"], chars)
            p.append(
                f'<text x="{bx + 6}" y="{by + BAR_H // 2 + 4}" '
                f'font-size="9" fill="#fff" font-weight="600" style="pointer-events:none;">'
                f'{html.escape(short)}</text>'
            )

        # Row label (left column) — first span per row only.
        if row not in labeled_rows:
            labeled_rows.add(row)
            label_y = AXIS_H + row * ROW_H + ROW_H // 2 + 4
            short = _shorten(_gantt_row_label(span), 30)
            p.append(
                f'<text x="{LEFT_PAD - 10}" y="{label_y}" text-anchor="end" '
                f'font-size="11" fill="{color}" font-weight="500">'
                f'{html.escape(short)}</text>'
            )

    caption_y = AXIS_H + display_rows * ROW_H + 14
    p.append(
        f'<text x="12" y="{caption_y}" font-size="9" fill="#64748b">'
        f'Each bar spans wall-clock time from agent start until model output is ready.</text>'
    )

    if num_rows > MAX_ROWS:
        p.append(
            f'<text x="{SVG_W // 2}" y="{caption_y + 12}" text-anchor="middle" '
            f'font-size="10" fill="#94a3b8">… {num_rows - MAX_ROWS} more rows not shown</text>'
        )

    p.append("</svg>")
    return "\n".join(p)


def _gantt_png(spans: list[dict[str, Any]], num_rows: int, total_ms: int) -> bytes:
    width = 1280
    left_pad = 272
    right_pad = 24
    chart_w = width - left_pad - right_pad
    row_h = 26
    axis_h = 34
    bottom = 38 if num_rows > 40 else 28  # caption (+ optional overflow line)
    display_rows = min(num_rows, 40)
    height = axis_h + max(display_rows, 1) * row_h + bottom
    canvas = _PngCanvas(width, height, "#ffffff")
    if not spans or total_ms <= 0:
        canvas.text(24, 44, "No agent timing data available", "#94a3b8", 2)
        return canvas.png()
    for row in range(display_rows):
        y = axis_h + row * row_h
        if row % 2:
            canvas.rect(left_pad, y, chart_w + right_pad, row_h, "#f8fafc")
    tick_ms = _nice_tick_ms(total_ms)
    tick = 0
    while tick <= total_ms + tick_ms:
        x = left_pad + int(tick / total_ms * chart_w)
        if x > width - right_pad:
            break
        canvas.rect(x, axis_h - 4, 1, height - axis_h - bottom + 4, "#e2e8f0")
        canvas.text(max(left_pad, x - 12), 14, _fmt_tick(tick), "#94a3b8", 1)
        tick += tick_ms
    canvas.rect(left_pad, axis_h, chart_w, 1, "#cbd5e1")
    labeled_rows: set[int] = set()
    for span in [s for s in spans if s["row"] < display_rows]:
        row = int(span["row"])
        x = left_pad + int(span["offset_ms"] / total_ms * chart_w)
        w = max(int(max(span["runtime_ms"], 50) / total_ms * chart_w), 3)
        w = min(w, left_pad + chart_w - x)
        y = axis_h + row * row_h + 5
        color = _role_color(str(span["role"]))
        if span.get("status") == "failed":
            color = "#fca5a5"
        canvas.rect(x, y, w, 16, color)
        if w > 56:
            canvas.text(x + 4, y + 3, _shorten(str(span["label"]), max(6, w // 9)), "#ffffff", 1)
        if row not in labeled_rows:
            labeled_rows.add(row)
            canvas.text(8, axis_h + row * row_h + 8, _shorten(_gantt_row_label(span), 32), color, 1)
    caption_y = axis_h + display_rows * row_h + 12
    canvas.text(12, caption_y, "Each bar: start to model output ready (wall clock).", "#64748b", 1)
    if num_rows > display_rows:
        canvas.text(width // 2 - 100, caption_y + 12, f"... {num_rows - display_rows} more rows", "#94a3b8", 1)
    return canvas.png()


def _event_rows_html(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return (
            '<tr><td colspan="6" style="color:#94a3b8;text-align:center;padding:20px;">'
            'No agent events recorded.</td></tr>'
        )
    rows: list[str] = []
    for span in sorted(spans, key=lambda s: s["offset_ms"]):
        color = _role_color(span["role"])
        badge = (
            f'<span style="display:inline-block;padding:1px 7px;border-radius:4px;'
            f'font-size:10px;font-weight:700;letter-spacing:.04em;color:#fff;'
            f'background:{color};">{html.escape(_role_short(span["role"]))}</span>'
        )
        tok   = f'{span["token_usage"]:,}' if span["token_usage"] else "—"
        dur   = _fmt_duration(span["runtime_ms"] / 1000)
        off   = f'+{_fmt_duration(span["offset_ms"] / 1000)}'
        dim   = ' style="opacity:.45;"' if span["status"] == "failed" else ""
        summ  = html.escape(span["summary"]) if span["summary"] else '<span style="color:#94a3b8">—</span>'
        name  = html.escape(_shorten(span["label"], 32))
        rows.append(
            f'<tr{dim}>'
            f'<td>{badge}</td>'
            f'<td style="font-family:\'SF Mono\',\'Fira Code\',monospace;font-size:12px;">{name}</td>'
            f'<td style="color:#64748b;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{summ}</td>'
            f'<td style="text-align:right;font-family:monospace;color:#475569;">{html.escape(tok)}</td>'
            f'<td style="text-align:right;font-family:monospace;">{html.escape(dur)}</td>'
            f'<td style="text-align:right;font-family:monospace;color:#94a3b8;">{html.escape(off)}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _stats_cards_html(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") or {}
    best   = float((summary.get("best_evaluation") or {}).get("score") or 0.0)
    items  = [
        ("Sources",     counts.get("sources", 0)),
        ("Claims",      counts.get("claims", 0)),
        ("Hypotheses",  counts.get("hypotheses", 0)),
        ("Variants",    counts.get("variants", 0)),
        ("Evals",       counts.get("evaluations", 0)),
        ("Rounds",      counts.get("outer_rounds", 0)),
        ("Decisions",   counts.get("continuation_decisions", 0)),
        ("Tasks",       f'{counts.get("passed_tasks", 0)}/{counts.get("tasks", 0)}'),
        ("Best score",  f"{best:.3f}"),
        ("Agents",      counts.get("agent_traces", 0)),
        ("Errors",      counts.get("failed_agents", 0)),
    ]
    return "".join(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:10px 14px;min-width:90px;">'
        f'<div style="font-size:10px;font-weight:600;letter-spacing:.08em;'
        f'text-transform:uppercase;color:#94a3b8;">{html.escape(lbl)}</div>'
        f'<div style="font-size:22px;font-weight:700;color:#1e293b;margin-top:3px;'
        f'font-family:\'SF Mono\',monospace;">{html.escape(str(val))}</div>'
        f'</div>'
        for lbl, val in items
    )


def _round_rows_html(summary: dict[str, Any]) -> str:
    rounds = summary.get("rounds") or []
    if not rounds:
        return (
            '<tr><td colspan="5" style="color:#94a3b8;text-align:center;padding:16px;">'
            'No evolution rounds recorded.</td></tr>'
        )
    rows: list[str] = []
    for r in rounds:
        score  = float(r.get("best_score") or 0.0)
        bar_w  = int(score * 80)
        bar    = (
            f'<span style="display:inline-block;height:6px;width:{bar_w}px;'
            f'border-radius:3px;background:#3b82f6;vertical-align:middle;margin-right:6px;"></span>'
        )
        signal = str(r.get("termination_signal") or "—")
        sig_color = "#10b981" if "threshold" in signal else "#f59e0b" if "plateau" in signal else "#64748b"
        rows.append(
            f'<tr>'
            f'<td style="color:#64748b;">{r.get("outer_iteration", "—")}</td>'
            f'<td>{html.escape(str(r.get("mode", "—")))}</td>'
            f'<td>{bar}<span style="font-family:monospace;">{score:.3f}</span></td>'
            f'<td><span style="color:{sig_color};font-size:11px;font-weight:600;">{html.escape(signal)}</span></td>'
            f'<td style="color:#94a3b8;">{r.get("plateau_count", 0)}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def write_run_benchmarks(store: ArtifactStore) -> None:
    if not store.harness_diagnosis_path.exists():
        store.write_harness_diagnosis()
    summary = build_run_summary(store)
    (store.root / "run_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dag = decision_dag_mermaid(summary)
    spans, num_rows, total_ms = _build_timeline_spans(summary, for_agent_chart=True)
    dag_svg = decision_dag_svg(summary)
    timeline_svg = _gantt_svg(spans, num_rows, total_ms)
    (store.root / "decision_dag.mmd").write_text(dag, encoding="utf-8")
    _write_png_from_svg_or_fallback(store.decision_dag_path, dag_svg, lambda: decision_dag_png(summary))
    _write_png_from_svg_or_fallback(store.agent_timeline_path, timeline_svg, lambda: _gantt_png(spans, num_rows, total_ms))
    (store.root / "run_benchmark.md").write_text(run_benchmark_markdown(summary, dag), encoding="utf-8")
    (store.root / "run_benchmark.html").write_text(run_benchmark_html(summary), encoding="utf-8")
    store.run_notebook_path.write_text(json.dumps(run_notebook_export(summary), indent=2) + "\n", encoding="utf-8")


def build_run_summary(store: ArtifactStore) -> dict[str, Any]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    traces = store.list("agent_traces")
    tasks = store.list("loop_tasks")
    decisions = store.list("task_ingestion_decisions")
    continuation_decisions = store.list("loop_continuation_decisions")
    variants = store.list("variants")
    evaluations = store.list("variant_evaluations")
    rounds = store.list("evolution_rounds")
    prd = read_json(store.prd_path, {})
    optimizer_seed_context = read_json(store.optimizer_seed_context_path, {})
    optimization_result = read_json(store.optimization_result_path, {})
    optimized_candidate_exists = store.optimized_candidate_path.exists()
    optimal_code_exists = store.optimal_code_path.exists()
    solution_exists = store.solution_path.exists()
    sources = store.list("sources")
    claims = store.list("claims")
    hypotheses = store.list("hypotheses")
    contradictions = store.list("contradictions")
    provenance_edges = store.list("provenance_edges")
    cost_events = store.list("cost_events")
    harness_diagnosis = read_json(store.harness_diagnosis_path, {})
    cost = read_json(store.cost_path, {})
    models = Counter(str(trace.get("model", "unknown")) for trace in traces)
    best_eval = max(evaluations, key=lambda row: float(row.get("score", 0.0)), default={})
    return {
        "run": run,
        "counts": {
            "tasks": len(tasks),
            "passed_tasks": sum(1 for task in tasks if task.get("passes")),
            "outer_rounds": len(rounds),
            "continuation_decisions": len(continuation_decisions),
            "variants": len(variants),
            "evaluations": len(evaluations),
            "sources": len(sources),
            "claims": len(claims),
            "hypotheses": len(hypotheses),
            "contradictions": len(contradictions),
            "provenance_edges": len(provenance_edges),
            "cost_events": len(cost_events),
            "agent_traces": len(traces),
            "failed_agents": sum(1 for trace in traces if trace.get("status") != "completed"),
        },
        "task_ingestion": decisions[0] if decisions else None,
        "prd": prd,
        "optimizer_seed_context": optimizer_seed_context,
        "optimization_result": optimization_result,
        "harness_diagnosis": harness_diagnosis,
        "cost": cost,
        "optimized_candidate": str(store.optimized_candidate_path) if optimized_candidate_exists else None,
        "optimal_code": str(store.optimal_code_path) if optimal_code_exists else None,
        "solution": str(store.solution_path) if solution_exists else None,
        "models": dict(models),
        "tasks": tasks,
        "rounds": rounds,
        "continuation_decisions": continuation_decisions,
        "variants": variants,
        "evaluations": evaluations,
        "best_evaluation": best_eval,
        "trace_summaries": [
            {
                "agent_name":  trace.get("agent_name"),
                "role":        trace.get("role"),
                "model":       trace.get("model"),
                "status":      trace.get("status"),
                "runtime_ms":  trace.get("runtime_ms"),
                "token_usage": trace.get("token_usage"),
                "started_at":  trace.get("started_at", ""),
                "summary":     trace.get("output_summary"),
            }
            for trace in traces
        ],
    }


def run_notebook_export(summary: dict[str, Any]) -> dict[str, Any]:
    run = summary.get("run") or {}
    counts = summary.get("counts") or {}
    diagnosis = summary.get("harness_diagnosis") or {}
    cost = summary.get("cost") or {}
    cells = [
        _markdown_cell(
            "# Research Harness Run\n\n"
            f"- Run: `{run.get('id', 'unknown')}`\n"
            f"- Status: `{run.get('status', 'unknown')}`\n"
            f"- Goal: {run.get('user_goal', '')}\n"
        ),
        _markdown_cell(
            "## Artifact Counts\n\n"
            + "\n".join(f"- {key}: {value}" for key, value in sorted(counts.items()))
        ),
        _markdown_cell(
            "## Observability\n\n"
            f"- Total cost: `${float(cost.get('cost_usd') or 0.0):.4f}`\n"
            f"- Total tokens: `{cost.get('total_tokens', run.get('total_tokens', 0))}`\n"
            f"- Model calls: `{cost.get('model_call_count', 0)}`\n"
        ),
        _code_cell("harness_diagnosis = " + json.dumps(diagnosis, indent=2, sort_keys=True)),
        _code_cell("trace_summaries = " + json.dumps(summary.get("trace_summaries", []), indent=2, sort_keys=True)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def decision_dag_mermaid(summary: dict[str, Any]) -> str:
    run = summary.get("run", {})
    decision = summary.get("task_ingestion") or {}
    lines = [
        "flowchart TD",
        f'  prompt["Prompt: {_mermaid(str(run.get("user_goal", "")))}"]',
        f'  route["Task router\\nProduct: {decision.get("product_agent", run.get("product_agent", "unknown"))}\\nMode: {decision.get("selected_mode", run.get("task_mode", "unknown"))}"]',
        '  memory["LeadResearcher memory\\nPRD + plan + objective + seed context"]',
        '  propose["Propose specialized subagent tasks / variants"]',
        '  subagents["Parallel subagents\\nresearch / optimizer evaluators"]',
        '  rank["Evaluate, rank, select parents"]',
        '  continue{"More research or optimization needed?"}',
        '  recover["Refine strategy / spawn next subagents"]',
        '  synth["Critic + synthesis"]',
        '  cite["Citation / grounding pass"]',
        '  persist["Persist report, traces, PNGs, costs"]',
        "  prompt --> route --> memory --> propose --> subagents --> rank --> continue",
        "  continue -- continue --> recover --> propose",
        "  continue -- exit --> synth --> cite --> persist",
    ]
    for index, round_record in enumerate(summary.get("rounds", []), start=1):
        node = f"round{index}"
        label = (
            f"Round {round_record.get('outer_iteration')}: "
            f"best={float(round_record.get('best_score', 0.0)):.3f}, "
            f"{round_record.get('termination_signal', 'continue')}"
        )
        lines.append(f'  {node}["{_mermaid(label)}"]')
        lines.append(f"  rank --> {node} --> continue")
    for index, item in enumerate(summary.get("continuation_decisions", []), start=1):
        node = f"decision{index}"
        label = f"Decision {item.get('iteration')}: {item.get('decision')} ({item.get('termination_signal')})"
        lines.append(f'  {node}["{_mermaid(label)}"]')
        lines.append(f"  continue --> {node}")
    return "\n".join(lines) + "\n"


def decision_dag_svg(summary: dict[str, Any]) -> str:
    run = summary.get("run") or {}
    decision = summary.get("task_ingestion") or {}
    counts = summary.get("counts") or {}
    continuations = summary.get("continuation_decisions") or []
    best = float((summary.get("best_evaluation") or {}).get("score") or 0.0)
    width = 1280
    cards: list[tuple[int, int, int, int, str, str, str]] = []
    x1, x2, x3 = 40, 460, 880
    y = 82
    cards.extend([
        (x1, y, 340, 54, "1 User Prompt", str(run.get("user_goal", ""))[:54], "#dbeafe"),
        (x2, y, 340, 54, "2 Task Router", f"{decision.get('product_agent', run.get('product_agent', 'unknown'))} / {decision.get('selected_mode', run.get('task_mode', 'unknown'))}", "#fce7f3"),
        (x3, y, 340, 54, "3 LeadResearcher Memory", "PRD, plan, objective, context", "#ccfbf1"),
    ])
    y += 88
    cards.extend([
        (x1, y, 340, 54, "4 Propose Subagent Tasks", f"{counts.get('variants', 0)} variants across {counts.get('outer_rounds', 0)} rounds", "#e0f2fe"),
        (x2, y, 340, 54, "5 Parallel Subagents", "research / optimizer evaluators fan out", "#dbeafe"),
        (x3, y, 340, 54, "6 Evaluate + Rank", f"{counts.get('evaluations', 0)} evals, best score {best:.3f}", "#fee2e2"),
    ])
    y += 88
    cards.append((x2, y, 340, 54, "7 Continue Decision", f"{counts.get('continuation_decisions', 0)} explicit loop decisions", "#ccfbf1"))
    for idx, cont in enumerate(continuations[:8], start=1):
        y += 70
        color = "#dcfce7" if cont.get("decision") == "continue" else "#ffedd5"
        cards.append((x2, y, 340, 54, f"Decision r{cont.get('iteration', idx)}: {cont.get('decision', '?')}", str(cont.get("reason", ""))[:52], color))
    y += 84
    cards.extend([
        (x1, y, 340, 54, "8 Critic + Synthesis", "review claims, contradictions, report", "#fef3c7"),
        (x2, y, 340, 54, "9 Citation / Grounding", "claim-source links and source-backed output", "#ede9fe"),
        (x3, y, 340, 54, "10 Persist Artifacts", "report, traces, PNGs, decisions, costs", "#e2e8f0"),
    ])
    height = max(card_y + card_h for _, card_y, _, card_h, _, _, _ in cards) + 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="28" y="45" font-size="24" font-weight="700" fill="#0f172a">Comprehensive decision DAG</text>',
    ]
    for x, y, w, h, title, body, fill in cards:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#64748b"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 22}" font-size="13" font-weight="700" fill="#0f172a">{html.escape(title)}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 42}" font-size="11" fill="#475569">{html.escape(body)}</text>')
    for start, end in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        parts.append(_svg_arrow(cards[start][0] + cards[start][2], cards[start][1] + 27, cards[end][0], cards[end][1] + 27))
    parts.append(_svg_arrow(x3 + 170, 136, x1 + 170, 170))
    # Evaluate+Rank → Continue Decision: bottom-center of right column card down to top-center of continue card
    parts.append(_svg_arrow(x3 + 170, cards[5][1] + 54, x2 + 170, cards[6][1]))
    if continuations:
        parts.append(_svg_arrow(x2 + 170, cards[6][1] + 54, x2 + 170, cards[7][1]))
        last_decision = min(len(continuations), 8) + 6
        parts.append(_svg_arrow(x2 + 170, cards[last_decision][1] + 54, x1 + 170, cards[-3][1]))
    else:
        parts.append(_svg_arrow(x2 + 170, cards[6][1] + 54, x1 + 170, cards[-3][1]))
    parts.append(_svg_arrow(cards[-3][0] + cards[-3][2], cards[-3][1] + 27, cards[-2][0], cards[-2][1] + 27))
    parts.append(_svg_arrow(cards[-2][0] + cards[-2][2], cards[-2][1] + 27, cards[-1][0], cards[-1][1] + 27))
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    mid = (x1 + x2) // 2
    if y1 == y2:
        path = f"M{x1},{y1} L{x2},{y2}"
    else:
        path = f"M{x1},{y1} L{mid},{y1} L{mid},{y2} L{x2},{y2}"
    return (
        f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="1.5"/>'
        f'<path d="M{x2 - 7},{y2 - 5} L{x2},{y2} L{x2 - 7},{y2 + 5}" fill="none" stroke="#64748b" stroke-width="1.5"/>'
    )


def decision_dag_png(summary: dict[str, Any]) -> bytes:
    run = summary.get("run") or {}
    decision = summary.get("task_ingestion") or {}
    counts = summary.get("counts") or {}
    continuations = summary.get("continuation_decisions") or []
    best = float((summary.get("best_evaluation") or {}).get("score") or 0.0)
    width = 1280
    cards: list[tuple[int, int, int, int, str, str, str]] = []
    x1, x2, x3 = 40, 460, 880
    y = 82
    cards.append((x1, y, 340, 54, "1 User prompt", str(run.get("user_goal", ""))[:54], "#dbeafe"))
    cards.append((x2, y, 340, 54, "2 Task router", f"{decision.get('product_agent', run.get('product_agent', 'unknown'))} / {decision.get('selected_mode', run.get('task_mode', 'unknown'))}", "#fce7f3"))
    cards.append((x3, y, 340, 54, "3 Lead plan memory", "PRD, source strategy, objective, context", "#ccfbf1"))
    y += 88
    cards.append((x1, y, 340, 54, "4 Propose subagent tasks", f"{counts.get('variants', 0)} variants across {counts.get('outer_rounds', 0)} rounds", "#e0f2fe"))
    cards.append((x2, y, 340, 54, "5 Parallel subagents", "research / optimize evaluators fan out", "#dbeafe"))
    cards.append((x3, y, 340, 54, "6 Evaluate + rank", f"{counts.get('evaluations', 0)} evals, best score {best:.3f}", "#fee2e2"))
    y += 88
    cards.append((x2, y, 340, 54, "7 Continue decision", f"{counts.get('continuation_decisions', 0)} explicit loop decisions", "#ccfbf1"))
    for idx, cont in enumerate(continuations[:8], start=1):
        y += 70
        color = "#dcfce7" if cont.get("decision") == "continue" else "#ffedd5"
        cards.append((x2, y, 340, 54, f"Decision r{cont.get('iteration', idx)}: {cont.get('decision', '?')}", str(cont.get("reason", ""))[:52], color))
    y += 84
    cards.append((x1, y, 340, 54, "8 Critic + synthesis", "review claims, contradictions, write report", "#fef3c7"))
    cards.append((x2, y, 340, 54, "9 Citation / grounding", "claim-source links and source-backed output", "#ede9fe"))
    cards.append((x3, y, 340, 54, "10 Persist artifacts", "report, traces, timeline, decisions, costs", "#e2e8f0"))
    height = max(card_y + card_h for _, card_y, _, card_h, _, _, _ in cards) + 42
    canvas = _PngCanvas(width, height, "#f8fafc")
    canvas.text(28, 24, "Comprehensive decision DAG", "#0f172a", 2)
    for card in cards:
        _draw_card(canvas, *card)
    # arrows
    for (a, b) in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        _draw_arrow(canvas, cards[a][0] + cards[a][2], cards[a][1] + 27, cards[b][0], cards[b][1] + 27)
    _draw_arrow(canvas, x3 + 170, 136, x1 + 170, 170)
    _draw_arrow(canvas, x3 + 170, cards[5][1] + 54, x2 + 170, cards[6][1])
    if continuations:
        _draw_arrow(canvas, x2 + 170, cards[6][1] + 54, x2 + 170, cards[7][1])
        last_decision = min(len(continuations), 8) + 6
        _draw_arrow(canvas, x2 + 170, cards[last_decision][1] + 54, x1 + 170, cards[-3][1])
    else:
        _draw_arrow(canvas, x2 + 170, cards[6][1] + 54, x1 + 170, cards[-3][1])
    _draw_arrow(canvas, cards[-3][0] + cards[-3][2], cards[-3][1] + 27, cards[-2][0], cards[-2][1] + 27)
    _draw_arrow(canvas, cards[-2][0] + cards[-2][2], cards[-2][1] + 27, cards[-1][0], cards[-1][1] + 27)
    return canvas.png()


def _draw_card(canvas: _PngCanvas, x: int, y: int, w: int, h: int, title: str, body: str, fill: str) -> None:
    canvas.rect(x, y, w, h, fill)
    canvas.outline(x, y, w, h, "#64748b")
    canvas.text(x + 10, y + 9, title, "#0f172a", 2, max_chars=32)
    canvas.text(x + 10, y + 34, body, "#475569", 1, max_chars=70)


def _draw_arrow(canvas: _PngCanvas, x1: int, y1: int, x2: int, y2: int) -> None:
    if x1 == x2:
        y0, yh = sorted((y1, y2))
        canvas.rect(x1, y0, 2, max(1, yh - y0), "#64748b")
    else:
        x0, xh = sorted((x1, x2))
        canvas.rect(x0, y1, max(1, xh - x0), 2, "#64748b")
        if y1 != y2:
            canvas.rect(x2, min(y1, y2), 2, abs(y2 - y1), "#64748b")
    canvas.rect(x2 - 6, y2 - 4, 7, 2, "#64748b")
    canvas.rect(x2 - 6, y2 + 2, 7, 2, "#64748b")


def _write_png_from_svg_or_fallback(path: Path, svg: str, fallback: Any) -> None:
    if _write_png_from_svg(path, svg):
        return
    path.write_bytes(fallback())


def _write_png_from_svg(path: Path, svg: str) -> bool:
    converters = [
        ("rsvg-convert", _convert_with_rsvg),
        ("magick", _convert_with_magick),
        ("convert", _convert_with_convert),
        ("qlmanage", _convert_with_qlmanage),
    ]
    with tempfile.TemporaryDirectory(prefix="research_harness_svg_") as directory:
        tmp_dir = Path(directory)
        svg_path = tmp_dir / "source.svg"
        svg_path.write_text(svg, encoding="utf-8")
        for command, converter in converters:
            if not shutil.which(command):
                continue
            try:
                if converter(svg_path, path, tmp_dir):
                    return True
            except Exception:
                continue
    return False


def _convert_with_rsvg(svg_path: Path, output_path: Path, _tmp_dir: Path) -> bool:
    completed = subprocess.run(["rsvg-convert", str(svg_path), "-o", str(output_path)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def _convert_with_magick(svg_path: Path, output_path: Path, _tmp_dir: Path) -> bool:
    completed = subprocess.run(["magick", str(svg_path), str(output_path)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def _convert_with_convert(svg_path: Path, output_path: Path, _tmp_dir: Path) -> bool:
    completed = subprocess.run(["convert", str(svg_path), str(output_path)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def _convert_with_qlmanage(svg_path: Path, output_path: Path, tmp_dir: Path) -> bool:
    completed = subprocess.run(
        ["qlmanage", "-t", "-s", "1600", "-o", str(tmp_dir), str(svg_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    generated = tmp_dir / f"{svg_path.name}.png"
    if completed.returncode == 0 and generated.exists():
        output_path.write_bytes(generated.read_bytes())
        return True
    return False


def run_benchmark_markdown(summary: dict[str, Any], dag: str) -> str:
    counts = summary.get("counts", {})
    decision = summary.get("task_ingestion") or {}
    lines = [
        "# Run Benchmark",
        "",
        f"- Run ID: `{(summary.get('run') or {}).get('id', 'unknown')}`",
        f"- Product agent: `{decision.get('product_agent', (summary.get('run') or {}).get('product_agent', 'unknown'))}`",
        f"- Mode: `{decision.get('selected_mode', (summary.get('run') or {}).get('task_mode', 'unknown'))}`",
        f"- Tasks passed: {counts.get('passed_tasks', 0)} / {counts.get('tasks', 0)}",
        f"- Outer rounds: {counts.get('outer_rounds', 0)}",
        f"- Variants evaluated: {counts.get('evaluations', 0)}",
        f"- Best score: {float((summary.get('best_evaluation') or {}).get('score', 0.0)):.3f}",
        "",
        "## Decision DAG",
        "",
        "```mermaid",
        dag.strip(),
        "```",
        "",
        "## Round Summary",
    ]
    for round_record in summary.get("rounds", []):
        lines.append(
            f"- Round {round_record.get('outer_iteration')}: best `{round_record.get('best_variant_id')}` "
            f"score {float(round_record.get('best_score', 0.0)):.3f}; signal `{round_record.get('termination_signal')}`."
        )
    return "\n".join(lines) + "\n"


def run_benchmark_html(summary: dict[str, Any]) -> str:
    run      = summary.get("run") or {}
    decision = summary.get("task_ingestion") or {}

    spans, num_rows, total_ms = _build_timeline_spans(summary, for_agent_chart=False)

    run_id      = str(run.get("id", "unknown"))
    goal        = str(run.get("user_goal", ""))
    status      = str(run.get("status", "running"))
    mode        = str(decision.get("selected_mode", run.get("task_mode", "—")))
    product     = str(decision.get("product_agent", run.get("product_agent", "—")))
    total_tok   = int(run.get("total_tokens") or 0)
    total_cost  = float(run.get("total_cost") or 0.0)
    dur_s       = total_ms / 1000

    status_color = {"completed": "#10b981", "failed": "#ef4444", "running": "#3b82f6"}.get(status, "#94a3b8")

    evt_rows  = _event_rows_html(spans)
    stats     = _stats_cards_html(summary)
    rnd_rows  = _round_rows_html(summary)

    # Compact colour legend.
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;'
        f'font-size:11px;color:#475569;">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:{color};display:inline-block;"></span>'
        f'{html.escape(role.replace("_", " ").title())}</span>'
        for role, color in _ROLE_COLORS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(run_id)} — Research Harness</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px 28px 48px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px; color: #1e293b; background: #f1f5f9; line-height: 1.5;
    }}
    h2 {{
      font-size: 10px; font-weight: 700; letter-spacing: .1em;
      text-transform: uppercase; color: #94a3b8; margin: 20px 0 8px;
    }}
    .card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 18px 22px; margin-bottom: 14px;
    }}
    .header-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
    .run-id {{ font-family: "SF Mono","Fira Code",monospace; font-size: 13px; font-weight: 600; color: #334155; }}
    .badge {{
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 9px; border-radius: 999px; font-size: 11px;
      font-weight: 700; color: #fff; background: {status_color};
    }}
    .goal {{ font-size: 15px; color: #0f172a; margin: 6px 0 10px; font-weight: 500; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: #64748b; }}
    .gantt-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 14px 16px 10px; margin-bottom: 14px; overflow-x: auto;
    }}
    .legend {{ margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 4px; }}
    .events-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
      overflow: hidden; margin-bottom: 14px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{
      padding: 8px 12px; text-align: left; font-weight: 600; color: #64748b;
      border-bottom: 1px solid #f1f5f9; background: #f8fafc;
      font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
    }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #f8fafc; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8fafc; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .rounds-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;
    }}
  </style>
</head>
<body>

  <!-- ── Header ───────────────────────────────────────── -->
  <div class="card">
    <div class="header-top">
      <span class="run-id">{html.escape(run_id)}</span>
      <span class="badge">● {html.escape(status)}</span>
    </div>
    <div class="goal">{html.escape(goal[:140])}</div>
    <div class="meta">
      <span>⏱ {html.escape(_fmt_duration(dur_s))}</span>
      <span>⬡ {total_tok:,} tokens</span>
      <span>${total_cost:.4f}</span>
      <span>mode: <b>{html.escape(mode)}</b></span>
      <span>agent: <b>{html.escape(product)}</b></span>
    </div>
  </div>

  <!-- ── Gantt timeline ───────────────────────────────── -->
  <h2>Agent Timeline</h2>
  <div class="gantt-card">
    <div class="legend">{legend}</div>
    <img src="agent_timeline.png" alt="Agent timeline" style="width:100%;display:block;">
  </div>

  <h2>Decision DAG</h2>
  <div class="gantt-card">
    <img src="decision_dag.png" alt="Decision DAG" style="width:100%;display:block;">
  </div>

  <!-- ── Event log ────────────────────────────────────── -->
  <h2>Agent Events</h2>
  <div class="events-card">
    <table>
      <thead>
        <tr>
          <th>Role</th>
          <th>Agent</th>
          <th>Summary</th>
          <th style="text-align:right;">Tokens</th>
          <th style="text-align:right;">Duration</th>
          <th style="text-align:right;">Offset</th>
        </tr>
      </thead>
      <tbody>{evt_rows}</tbody>
    </table>
  </div>

  <!-- ── Stats cards ──────────────────────────────────── -->
  <h2>Run Stats</h2>
  <div class="stats">{stats}</div>

  <!-- ── Evolution rounds ─────────────────────────────── -->
  <h2>Evolution Rounds</h2>
  <div class="rounds-card">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Mode</th><th>Best score</th>
          <th>Signal</th><th>Plateau</th>
        </tr>
      </thead>
      <tbody>{rnd_rows}</tbody>
    </table>
  </div>

</body>
</html>"""


def _mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")[:110]
