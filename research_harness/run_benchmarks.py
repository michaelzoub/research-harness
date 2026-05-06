from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .store import ArtifactStore


def write_run_benchmarks(store: ArtifactStore) -> None:
    summary = build_run_summary(store)
    (store.root / "run_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dag = decision_dag_mermaid(summary)
    (store.root / "decision_dag.mmd").write_text(dag, encoding="utf-8")
    (store.root / "decision_dag.svg").write_text(decision_dag_svg(summary), encoding="utf-8")
    (store.root / "run_benchmark.md").write_text(run_benchmark_markdown(summary, dag), encoding="utf-8")
    (store.root / "run_benchmark.html").write_text(run_benchmark_html(summary), encoding="utf-8")


def build_run_summary(store: ArtifactStore) -> dict[str, Any]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    traces = store.list("agent_traces")
    tasks = store.list("loop_tasks")
    decisions = store.list("task_ingestion_decisions")
    variants = store.list("variants")
    evaluations = store.list("variant_evaluations")
    rounds = store.list("evolution_rounds")
    sources = store.list("sources")
    claims = store.list("claims")
    hypotheses = store.list("hypotheses")
    contradictions = store.list("contradictions")
    models = Counter(str(trace.get("model", "unknown")) for trace in traces)
    best_eval = max(evaluations, key=lambda row: float(row.get("score", 0.0)), default={})
    return {
        "run": run,
        "counts": {
            "tasks": len(tasks),
            "passed_tasks": sum(1 for task in tasks if task.get("passes")),
            "outer_rounds": len(rounds),
            "variants": len(variants),
            "evaluations": len(evaluations),
            "sources": len(sources),
            "claims": len(claims),
            "hypotheses": len(hypotheses),
            "contradictions": len(contradictions),
            "agent_traces": len(traces),
            "failed_agents": sum(1 for trace in traces if trace.get("status") != "completed"),
        },
        "task_ingestion": decisions[0] if decisions else None,
        "models": dict(models),
        "tasks": tasks,
        "rounds": rounds,
        "variants": variants,
        "evaluations": evaluations,
        "best_evaluation": best_eval,
        "trace_summaries": [
            {
                "agent_name": trace.get("agent_name"),
                "role": trace.get("role"),
                "model": trace.get("model"),
                "status": trace.get("status"),
                "runtime_ms": trace.get("runtime_ms"),
                "summary": trace.get("output_summary"),
            }
            for trace in traces
        ],
    }


def decision_dag_mermaid(summary: dict[str, Any]) -> str:
    run = summary.get("run", {})
    decision = summary.get("task_ingestion") or {}
    lines = [
        "flowchart TD",
        f'  prompt["Prompt: {_mermaid(str(run.get("user_goal", "")))}"]',
        f'  route["Route: {decision.get("selected_mode", run.get("task_mode", "unknown"))}"]',
        '  outer["Outer orchestrator: propose variants"]',
        '  inner["Inner loop: evaluate and rank"]',
        '  select["Tournament selection"]',
        '  stop{"Threshold or plateau?"}',
        '  synth["Critic + synthesis + run benchmark"]',
        "  prompt --> route --> outer --> inner --> select --> stop",
        "  stop -- continue --> outer",
        "  stop -- stop --> synth",
    ]
    for index, round_record in enumerate(summary.get("rounds", []), start=1):
        node = f"round{index}"
        label = (
            f"Round {round_record.get('outer_iteration')}: "
            f"best={float(round_record.get('best_score', 0.0)):.3f}, "
            f"{round_record.get('termination_signal', 'continue')}"
        )
        lines.append(f'  {node}["{_mermaid(label)}"]')
        lines.append(f"  inner --> {node} --> select")
    return "\n".join(lines) + "\n"


def decision_dag_svg(summary: dict[str, Any]) -> str:
    steps = [
        ("Prompt", str((summary.get("run") or {}).get("user_goal", ""))[:80]),
        ("Route", str((summary.get("task_ingestion") or {}).get("selected_mode", "unknown"))),
        ("Outer", f"{summary.get('counts', {}).get('variants', 0)} variants"),
        ("Inner", f"{summary.get('counts', {}).get('evaluations', 0)} evaluations"),
        ("Best", f"{float((summary.get('best_evaluation') or {}).get('score', 0.0)):.3f} score"),
        ("Output", "report + per-run benchmark"),
    ]
    width = 1100
    height = 170
    gap = 18
    box_w = 160
    y = 46
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
    ]
    for index, (title, body) in enumerate(steps):
        x = 24 + index * (box_w + gap)
        parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='78' rx='12' fill='#ffffff' stroke='#243b53'/>")
        parts.append(f"<text x='{x + 12}' y='{y + 28}' font-size='15' font-family='Georgia' fill='#102a43'>{html.escape(title)}</text>")
        parts.append(f"<text x='{x + 12}' y='{y + 52}' font-size='11' font-family='Verdana' fill='#52606d'>{html.escape(body)}</text>")
        if index < len(steps) - 1:
            arrow_x = x + box_w
            parts.append(f"<path d='M {arrow_x + 3} {y + 39} L {arrow_x + gap - 5} {y + 39}' stroke='#627d98' stroke-width='2'/>")
            parts.append(f"<path d='M {arrow_x + gap - 12} {y + 33} L {arrow_x + gap - 4} {y + 39} L {arrow_x + gap - 12} {y + 45}' fill='none' stroke='#627d98' stroke-width='2'/>")
    parts.append("</svg>")
    return "\n".join(parts)


def run_benchmark_markdown(summary: dict[str, Any], dag: str) -> str:
    counts = summary.get("counts", {})
    decision = summary.get("task_ingestion") or {}
    lines = [
        "# Run Benchmark",
        "",
        f"- Run ID: `{(summary.get('run') or {}).get('id', 'unknown')}`",
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
    counts = summary.get("counts", {})
    cards = "".join(
        f"<div class='card'><b>{html.escape(str(key))}</b><span>{html.escape(str(value))}</span></div>"
        for key, value in counts.items()
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('outer_iteration')))}</td>"
        f"<td>{html.escape(str(row.get('best_variant_id')))}</td>"
        f"<td>{float(row.get('best_score', 0.0)):.3f}</td>"
        f"<td>{html.escape(str(row.get('termination_signal')))}</td>"
        "</tr>"
        for row in summary.get("rounds", [])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Run Benchmark</title>
  <style>
    body {{ margin: 32px; color: #102a43; background: #fbfaf7; font-family: Georgia, "Times New Roman", serif; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ background: white; border: 1px solid #bcccdc; border-radius: 12px; padding: 14px; }}
    .card span {{ display: block; font-size: 26px; margin-top: 8px; }}
    img {{ width: 100%; max-width: 1100px; border: 1px solid #d9e2ec; border-radius: 12px; background: white; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Run Benchmark</h1>
  <p>{html.escape(str((summary.get("run") or {}).get("user_goal", "")))}</p>
  <img src="decision_dag.svg" alt="Decision DAG">
  <div class="cards">{cards}</div>
  <h2>Evolution Rounds</h2>
  <table><thead><tr><th>Round</th><th>Best Variant</th><th>Score</th><th>Signal</th></tr></thead><tbody>{rows}</tbody></table>
</body>
</html>
"""


def _mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")[:110]
