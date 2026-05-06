# Examples

This folder contains reusable inputs and example commands. Generated run outputs
belong in the root `outputs/` directory and generated benchmark dashboards belong
in the root `benchmarks/` directory. Both are ignored by git except for
placeholder `.gitkeep` files.

## Example Runs

Offline deterministic demo:

```bash
./autore "Research how multi-agent systems improve automated literature review quality" --retriever local
```

Live arXiv literature run:

```bash
./autore "Please research new agent paradigms on arxiv and determine which ones will be used in 5 years based on current workplace trends" --retriever arxiv
```

Benchmark all local runs:

```bash
./autore-bench --benchmark-output benchmarks/latest
```

## Human-Readable Run Names

Run directories are named from the research goal:

```text
outputs/run_new-agent-paradigms-arxiv-determine-5-years-workplace-trends/
outputs/run_new-agent-paradigms-arxiv-determine-5-years-workplace-trends-02/
```

The numeric suffix is added automatically when the same goal is run more than once.
