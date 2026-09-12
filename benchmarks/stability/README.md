# Ren v1.0.0 browser stability benchmark (100 points)

Drives the real Phase 6 dashboard in headless Firefox (Playwright) through
the interactive v1 workflow: launch, import, vision, review, retrieval,
evidence, synthesis, provenance, sign-off, export, reloads, failures.

## Prerequisites

- Repo environment working (`direnv allow`, `uv sync`).
- Playwright browser present (`~/.cache/ms-playwright`, installed once).
- Ports 8080 (MedGemma) and 8091 (dashboard under test) free.
- Bench venv at `/tmp/benchvenv` with `playwright` installed (outside the repo venv).

## Launch

```bash
source /tmp/bench_env.sh
/tmp/benchvenv/bin/python benchmarks/stability/run_bench.py
```

The runner starts the dashboard backend and the MedGemma server itself,
executes all scenarios, writes machine-readable
`artifacts/benchmark_results/stability/results.json` plus a human
`report.md`, captures screenshots only for failed scenarios, then tears
both servers down. Filter with `BENCH_ONLY=<substring>`.

## Scoring

`rubric.json` fixes category weights totaling 100. Each scenario splits
its weight equally across ordered checks; crashes, hangs, and timeouts
score zero. Partial credit applies per check. The rubric is frozen before
each full run; repeated stability flows are separate scenarios, not
reweighting.

## Expected runtime

Roughly 30–50 minutes for the full suite (vision analysis and MedGemma
generation dominate). Results record per-scenario milliseconds.

## Output

- `results.json`: per-scenario checks, scores, timings, console errors,
  failed requests, category totals.
- `report.md`: human summary with partial/fail detail.
- `shots/`: screenshots for failed scenarios only.

## Limitations

- Headless Firefox only; no mobile viewports (desktop workstation target).
- Console-error capture starts at page creation; pre-existing backend log
  noise is out of scope.
- Timing-sensitive scenarios use generous timeouts; slowness alone never
  fails a scenario, only wrong or missing behavior does.
