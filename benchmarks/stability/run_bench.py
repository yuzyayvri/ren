"""Stability benchmark entry point. Starts backend + LLM, runs scenarios, writes results."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "benchmark_results" / "stability"
UI_PORT = 8091
LLM_PORT = 8080


def wait_http(url, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - polling until ready
            time.sleep(1)
    return False


def main() -> int:
    sys.path.insert(0, str(ROOT / "benchmarks" / "stability"))
    from bench import Bench, make_images
    from scenarios import register

    RESULTS.mkdir(parents=True, exist_ok=True)
    for port, name in ((LLM_PORT, "MedGemma"), (UI_PORT, "dashboard")):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3):
                print(f"port {port} busy — stop your own {name} server first")
                return 2
        except Exception:  # noqa: BLE001 - free port is the expected case
            pass
    backend = subprocess.Popen(
        [str(ROOT / "scripts" / "vision_python.sh"), "-m", "uvicorn",
         "scripts.phase6_server:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(UI_PORT)],
        cwd=str(ROOT), stdout=open(RESULTS / "backend.log", "w"),
        stderr=subprocess.STDOUT)
    try:
        if not wait_http(f"http://127.0.0.1:{UI_PORT}/api/health"):
            print("backend failed to start; see artifacts/benchmark_results/stability/backend.log")
            return 2
        # LLM server via the backend's own managed endpoint (frozen winner).
        req = urllib.request.Request(f"http://127.0.0.1:{UI_PORT}/api/server/start",
                                     data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=400) as r:
            print("llm start:", r.read().decode()[:120])
        from playwright.async_api import async_playwright

        async def _amain():
            async with async_playwright() as pw:
                browser = await pw.firefox.launch(headless=True)
                try:
                    tmp = RESULTS / "tmp"
                    tmp.mkdir(exist_ok=True)
                    imgs = make_images(tmp)
                    txl = sorted((ROOT / "data/blood/txl-pbc/TXL-PBC/images/val").glob("*.png"))
                    imgs["txl_real"] = txl[1]
                    imgs["txl_real2"] = txl[2]
                    page = await browser.new_page(viewport={"width": 1440, "height": 900})
                    hosts: set[str] = set()

                    def _track(request):
                        try:
                            hosts.add(urllib.parse.urlparse(request.url).hostname or "")
                        except Exception:  # noqa: BLE001
                            pass

                    page.on("request", _track)
                    bench = Bench(page, f"http://127.0.0.1:{UI_PORT}")
                    bench.hosts = hosts
                    register(bench, imgs)
                    import os
                    only = os.environ.get("BENCH_ONLY", "")
                    if only:
                        bench.scenarios = [s for s in bench.scenarios if only in s["id"]]
                        print(f"filtered to {len(bench.scenarios)} scenarios matching {only!r}")
                    t0 = time.monotonic()
                    results = await bench.run_all(
                        progress_path=RESULTS / 'progress.json')
                    return results, round(time.monotonic() - t0, 1), bench
                finally:
                    await browser.close()

        import asyncio
        results, runtime_s, bench = asyncio.run(_amain())
        cats: dict[str, dict] = {}
        for r in results:
            c = cats.setdefault(r["cat"], {"weight": 0.0, "score": 0.0, "scenarios": 0})
            c["weight"] += r["weight"]
            c["score"] += r["score"]
            c["scenarios"] += 1
        total = round(sum(r["score"] for r in results), 1)
        payload = {"total": total, "of": 100, "runtime_s": runtime_s,
                   "n_scenarios": len(results),
                   "categories": {k: {"score": round(v["score"], 1), "weight": v["weight"],
                                      "scenarios": v["scenarios"]} for k, v in cats.items()},
                   "console_errors": page_ctx_errors(bench),
                   "scenarios": results}
        (RESULTS / "results.json").write_text(json.dumps(payload, indent=1) + "\n")
        lines = [f"# Stability benchmark: {total}/100 ({runtime_s}s, {len(results)} scenarios)", ""]
        for cid, c in cats.items():
            lines.append(f"- {cid}: {round(c['score'],1)}/{c['weight']} ({c['scenarios']} scenarios)")
        lines.append("")
        for r in results:
            if r["score"] < r["weight"]:
                lines.append(f"- PARTIAL/FAIL {r['id']}: {r['score']}/{r['weight']}")
                for ch in r["checks"]:
                    if not ch["ok"]:
                        lines.append(f"    - {ch['name']}: {ch['detail'][:120]}")
        (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0
    finally:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{UI_PORT}/api/server/stop",
                                         data=b"{}", method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        backend.terminate()


def page_ctx_errors(bench):
    try:
        return bench.ctx.console_errors[:20]
    except Exception:  # noqa: BLE001
        return []


if __name__ == "__main__":
    raise SystemExit(main())
