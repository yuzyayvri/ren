"""Stability benchmark entry point. Starts backend + LLM, runs scenarios, writes results."""
from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

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


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _source_state() -> tuple[str, str, str]:
    """Return candidate commit, content digest, and porcelain status."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True)
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    digest = hashlib.sha256()
    for raw_path in paths.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / raw_path.decode("utf-8")
        digest.update(raw_path)
        digest.update(b"\0")
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return commit, digest.hexdigest(), status


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_port_free(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_free(port):
            return True
        time.sleep(0.25)
    return _port_free(port)


def _post_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    req = urllib.request.Request(url, data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    try:
        value = json.loads(body)
    except ValueError:
        value = {"raw": body[:200]}
    return value if isinstance(value, dict) else {"value": value}


def _stop_llm(ui_port: int) -> dict[str, Any]:
    record: dict[str, Any] = {"requested": True}
    try:
        record["response"] = _post_json(f"http://127.0.0.1:{ui_port}/api/server/stop", timeout=30)
    except Exception as exc:  # noqa: BLE001 - teardown evidence records failures
        record["error"] = repr(exc)
    record["port_free"] = _wait_port_free(LLM_PORT)
    record["asserted"] = bool(record.get("port_free")) and "error" not in record
    return record


def _stop_backend(process: subprocess.Popen[Any] | None) -> dict[str, Any]:
    record: dict[str, Any] = {"requested": process is not None}
    if process is None:
        record.update({"exit_code": None, "exited": True, "asserted": True})
        return record
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        record["timeout"] = True
        process.kill()
        process.wait(timeout=10)
    record["exit_code"] = process.returncode
    record["exited"] = process.poll() is not None
    record["asserted"] = bool(record["exited"])
    return record


def _copy_if_present(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    sys.path.insert(0, str(ROOT / "benchmarks" / "stability"))
    from bench import Bench, make_images
    from scenarios import register

    RESULTS.mkdir(parents=True, exist_ok=True)
    commit, source_sha256, source_status = _source_state()
    run_tag = __import__("uuid").uuid4().hex[:12]
    run_dir = RESULTS / "runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)
    # A canonical shots directory is retained for compatibility, but it must
    # never contain a prior run's evidence. The run-scoped directory is the
    # authoritative location recorded in each failed scenario.
    shutil.rmtree(RESULTS / "shots", ignore_errors=True)
    (RESULTS / "shots").mkdir(parents=True, exist_ok=True)
    context: dict[str, Any] = {
        "schema": "ren-stability-run-v2",
        "run_tag": run_tag,
        "candidate_commit": commit,
        "source_sha256": source_sha256,
        "source_status": source_status,
        "source_dirty": bool(source_status),
        "source_digest_scope": "all git-tracked files at run start",
        "started_at": _utc_now(),
        "filtered": bool(os.environ.get("BENCH_ONLY", "")),
        "filter": os.environ.get("BENCH_ONLY", ""),
        "status": "running",
    }
    _write_json(run_dir / "manifest.json", context)

    regdir = ROOT / "artifacts" / "v1_specimens"
    preexisting = {d.name for d in regdir.iterdir() if d.is_dir()} if regdir.is_dir() else set()
    backend: subprocess.Popen[Any] | None = None
    backend_log_handle = None
    bench = None
    payload: dict[str, Any] | None = None
    runtime_s: float | None = None
    outcome = 1

    try:
        for port, name in ((LLM_PORT, "MedGemma"), (UI_PORT, "dashboard")):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3):
                    raise RuntimeError(f"port {port} busy — stop your own {name} server first")
            except urllib.error.HTTPError:
                raise
            except urllib.error.URLError:
                pass

        backend_log_handle = (run_dir / "backend.log").open("w", encoding="utf-8")
        backend = subprocess.Popen(
            [str(ROOT / "scripts" / "vision_python.sh"), "-m", "uvicorn",
             "scripts.phase6_server:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(UI_PORT)],
            cwd=str(ROOT), stdout=backend_log_handle, stderr=subprocess.STDOUT)
        if not wait_http(f"http://127.0.0.1:{UI_PORT}/api/health"):
            raise RuntimeError("backend failed to start; see run-scoped backend.log")

        print("llm start:", _post_json(f"http://127.0.0.1:{UI_PORT}/api/server/start", timeout=400))
        from playwright.async_api import async_playwright

        async def _amain():
            async with async_playwright() as pw:
                browser = await pw.firefox.launch(headless=True)
                try:
                    tmp = run_dir / "tmp"
                    tmp.mkdir(parents=True, exist_ok=True)
                    imgs = make_images(tmp, tag=run_tag)
                    print(f"run tag {run_tag}: content-unique imports for isolation")
                    from PIL import Image as _Image

                    _txl = sorted((ROOT / "data/blood/txl-pbc/TXL-PBC/images/val").glob("*.png"))
                    if len(_txl) < 3:
                        raise RuntimeError("benchmark needs at least three TXL validation images")
                    _seed = sum(run_tag.encode()) % 251
                    for _key, _src in (("txl_real", _txl[1]), ("txl_real2", _txl[2]),
                                       ("txl_review", _txl[1]), ("txl_reject", _txl[2]),
                                       ("txl_mode", _txl[1]), ("txl_bulk", _txl[2]),
                                       ("txl_double", _txl[1]), ("txl_async", _txl[2])):
                        _img = _Image.open(_src).convert("RGB")
                        _px = _img.load()
                        _key_seed = (_seed + sum(_key.encode())) % 251
                        _px[0, 0] = (_key_seed, (_key_seed * 7) % 256, (_key_seed * 13) % 256)
                        _dst = tmp / f"{_key}.png"
                        _img.save(_dst)
                        imgs[_key] = _dst
                    page = await browser.new_page(viewport={"width": 1440, "height": 900})
                    hosts: set[str] = set()

                    def _track(request):
                        try:
                            hosts.add(urllib.parse.urlparse(request.url).hostname or "")
                        except Exception:  # noqa: BLE001 - forensics must never break the run
                            pass

                    page.on("request", _track)
                    local_bench = Bench(page, f"http://127.0.0.1:{UI_PORT}",
                                        artifact_dir=run_dir, backend_log=run_dir / "backend.log")
                    local_bench.hosts = hosts
                    register(local_bench, imgs)
                    only = os.environ.get("BENCH_ONLY", "")
                    if only:
                        local_bench.scenarios = [s for s in local_bench.scenarios if only in s["id"]]
                        print(f"filtered to {len(local_bench.scenarios)} scenarios matching {only!r}")
                    context["scenario_ids"] = [s["id"] for s in local_bench.scenarios]
                    _write_json(run_dir / "manifest.json", context)
                    t0 = time.monotonic()
                    local_results = await local_bench.run_all(
                        progress_path=run_dir / "progress.json")
                    return local_results, round(time.monotonic() - t0, 1), local_bench
                finally:
                    await browser.close()

        import asyncio
        results, runtime_s, bench = asyncio.run(_amain())
        cats: dict[str, dict[str, Any]] = {}
        for result in results:
            category = cats.setdefault(result["cat"], {"weight": 0.0, "score": 0.0, "scenarios": 0})
            category["weight"] += result["weight"]
            category["score"] += result["score"]
            category["scenarios"] += 1
        total = round(sum(result["score"] for result in results), 1)
        payload = {
            "schema": "ren-stability-results-v2",
            "run_tag": run_tag,
            "candidate_commit": commit,
            "source_sha256": source_sha256,
            "source_status": source_status,
            "source_dirty": bool(source_status),
            "source_digest_scope": "all git-tracked files at run start",
            "filtered": bool(os.environ.get("BENCH_ONLY", "")),
            "filter": os.environ.get("BENCH_ONLY", ""),
            "total": total,
            "of": 100,
            "runtime_s": runtime_s,
            "n_scenarios": len(results),
            "categories": {key: {"score": round(value["score"], 1),
                                  "weight": value["weight"],
                                  "scenarios": value["scenarios"]}
                           for key, value in cats.items()},
            "browser": {
                "console_errors": bench.ctx.console_errors,
                "failed_requests": bench.ctx.failed_requests,
                "expected_failed_requests": [
                    request for result in results
                    for request in result.get("expected_failed_requests", [])
                ],
                "unexpected_failed_requests": bench.ctx.unexpected_failed_requests,
                "bad_responses": bench.ctx.bad_responses,
            },
            # Keep these top-level aliases easy for independent consumers to
            # find while the nested browser object groups the evidence.
            "console_errors": bench.ctx.console_errors,
            "failed_requests": bench.ctx.failed_requests,
            "unexpected_failed_requests": bench.ctx.unexpected_failed_requests,
            "bad_responses": bench.ctx.bad_responses,
            "failures": [result["id"] for result in results if result["score"] < result["weight"]],
            "skips": [result["id"] for result in results
                      if any(str(check_result["name"]).startswith("skipped")
                             for check_result in result["checks"])],
            "scenarios": results,
        }
        _write_json(run_dir / "results.json", payload)
        lines = [f"# Stability benchmark: {total}/100 ({runtime_s}s, {len(results)} scenarios)", "",
                 f"- run tag: `{run_tag}`",
                 f"- candidate commit: `{commit}`",
                 f"- source sha256: `{source_sha256}`",
                 f"- filtered: `{bool(os.environ.get('BENCH_ONLY', ''))}`", ""]
        for cid, category in cats.items():
            lines.append(f"- {cid}: {round(category['score'], 1)}/{category['weight']} ({category['scenarios']} scenarios)")
        lines.append("")
        for result in results:
            if result["score"] < result["weight"]:
                lines.append(f"- PARTIAL/FAIL {result['id']}: {result['score']}/{result['weight']}")
                for check_result in result["checks"]:
                    if not check_result["ok"]:
                        lines.append(f"    - {check_result['name']}: {check_result['detail'][:120]}")
        if bench.ctx.failed_requests:
            lines.append(f"- browser failed requests: {len(bench.ctx.failed_requests)} "
                         f"(expected {len(bench.ctx.failed_requests) - len(bench.ctx.unexpected_failed_requests)}, "
                         f"unexpected {len(bench.ctx.unexpected_failed_requests)})")
        if bench.ctx.bad_responses:
            lines.append(f"- browser bad responses (5xx): {len(bench.ctx.bad_responses)}")
        (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        progress = {"done": len(results), "of": len(results), "last": results[-1]["id"] if results else None,
                    "score_so_far": total, "run_tag": run_tag, "candidate_commit": commit,
                    "source_sha256": source_sha256, "filtered": bool(os.environ.get("BENCH_ONLY", ""))}
        _write_json(run_dir / "progress.json", progress)
        print("\n".join(lines))
        filtered = bool(os.environ.get("BENCH_ONLY", ""))
        no_vacuous = all(not str(check_result["name"]).startswith("skipped")
                         for result in results for check_result in result["checks"])
        all_checks = all(check_result["ok"]
                         for result in results for check_result in result["checks"])
        no_crashes = all(not result["crashed"] for result in results)
        unique_scenarios = len({result["id"] for result in results}) == 69
        outcome = int(not (total == 100.0 and len(results) == 69 and unique_scenarios
                           and not filtered and not source_status and no_vacuous
                           and all_checks and no_crashes
                           and not bench.ctx.unexpected_failed_requests
                           and not bench.ctx.bad_responses))
    except Exception as exc:  # noqa: BLE001 - manifest records failed runs
        context["error"] = repr(exc)
        print(f"benchmark failed: {exc}")
    finally:
        # Only the LLM started through this run is stopped.  The port preflight
        # rejects an already-listening endpoint, so this cannot stop user work.
        teardown = {"llm": _stop_llm(UI_PORT) if backend is not None else {
            "requested": False, "port_free": _port_free(LLM_PORT),
            "asserted": _port_free(LLM_PORT)},
                    "backend": _stop_backend(backend),
                    "ui_port_free": _wait_port_free(UI_PORT),
                    "llm_port_free": _wait_port_free(LLM_PORT)}
        teardown["asserted"] = bool(teardown["backend"].get("asserted")
                                    and teardown["ui_port_free"]
                                    and teardown["llm_port_free"]
                                    and teardown["llm"].get("asserted"))
        if outcome == 0 and not teardown["asserted"]:
            outcome = 1
        context.update({"status": "complete" if outcome == 0 else "failed",
                        "release_eligible": outcome == 0,
                        "finished_at": _utc_now(), "runtime_s": runtime_s,
                        "teardown": teardown})
        _write_json(run_dir / "manifest.json", context)
        if payload is not None:
            payload["teardown"] = teardown
            _write_json(run_dir / "results.json", payload)
        if backend_log_handle is not None:
            backend_log_handle.close()
        # Remove only specimens created by this run; pre-existing user records
        # are preserved even if a scenario or process exits abnormally.
        cleaned = 0
        if regdir.is_dir():
            for directory in list(regdir.iterdir()):
                if directory.is_dir() and directory.name not in preexisting:
                    shutil.rmtree(directory, ignore_errors=True)
                    cleaned += 1
        context["cleaned_specimens"] = cleaned
        context["teardown"]["specimens_cleaned"] = cleaned
        _write_json(run_dir / "manifest.json", context)
        if payload is not None:
            payload["teardown"] = context["teardown"]
            _write_json(run_dir / "results.json", payload)
        report_path = run_dir / "report.md"
        if report_path.is_file():
            with report_path.open("a", encoding="utf-8") as report:
                report.write(f"- teardown asserted: {context['teardown']['asserted']}\n")
        # Canonical files are latest-run pointers; their payloads still carry
        # the run tag and provenance so they cannot be mistaken for timeless
        # benchmark claims.  Keep screenshots scoped to this run.
        _copy_if_present(run_dir / "results.json", RESULTS / "results.json")
        _copy_if_present(run_dir / "report.md", RESULTS / "report.md")
        _copy_if_present(run_dir / "progress.json", RESULTS / "progress.json")
        _copy_if_present(run_dir / "backend.log", RESULTS / "backend.log")
        if (RESULTS / "shots").is_dir():
            for screenshot in (RESULTS / "shots").iterdir():
                if screenshot.is_file():
                    screenshot.unlink()
        source_shots = run_dir / "shots"
        if source_shots.is_dir():
            for screenshot in source_shots.iterdir():
                if screenshot.is_file():
                    shutil.copy2(screenshot, RESULTS / "shots" / screenshot.name)
        print(f"cleaned {cleaned} harness-imported specimens; teardown asserted={context['teardown']['asserted']}")
    return outcome


if __name__ == "__main__":
    raise SystemExit(main())
