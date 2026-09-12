"""Ren v1.0.0 browser stability benchmark (100 points).

Drives the real dashboard in headless Firefox via Playwright: clicks,
scrolls, imports, reviews, retrieval, synthesis, sign-off, export,
reloads, failures. Results as machine-readable JSON plus a human report.
See README.md for prerequisites and rerun instructions.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "benchmark_results" / "stability"
UI_PORT = 8091
UI_URL = f"http://127.0.0.1:{UI_PORT}"
LLM_PORT = 8080

TIMEOUT_ANALYZE = 240_000
TIMEOUT_SYNTH = 240_000
TIMEOUT_UI = 20_000


class Ctx:
    def __init__(self, page):
        self.page = page
        self.console_errors: list[str] = []
        self.bad_responses: list[str] = []
        self.failed_requests: list[str] = []
        page.on("console", lambda m: self.console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self.console_errors.append(str(e)))
        page.on("requestfailed", lambda r: self.failed_requests.append(f"{r.method} {r.url}"))
        page.on("response", lambda r: self.bad_responses.append(f"{r.status} {r.url}") if r.status >= 500 else None)

    def errors_since(self, n):
        return self.console_errors[n:]


def check(name, ok, detail=""):
    return {"name": name, "ok": bool(ok), "detail": str(detail)[:200]}


def make_images(tmp: Path, tag: str = ""):
    """Test images. With a tag, one corner pixel is varied so each benchmark
    run imports content-unique specimens (deterministic registry ids would
    otherwise resurrect prior runs' review state)."""
    from PIL import Image

    def _vary(img):
        if not tag:
            return img
        px = img.load()
        seed = sum(tag.encode()) % 251
        px[0, 0] = (seed, (seed * 7) % 256, (seed * 13) % 256)
        return img

    imgs = {}
    imgs["valid"] = tmp / "valid.png"
    _vary(Image.new("RGB", (360, 363), (190, 170, 165))).save(imgs["valid"])
    imgs["blank"] = tmp / "blank.png"
    Image.new("RGB", (300, 300), (128, 128, 128)).save(imgs["blank"])
    imgs["rgba"] = tmp / "rgba.png"
    Image.new("RGBA", (200, 200), (10, 20, 30, 40)).save(imgs["rgba"])
    imgs["gray"] = tmp / "gray.png"
    Image.new("L", (200, 200), 128).save(imgs["gray"])
    imgs["corrupt"] = tmp / "corrupt.png"
    imgs["corrupt"].write_bytes(b"definitely not image data at all" * 4)
    imgs["textpng"] = tmp / "fake.png"
    imgs["textpng"].write_bytes(b"plain text pretending to be a png file here" * 3)
    imgs["huge"] = tmp / "huge.png"
    Image.new("RGB", (5000, 5000), (5, 5, 5)).save(imgs["huge"])
    return imgs


async def import_file(page, path: Path):
    # Direct input assignment: identical change event and backend request as
    # the native dialog path, without dialog flakiness. The dialog opening
    # itself is covered once by G03-dialog-opens.
    # Returns the import-note text so callers can track created specimen ids.
    await page.evaluate("() => { document.querySelector('#import-note').textContent = ''; }")
    await page.set_input_files("#import-file", str(path))
    await page.wait_for_function(
        "() => document.querySelector('#import-note').textContent.length > 0",
        timeout=30000)
    return await page.text_content("#import-note")


class Bench:
    def __init__(self, page, base_url):
        self.page = page
        self.base_url = base_url
        self.hosts: set = set()
        self.ctx = Ctx(page)
        self.scenarios = []

    def scenario(self, sid, cat, weight):
        def deco(fn):
            self.scenarios.append({"id": sid, "cat": cat, "weight": weight, "fn": fn})
            return fn
        return deco

    async def goto_app(self):
        await self.page.goto(self.base_url + "/", wait_until="networkidle")
        await self.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5", timeout=TIMEOUT_UI)

    async def specimen_ids(self):
        return await self.page.eval_on_selector_all(
            "#specimens option", "els => els.map(e => e.textContent)")

    async def select_by_text(self, text):
        await self.page.select_option("#specimens", label=text)

    async def wait_findings(self, timeout=TIMEOUT_ANALYZE):
        await self.page.wait_for_function(
            "() => document.querySelectorAll('#findings .ev').length > 0", timeout=timeout)

    async def findings_count(self):
        return await self.page.eval_on_selector_all("#findings .ev", "els => els.length")

    async def scroll_findings(self, y):
        await self.page.evaluate(f"document.querySelector('#side').scrollTop = {y}")
        return await self.page.evaluate("document.querySelector('#side').scrollTop")

    async def job_text(self):
        return await self.page.text_content("#job")

    async def note_text(self):
        return await self.page.text_content("#note")

    async def run_all(self, progress_path=None, per_scenario_s=480):
        import asyncio
        import json as _json

        results = []
        for spec in self.scenarios:
            t0 = time.monotonic()
            if progress_path is not None:
                progress_path.write_text(_json.dumps({'running': spec['id'], 'done': len(results), 'of': len(self.scenarios)}) + chr(10))
            err0 = len(self.ctx.console_errors)
            try:
                checks = await asyncio.wait_for(spec["fn"](self), timeout=per_scenario_s)
                crashed = False
            except Exception as exc:  # noqa: BLE001 - a crash/timeout is a scored outcome
                checks = [check("no-crash", False, repr(exc)[:200])]
                crashed = True
                try:
                    shot = RESULTS / "shots" / f"{spec['id']}.png"
                    shot.parent.mkdir(parents=True, exist_ok=True)
                    await self.page.screenshot(path=str(shot))
                except Exception:  # noqa: BLE001
                    pass
            ok = sum(1 for c in checks if c["ok"])
            score = spec["weight"] * ok / max(1, len(checks))
            results.append({"id": spec["id"], "cat": spec["cat"], "weight": spec["weight"],
                            "checks": checks, "score": round(score, 2),
                            "ms": int((time.monotonic() - t0) * 1000),
                            "crashed": crashed,
                            "new_console_errors": self.ctx.errors_since(err0)})
            if progress_path is not None:
                progress_path.write_text(_json.dumps(
                    {"done": len(results), "of": len(self.scenarios),
                     "last": results[-1]["id"],
                     "score_so_far": round(sum(r["score"] for r in results), 1)}) + "\n")
        return results
