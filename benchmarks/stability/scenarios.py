"""Scenario definitions. Each returns ordered checks sharing the weight."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from bench import check


def register(B, IMG):
    S = B.scenario

    async def select_by_value(b, sid, timeout=15000):
        if not sid:
            raise RuntimeError("cannot select an empty specimen id")
        await b.page.wait_for_function(
            f"""() => [...document.querySelectorAll('#specimens option')]
                .some(o => o.value === '{sid}')""",
            timeout=timeout)
        await b.page.select_option("#specimens", value=sid)
        await b.page.wait_for_timeout(800)
        label = await b.page.text_content("#specimen-label")
        if sid not in (label or ""):
            raise RuntimeError(f"selection did not land on {sid}: {label[:60]}")

    async def import_note_id(b):
        txt = await b.page.text_content("#import-note")
        m = re.search(r"imported\s+([0-9a-f]{12})", txt or "")
        return m.group(1) if m else None

    async def catalog_snapshot(b):
        return await b.page.evaluate(
            """async () => (await (await fetch('/api/specimens?limit=500')).json())
                .map(item => ({id: item.id, label: `${item.source}: ${item.name}`}))""")

    async def v1_findings(b, sid):
        return await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/findings/${id}`)).json()).findings""", sid)

    def stable_findings(records):
        volatile = {"run_id", "created_unix", "review_id"}
        return [{key: value for key, value in record.items() if key not in volatile}
                for record in records]

    async def v1_export(b, sid):
        return await b.page.evaluate(
            """async id => await (await fetch(`/api/v1/export/${id}`)).json()""", sid)

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def stable_retrieval(body):
        if not isinstance(body, dict):
            return body
        stable = dict(body)
        stable["sets"] = {
            finding_id: {key: value for key, value in group.items()
                         if key != "retrieved_unix"}
            for finding_id, group in (body.get("sets") or {}).items()
        }
        return stable

    async def selection_snapshot(b):
        return await b.page.evaluate(
            """() => ({
                selected: document.querySelector('#specimens')?.value || '',
                label: document.querySelector('#specimen-label')?.textContent || '',
                findings: document.querySelector('#findings')?.innerHTML || '',
                counts: document.querySelector('#findings')?.dataset.counts || '',
                evidence: document.querySelector('#evidence')?.innerHTML || '',
                note: document.querySelector('#note')?.textContent || '',
                job: document.querySelector('#job')?.textContent || '',
                st_job: document.querySelector('#st-job')?.textContent || ''
            })""")

    async def rendered_v1_evidence(b):
        return await b.page.evaluate(
            """() => ({
                groups: [...document.querySelectorAll('#evidence .ev[data-fid]')]
                    .map(e => ({fid: e.dataset.fid || '', query: e.dataset.query || ''})),
                entries: [...document.querySelectorAll('#evidence input[data-v1-evidence]')]
                    .map(e => ({fid: e.dataset.fid || '', eid: e.dataset.eid || '',
                                go_id: e.dataset.goId || '', origin: e.dataset.origin || '',
                                rank: Number(e.dataset.rank), query: e.dataset.query || '',
                                checked: !!e.checked,
                                text: e.parentElement?.textContent || ''}))
            })""")

    def exported_v1_evidence_rows(bundle):
        rows = []
        groups = bundle.get("evidence", {}) if isinstance(bundle, dict) else {}
        for fid, group in groups.items():
            excluded = {entry.get("evidence_id") for entry in group.get("excluded", [])}
            for entry in [*(group.get("evidence", []) or []),
                          *(group.get("manual_adds", []) or [])]:
                origin = entry.get("origin") or (
                    f"auto-{group.get('rule')}" if entry in group.get("evidence", [])
                    else "human:unknown")
                rows.append({
                    "fid": fid, "eid": entry.get("evidence_id"),
                    "go_id": entry.get("go_id"), "origin": origin,
                    "rank": entry.get("rank"),
                    "query": entry.get("query") or group.get("query", ""),
                    "checked": entry.get("evidence_id") not in excluded,
                    "name": entry.get("name", ""),
                })
        return rows

    def independent_sealed_digests():
        # Keep the file names explicit here: the benchmark computes these
        # digests independently of the dashboard's provenance response.
        names = (
            "protocols/phase5_v1/freeze_manifest.json",
            "protocols/phase5_v2/freeze_manifest.json",
            "artifacts/phase5_comparison_v2/selection_manifest.json",
            "artifacts/phase5_final_v1/final_report.json",
        )
        root = Path(__file__).resolve().parents[2]
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}

    async def import_and_select(b, path, fixture_key=None):
        note = await _note(b, path)
        sid = await import_note_id(b)
        if not sid:
            raise RuntimeError(f"import did not return a specimen id: {note[:120]}")
        await select_by_value(b, sid)
        if fixture_key:
            b.ctx.bind_fixture(fixture_key, sid)
        return sid

    async def analyze_current(b, timeout=240000):
        """Run analysis for the selected v1 specimen and require completion."""
        if not await b.page.is_visible("#analyze"):
            raise RuntimeError("selected specimen has no v1 analyze action")
        await b.page.click("#analyze")
        await b.page.wait_for_function(
            "() => /analyze: (done|failed|cancelled)/.test(document.querySelector('#st-job').textContent || '')",
            timeout=timeout)
        state = await b.page.text_content("#st-job")
        if "analyze: done" not in state:
            raise RuntimeError(f"analysis did not complete: {state[:120]}")
        return state

    async def ensure_findings(b, timeout=240000):
        """Ensure the selected imported specimen has a nonempty findings list."""
        try:
            await b.wait_findings(timeout=10000)
        except Exception:  # noqa: BLE001 - an imported but unanalyzed specimen
            await analyze_current(b, timeout=timeout)
            await b.wait_findings(timeout=timeout)
        n = await b.findings_count()
        if n <= 0:
            raise RuntimeError("fixture has no findings")
        return n

    async def prepared_v1(b, image_key):
        """A v1 specimen with analyzed, bulk-approved findings and retrieved evidence."""
        from bench import import_file
        await import_file(b.page, IMG[image_key])
        await b.page.wait_for_function(
            "() => /imported [0-9a-f]{12}/.test(document.querySelector('#import-note').textContent || '')",
            timeout=30000)
        sid = await import_note_id(b)
        await select_by_value(b, sid)
        await ensure_findings(b)
        await b.page.click("#mode-auto")
        await b.page.wait_for_timeout(300)
        approve_all = b.page.locator("#approve-all")
        if await approve_all.count():
            await approve_all.click()
            await b.page.wait_for_function(
                "() => (document.querySelector('#st-job').textContent || '').includes('evidence attached automatically')",
                timeout=30000)
        else:
            await b.page.evaluate("id => v1RetrieveAll(id)", sid)
        await b.page.wait_for_function(
            "() => (document.querySelector('#evidence').textContent || '').includes('GO:')",
            timeout=30000)
        b.ctx.bind_fixture(image_key, sid)
        return sid

    async def review_fixture(b, image_key):
        """Create a fresh analyzed fixture with reviewer action controls."""
        sid = await import_and_select(b, IMG[image_key], image_key)
        await ensure_findings(b)
        await b.page.click("#mode-manual")
        await b.page.wait_for_timeout(300)
        if await b.page.locator("#findings button[data-act]").count() == 0:
            raise RuntimeError("review fixture has no finding actions")
        return sid

    async def synth_done(b, timeout=240000):
        before_jobs = await b.page.evaluate("() => (S.v1JobIds || []).length")
        await b.page.click("#synthesize")
        await b.page.wait_for_function(
            "before => (S.v1JobIds || []).length > before || /v1 failed:/.test(document.querySelector('#job').textContent || '')",
            arg=before_jobs, timeout=30000)
        submitted = await b.page.evaluate("before => (S.v1JobIds || []).slice(before)", before_jobs)
        if not submitted:
            raise RuntimeError("synthesis did not submit a new job")
        job_id = submitted[-1]
        await b.page.wait_for_function(
            "() => { const t = document.querySelector('#job').textContent || ''; return /done|failed|cancelled|v1 failed/.test(t); }",
            timeout=timeout)
        job_text, note = await b.job_text(), await b.note_text()
        backend_job = await b.page.evaluate(
            """async id => await (await fetch(`/api/jobs/${id}`)).json()""", job_id)
        if backend_job.get("state") not in ("done", "failed", "cancelled"):
            raise RuntimeError(f"browser terminal text did not match backend job: {backend_job}")
        if backend_job.get("state") == "done" and backend_job.get("result", {}).get("note") != note:
            raise RuntimeError("browser note does not match the submitted synthesis result")
        return job_text, note

    # ---- startup (6) ----
    @S("S01-clean-launch", "startup", 2)
    async def _(b):
        await b.goto_app()
        title = await b.page.title()
        n = await b.page.eval_on_selector_all("#specimens option", "e=>e.length")
        return [check("title", "ren" in title.lower(), title),
                check("specimen-list", n > 10, n),
                check("no-console-errors", not b.ctx.console_errors, b.ctx.console_errors[:2])]

    @S("S02-reload-stable", "startup", 2)
    async def _(b):
        before = await catalog_snapshot(b)
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        after = await catalog_snapshot(b)
        return [check("same-list", before == after, (before[:3], after[:3], len(before), len(after))),
                check("canvas-present", await b.page.is_visible("#canvas"))]

    @S("S03-dead-backend", "startup", 1)
    async def _(b):
        b.ctx.expect_failed_request("127.0.0.1:8099")
        result = await b.page.evaluate(
            """() => new Promise(resolve => {
                const probe = document.createElement('img');
                probe.onload = () => { probe.remove(); resolve({accepted: true}); };
                probe.onerror = () => { probe.remove(); resolve({accepted: false, error: 'resource error'}); };
                probe.src = 'http://127.0.0.1:8099/';
                document.body.appendChild(probe);
            })""")
        message = str(result)
        refused = not result.get("accepted") and "error" in result
        return [check("connection-refused", refused, message[:120])]

    @S("S04-synth-indicator-down", "startup", 1)
    async def _(b):
        await b.goto_app()
        await b.page.evaluate("() => fetch('/api/server/stop', {method: 'POST'})")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('down')",
            timeout=15000)
        await b.page.reload(wait_until="networkidle")
        label = await b.page.text_content("#server-label")
        ok = "down" in label.lower()
        await b.page.evaluate("() => fetch('/api/server/start', {method: 'POST'})")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('up')",
            timeout=180000)
        final = await b.page.text_content("#server-label")
        return [check("down-and-restored", ok and "up" in final.lower(),
                      (label, final))]

    # ---- import (10) ----
    async def _note(b, path, timeout=30000):
        from bench import import_file
        await import_file(b.page, path)
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.length > 0",
            timeout=timeout)
        return await b.page.text_content("#import-note")

    async def rejected_import(b, path, timeout=30000):
        before = await catalog_snapshot(b)
        note = await _note(b, path, timeout=timeout)
        after = await catalog_snapshot(b)
        return note, before, after

    @S("I01-valid-png", "import", 2)
    async def _(b):
        note = await _note(b, IMG["valid"])
        sid = await import_note_id(b)
        catalog = await catalog_snapshot(b)
        return [check("imported-msg", "imported" in note and bool(sid), (sid, note)),
                check("record-created", sid in {item["id"] for item in catalog},
                      (sid, len(catalog)))]

    @S("I02-duplicate-import", "import", 1.5)
    async def _(b):
        n0 = len(await b.specimen_ids())
        first_note = await _note(b, IMG["duplicate_i02"])
        first_sid = await import_note_id(b)
        if not first_sid:
            raise RuntimeError(f"initial duplicate fixture import did not return an id: {first_note[:80]}")
        n1 = len(await b.specimen_ids())
        b.ctx.bind_fixture("duplicate_i02", first_sid)
        second_note = await _note(b, IMG["duplicate_i02"])
        second_sid = await import_note_id(b)
        n2 = len(await b.specimen_ids())
        return [check("first-import-recorded", "imported" in first_note and first_sid in first_note,
                      (first_sid, n0, n1, first_note[:60])),
                check("same-specimen", first_sid == second_sid,
                      (first_sid, second_sid, second_note[:60])),
                check("duplicate-count-stable", n2 == n1, (n1, n2))]

    @S("I03-corrupt-rejected", "import", 1.5)
    async def _(b):
        note, before, after = await rejected_import(b, IMG["corrupt"])
        return [check("rejected-msg", "fail" in note.lower() or "reject" in note.lower(), note),
                check("no-record-created", before == after, (len(before), len(after)))]

    @S("I04-wrong-extension", "import", 1)
    async def _(b):
        note, before, after = await rejected_import(b, IMG["textpng"])
        return [check("rejected-msg", "fail" in note.lower() or "reject" in note.lower(), note),
                check("no-record-created", before == after, (len(before), len(after)))]

    @S("I05-rgba-accepted", "import", 1)
    async def _(b):
        note = await _note(b, IMG["rgba"])
        sid = await import_note_id(b)
        catalog = await catalog_snapshot(b)
        return [check("accepted", "imported" in note and bool(sid), (sid, note)),
                check("record-created", sid in {item["id"] for item in catalog},
                      (sid, len(catalog)))]

    @S("I06-grayscale-accepted", "import", 1)
    async def _(b):
        note = await _note(b, IMG["gray"])
        sid = await import_note_id(b)
        catalog = await catalog_snapshot(b)
        return [check("accepted", "imported" in note and bool(sid), (sid, note)),
                check("record-created", sid in {item["id"] for item in catalog},
                      (sid, len(catalog)))]

    @S("I07-oversized-rejected", "import", 1)
    async def _(b):
        note, before, after = await rejected_import(b, IMG["huge"], timeout=60000)
        low = note.lower()
        return [check("rejected-msg", "fail" in low or "reject" in low or "large" in low, note[:80]),
                check("no-record-created", before == after, (len(before), len(after)))]

    @S("I08-repeated-imports", "import", 1)
    async def _(b):
        ids = []
        first_note = await _note(b, IMG["repeated_i08"])
        first_sid = await import_note_id(b)
        if not first_sid:
            raise RuntimeError(f"repeated import did not return an id: {first_note[:80]}")
        ids.append(first_sid)
        n1 = len(await b.specimen_ids())
        b.ctx.bind_fixture("repeated_i08", first_sid)
        for _ in range(2):
            note = await _note(b, IMG["repeated_i08"])
            sid = await import_note_id(b)
            if not sid:
                raise RuntimeError(f"repeated import did not return an id: {note[:80]}")
            ids.append(sid)
        n2 = len(await b.specimen_ids())
        return [check("first-import-recorded", "imported" in first_note and first_sid in first_note,
                      (first_sid, n1, first_note[:60])),
                check("same-specimen", len(set(ids)) == 1, ids),
                check("repeated-count-stable", n2 == n1, (n1, n2)),
                check("final-note-accepted", "imported" in (await b.page.text_content("#import-note")),
                      await b.page.text_content("#import-note"))]

    # ---- vision (10) ----
    async def _analyze_current(b, timeout=240000):
        await b.page.click("#analyze")
        await b.page.wait_for_function(
            "() => /done|failed/.test(document.querySelector('#st-job').textContent || '')",
            timeout=timeout)
        return await b.page.text_content("#st-job")

    @S("V01-analyze-findings", "vision", 3)
    async def _(b):
        await b.goto_app()
        sid = await prepared_v1(b, "txl_v01")
        n = await b.findings_count()
        return [check("sid-selected", bool(sid), sid),
                check("findings-rendered", n > 0, n)]

    @S("V02-blank-zero-findings", "vision", 2)
    async def _(b):
        from bench import import_file
        await import_file(b.page, IMG["blank_v02"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.length > 0", timeout=30000)
        sid = await import_note_id(b)
        await select_by_value(b, sid)
        b.ctx.bind_fixture("blank_v02", sid)
        st = await _analyze_current(b)
        body = await b.page.text_content("#findings")
        n = await b.findings_count()
        return [check("job-done", "done" in st, st),
                check("empty-state", n == 0 and "no " in body.lower(), body[:80])]

    @S("V03-reanalyze-stable", "vision", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_v03")
        n0 = await b.findings_count()
        before = await v1_findings(b, sid)
        st = await analyze_current(b)
        n1 = await b.findings_count()
        after = await v1_findings(b, sid)
        stable_before = stable_findings(before)
        stable_after = stable_findings(after)
        return [check("job-done", "analyze: done" in st, st),
                check("initial-findings", n0 > 0, n0),
                check("same-count", n0 == n1 and n1 > 0, (n0, n1)),
                check("same-finding-content", stable_before == stable_after,
                      (stable_before, stable_after))]

    @S("V04-refresh-persists", "vision", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_v04")
        n0 = await b.findings_count()
        catalog_before = await catalog_snapshot(b)
        findings_before = await v1_findings(b, sid)
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        await select_by_value(b, sid)
        await ensure_findings(b)
        n1 = await b.findings_count()
        catalog_after = await catalog_snapshot(b)
        findings_after = await v1_findings(b, sid)
        return [check("list-back", len(catalog_after) > 0),
                check("same-selection", await b.page.input_value("#specimens") == sid, sid),
                check("count-persists", n0 == n1 and n1 > 0, (n0, n1)),
                check("catalog-content-persists", catalog_before == catalog_after,
                      (len(catalog_before), len(catalog_after))),
                check("finding-content-persists",
                      stable_findings(findings_before) == stable_findings(findings_after),
                      (findings_before, findings_after))]

    @S("V05-navigate-mid-run", "vision", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_v05")
        await b.page.evaluate("() => { S.analysisJobIds = []; }")
        await b.page.click("#analyze")
        await b.page.wait_for_function(
            "() => (S.analysisJobIds || []).length > 0",
            timeout=30000)
        job_id = (await b.page.evaluate("() => S.analysisJobIds || []"))[-1]
        job_before_navigation = await b.page.evaluate(
            """async id => await (await fetch(`/api/jobs/${id}`)).json()""", job_id)
        nonterminal_before_navigation = job_before_navigation.get("state") in ("queued", "running")
        options = await b.page.eval_on_selector_all(
            "#specimens option", "function(els, excluded) { return els.map(e => e.value).filter(v => v !== excluded && v.startsWith('txl-')); }", sid)
        target = options[0] if options else None
        if not target:
            raise RuntimeError("no alternate specimen available for mid-run navigation")
        await select_by_value(b, target)
        await b.page.wait_for_function(
            "() => !(document.querySelector('#findings').textContent || '').includes('loading findings')",
            timeout=30000)
        target_before = await selection_snapshot(b)
        target_expected = await b.page.evaluate(
            """async id => await (await fetch(`/api/specimens/${id}/overlays`)).json()""", target)
        errs = [e for e in b.ctx.console_errors if "uncaught" in e.lower()]
        state = await b.page.evaluate(
            """async id => { for (let i = 0; i < 240; i++) {
                const r = await fetch(`/api/jobs/${id}`); const j = await r.json();
                if (["done", "failed", "cancelled"].includes(j.state)) return j.state;
                await new Promise(resolve => setTimeout(resolve, 1000));
            } return "timeout"; }""", job_id)
        await b.page.wait_for_timeout(1000)
        target_after = await selection_snapshot(b)
        old_job_id = str(job_id)
        uncontaminated = (target_before == target_after and target_after["selected"] == target
                          and json.loads(target_after["counts"] or "{}") == target_expected.get("counts", {})
                          and target_before["note"] == "" and target_before["evidence"] == ""
                          and old_job_id not in target_after["job"]
                          and old_job_id not in target_after["note"]
                          and "analyze: done" not in target_after["st_job"])
        return [check("navigated", await b.page.input_value("#specimens") == target, target),
                check("job-nonterminal-before-navigation", nonterminal_before_navigation,
                      job_before_navigation),
                check("no-uncached-crash", not errs, errs[:1]),
                check("analysis-job-settled", state in ("done", "failed", "cancelled"), state),
                check("new-selection-uncontaminated", uncontaminated,
                      {"before": target_before, "after": target_after, "old_job": old_job_id})]

    # ---- findings (12) ----
    @S("F01-approve-scrolled", "findings", 2)
    async def _(b):
        await b.goto_app()
        await b.page.set_viewport_size({"width": 1440, "height": 500})
        try:
            sid = await review_fixture(b, "txl_review")
            n0 = await b.findings_count()
            reviews_before = (await v1_export(b, sid)).get("reviews", [])
            await b.scroll_findings(600)
            btns = b.page.locator("#findings button[data-act='confirm']")
            nb = await btns.count()
            if nb == 0:
                raise RuntimeError("review fixture has no confirm action")
            # Use a lower row so bringing the target into view leaves the
            # findings pane genuinely scrolled before the app re-renders it.
            btn = btns.nth(nb - 1)
            # Bring the row into view first, exactly as a user would: this keeps
            # the driver's own scroll out of the measurement so the assertion
            # covers the app's re-render behavior only.
            await btn.scroll_into_view_if_needed()
            await b.page.wait_for_timeout(400)
            y0 = await b.page.evaluate("document.querySelector('#side').scrollTop")
            fid = await btn.get_attribute("data-fid")
            if not fid:
                raise RuntimeError("confirm action has no finding id")
            await btn.click(force=True)
            await b.page.wait_for_timeout(2000)
            await b.page.wait_for_function(
                "() => (document.querySelector('#evidence').textContent || '').includes('GO:')",
                timeout=30000)
            y1 = await b.page.evaluate("document.querySelector('#side').scrollTop")
            n1 = await b.findings_count()
            mode = await b.page.evaluate("() => document.querySelector('#findings').innerHTML.length")
            row = b.page.locator(f"#findings .ev[data-fid='{fid}']")
            state = await row.text_content() if await row.count() else ""
            reviews_after = (await v1_export(b, sid)).get("reviews", [])
            target_reviews = [r for r in reviews_after[len(reviews_before):]
                              if r.get("finding_id") == fid and r.get("action") == "confirm"]
            return [check("had-scroll", y0 > 50, (y0, y1)),
                    check("confirm-changed-target", "confirmed" in state.lower(), state[:120]),
                    check("scroll-preserved", abs(y1 - y0) < 40, (y0, y1, n0, n1, mode)),
                    check("confirm-persisted-on-target", len(target_reviews) == 1,
                          {"fid": fid, "reviews": target_reviews})]
        finally:
            await b.page.set_viewport_size({"width": 1440, "height": 900})

    @S("F02-reject-scrolled", "findings", 1.5)
    async def _(b):
        sid = await review_fixture(b, "txl_reject")
        reviews_before = (await v1_export(b, sid)).get("reviews", [])
        await b.scroll_findings(600)
        btns = b.page.locator("#findings button[data-act='reject']")
        if await btns.count() == 0:
            raise RuntimeError("review fixture has no reject action")
        btn = btns.last
        await btn.scroll_into_view_if_needed()
        await b.page.wait_for_timeout(400)
        y0 = await b.page.evaluate("document.querySelector('#side').scrollTop")
        fid = await btn.get_attribute("data-fid")
        if not fid:
            raise RuntimeError("reject action has no finding id")
        await btn.evaluate("(el) => el.click()")
        await b.page.wait_for_function(
            "fid => ![...document.querySelectorAll('#findings .ev')].some(e => e.dataset.fid === fid)",
            arg=fid, timeout=30000)
        y1 = await b.page.evaluate("document.querySelector('#side').scrollTop")
        remaining = await b.page.eval_on_selector_all(
            "#findings .ev[data-fid]", "els => els.map(e => e.dataset.fid)")
        reviews_after = (await v1_export(b, sid)).get("reviews", [])
        target_reviews = [r for r in reviews_after[len(reviews_before):]
                          if r.get("finding_id") == fid and r.get("action") == "reject"]
        return [check("had-scroll", y0 > 50, (y0, y1)),
                check("reject-changed-target", bool(fid) and fid not in remaining, (fid, remaining)),
                check("scroll-preserved", y1 > 50 and y1 <= y0, (y0, y1)),
                check("reject-persisted-on-target", len(target_reviews) == 1,
                      {"fid": fid, "reviews": target_reviews})]

    @S("F03-bulk-approve", "findings", 1.5)
    async def _(b):
        sid = await import_and_select(b, IMG["txl_bulk"], "txl_bulk")
        await ensure_findings(b)
        await b.page.click("#mode-auto")
        await b.page.wait_for_selector("#approve-all", timeout=15000)
        await b.page.click("#approve-all")
        await b.page.wait_for_function(
            "() => (document.querySelector('#st-job').textContent || '').includes('auto-approved')",
            timeout=30000)
        await b.page.wait_for_function(
            "() => (document.querySelector('#evidence').textContent || '').includes('GO:')",
            timeout=30000)
        txt = await b.page.text_content("#st-job")
        left = await b.page.evaluate(
            "() => [...document.querySelectorAll('#findings .ev')]"
            ".filter(e => (e.textContent || '').includes('unreviewed')).length")
        records = await v1_findings(b, sid)
        return [check("bulk-ran", "auto-approved" in txt, txt[:80]),
                check("none-unreviewed", left == 0 and records
                      and all(f.get("review_state") == "confirmed" for f in records),
                      {"dom": left, "findings": records})]

    @S("F04-mode-switch", "findings", 1.5)
    async def _(b):
        sid = await review_fixture(b, "txl_mode")
        finding_ids = [f["finding_id"] for f in await v1_findings(b, sid)]
        transitions = []
        for _ in range(3):
            await b.page.click("#mode-manual")
            await b.page.wait_for_function(
                "() => S.reviewMode === 'manual' && document.querySelector('#mode-manual')?.classList.contains('primary')",
                timeout=15000)
            manual_html = await b.page.inner_html("#findings")
            manual_has_bulk = await b.page.locator("#approve-all").count() > 0
            await b.page.click("#mode-auto")
            await b.page.wait_for_function(
                "() => S.reviewMode === 'auto' && document.querySelector('#mode-auto')?.classList.contains('primary')",
                timeout=15000)
            auto_html = await b.page.inner_html("#findings")
            transitions.append({"manual": manual_html, "auto": auto_html,
                                "manual_has_bulk": manual_has_bulk,
                                "auto_has_bulk": await b.page.locator("#approve-all").count() > 0})
        n = await b.findings_count()
        changed = all(item["manual"] != item["auto"] for item in transitions)
        controls = all(not item["manual_has_bulk"] and item["auto_has_bulk"]
                       for item in transitions)
        ids_after = [f["finding_id"] for f in await v1_findings(b, sid)]
        return [check("mode-toggled-and-settled", len(transitions) == 3 and changed,
                      [{"changed": item["manual"] != item["auto"]} for item in transitions]),
                check("mode-controls-change", controls, transitions),
                check("list-intact", n == len(finding_ids) and n > 0 and ids_after == finding_ids,
                      (finding_ids, ids_after, n))]

    @S("F05-alter-after-auto", "findings", 1)
    async def _(b):
        await prepared_v1(b, "txl_f05")
        n0 = await b.findings_count()
        btns = b.page.locator("#findings button[data-act='reject']")
        if await btns.count() == 0:
            raise RuntimeError("prepared fixture has no reject action")
        fid = await btns.first.get_attribute("data-fid")
        if not fid:
            raise RuntimeError("reject action has no finding id")
        await btns.first.click()
        await b.page.wait_for_function(
            "fid => ![...document.querySelectorAll('#findings .ev')].some(e => e.dataset.fid === fid)",
            arg=fid, timeout=30000)
        n1 = await b.findings_count()
        return [check("finding-removed", n1 == n0 - 1 and n1 < n0, (n0, n1)),
                check("rejected-target-removed", fid not in await b.page.eval_on_selector_all(
                    "#findings .ev[data-fid]", "els => els.map(e => e.dataset.fid)"), fid)]

    @S("F06-double-approve", "findings", 1)
    async def _(b):
        sid = await review_fixture(b, "txl_double")
        before_reviews = (await v1_export(b, sid)).get("reviews", [])
        btn = b.page.locator("#findings button[data-act='confirm']").first
        if await btn.count() == 0:
            raise RuntimeError("review fixture has no confirm action")
        fid = await btn.get_attribute("data-fid")
        await btn.dblclick()
        await b.page.wait_for_timeout(2500)
        await b.page.wait_for_function(
            "() => (document.querySelector('#evidence').textContent || '').includes('GO:')",
            timeout=30000)
        row = b.page.locator(f"#findings .ev[data-fid='{fid}']")
        state = await row.text_content() if await row.count() else ""
        n = await b.findings_count()
        after_reviews = (await v1_export(b, sid)).get("reviews", [])
        target_reviews = [r for r in after_reviews[len(before_reviews):]
                          if r.get("finding_id") == fid and r.get("action") == "confirm"]
        effective = await v1_findings(b, sid)
        target_effective = [f for f in effective if f.get("finding_id") == fid]
        return [check("action-settled", "confirmed" in state.lower(), state[:120]),
                check("single-row-remains", n > 0 and await row.count() == 1, n),
                check("single-durable-effect", len(target_reviews) == 1 and len(target_effective) == 1,
                      {"fid": fid, "reviews": target_reviews, "effective": target_effective})]

    @S("F07-rows-have-ids", "findings", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_f07")
        expected = [f["finding_id"] for f in await v1_findings(b, sid)]
        row_ids = await b.page.eval_on_selector_all(
            "#findings .ev", "els => els.map(e => e.dataset.fid || '')")
        action_ids = await b.page.eval_on_selector_all(
            "#findings button[data-act]", "els => els.map(e => e.dataset.fid || '')")
        usable_rows = (bool(expected) and len(row_ids) == len(set(row_ids))
                       and all(row_ids) and set(row_ids) == set(expected))
        usable_actions = (len(action_ids) == len(expected) * 2
                          and len(action_ids) == len(set(action_ids)) * 2
                          and set(action_ids) == set(expected))
        return [check("rows-identified", usable_rows, (row_ids, expected)),
                check("actions-present", usable_actions, (action_ids, expected))]

    @S("F08-many-usable", "findings", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_f08")
        n = await b.findings_count()
        side_h = await b.page.evaluate("document.querySelector('#side').clientHeight")
        side_scroll = await b.page.evaluate("document.querySelector('#side').scrollHeight")
        rows = await b.page.eval_on_selector_all(
            "#findings .ev", "els => els.map(e => ({fid: e.dataset.fid || '', text: e.textContent || ''}))")
        actual = await v1_findings(b, sid)
        row_ids = [row["fid"] for row in rows]
        many = (len(actual) >= 3 and len(rows) == len(actual)
                and len(row_ids) == len(set(row_ids)) and all(row_ids)
                and set(row_ids) == {f["finding_id"] for f in actual}
                and all(row["text"].strip() for row in rows))
        return [check("many-rows-present", many, {"rows": rows, "findings": actual}),
                check("scrollable", many and side_scroll > side_h and side_scroll > 100,
                      (side_scroll, side_h, len(rows)))]

    @S("F09-empty-message", "findings", 1.5)
    async def _(b):
        # Import and select by the returned content id.  Registry deduplication
        # preserves the first filename, so a display-name lookup can resolve
        # to an old fixture after a prior benchmark run.
        note = await _note(b, IMG["blank_f09"])
        sid = await import_note_id(b)
        if not sid:
            raise RuntimeError(f"blank fixture import did not return an id: {note}")
        await select_by_value(b, sid)
        b.ctx.bind_fixture("blank_f09", sid)
        st = await analyze_current(b)
        await b.page.wait_for_function(
            "() => /no findings|no overlay findings/.test(document.querySelector('#findings').textContent || '')",
            timeout=30000)
        body = await b.page.text_content("#findings")
        return [check("analysis-done", "analyze: done" in st, st),
                check("empty-shown", "no findings" in body.lower(), body[:80])]

    # ---- retrieval (12) ----
    @S("R01-auto-no-typing", "retrieval", 3)
    async def _(b):
        sid = await prepared_v1(b, "txl_r01")
        await b.page.fill("#query", "")
        html = await b.page.inner_html("#evidence")
        typed = await b.page.input_value("#query")
        links = await b.page.eval_on_selector_all(
            "#evidence .ev[data-fid]", "els => els.map(e => e.dataset.fid)")
        findings = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/findings/${id}`)).json()).findings
                .map(f => f.finding_id)""", sid)
        return [check("evidence-present", "GO:" in html and "automatic derivation" in html, html[:100]),
                check("findings-grounded", links and sorted(set(links)) == sorted(set(findings))
                      and len(links) == len(findings), (links, findings)),
                check("no-typing-needed", typed == "", typed)]

    @S("R02-multiple-findings", "retrieval", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_r02")
        html = await b.page.inner_html("#evidence")
        fids = await b.page.eval_on_selector_all(
            "#evidence .ev[data-fid]", "els => els.map(e => e.dataset.fid || '')")
        findings = await v1_findings(b, sid)
        finding_ids = [f["finding_id"] for f in findings]
        return [check("evidence-rendered", len(html) > 50 and "GO:" in html, len(html)),
                check("finding-links", len(findings) > 1 and len(fids) == len(findings)
                      and len(fids) == len(set(fids)) and all(fids)
                      and set(fids) == set(finding_ids),
                      (sid, finding_ids, fids))]

    @S("R03-rejected-excluded", "retrieval", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_r03")
        before = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/findings/${id}`)).json()).findings""", sid)
        before_ids = {finding["finding_id"] for finding in before}
        btn = b.page.locator("#findings button[data-act='reject']").first
        if await btn.count() == 0:
            raise RuntimeError("prepared fixture has no reject action")
        fid = await btn.get_attribute("data-fid")
        retained_ids = before_ids - {fid}
        await btn.click()
        await b.page.wait_for_function(
            "fid => ![...document.querySelectorAll('#findings [data-fid]')].some(e => e.dataset.fid === fid)",
            arg=fid, timeout=30000)
        await b.page.evaluate("id => v1RetrieveAll(id)", sid)
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#evidence .ev[data-fid]').length > 0", timeout=30000)
        exported = await b.page.evaluate(
            """async id => await (await fetch(`/api/v1/export/${id}`)).json()""", sid)
        evidence = exported.get("evidence", {})
        evidence_fids = set(evidence)
        retained_evidence = {key: value for key, value in evidence.items() if key in retained_ids}
        retained_ok = retained_ids == set(retained_evidence) and all(
            bool(group.get("evidence"))
            and all(entry.get("finding_id") == key for entry in group.get("evidence", []))
            for key, group in retained_evidence.items())
        return [check("rejected-finding-excluded", fid not in evidence_fids and fid not in retained_ids,
                       (fid, sorted(evidence_fids), sorted(retained_ids))),
                check("non-rejected-evidence-retained", retained_ok,
                      (sorted(retained_ids), {key: len(value.get("evidence", [])) for key, value in retained_evidence.items()}))]

    @S("R04-manual-override", "retrieval", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_r04")
        automatic_html = await b.page.inner_html("#evidence")
        if "automatic derivation" not in automatic_html:
            raise RuntimeError("manual override has no automatic baseline to replace")
        automatic = await v1_export(b, sid)
        automatic_entries = [entry for group in automatic.get("evidence", {}).values()
                             for entry in group.get("evidence", [])]
        automatic_ids = [entry.get("go_id") for entry in automatic_entries]
        async def manual_query(query):
            await b.page.fill("#query", query)
            await b.page.click("#retrieve")
            await b.page.wait_for_function(
                "q => S.lastRetrieval?.query === q && "
                "(document.querySelector('#evidence').textContent || '').includes('rank order preserved')",
                arg=query, timeout=30000)
            response = await b.page.evaluate("() => S.lastRetrieval")
            shown = await b.page.eval_on_selector_all(
                "#evidence .ev[data-go-id]",
                "els => els.map(e => ({go_id: e.dataset.goId, rank: Number(e.dataset.rank), "
                "query: e.dataset.query, text: e.textContent || ''}))")
            entries = (response or {}).get("entries", [])
            bound = (len(shown) == len(entries[:8])
                     and all(row["go_id"] == entry.get("go_id")
                             and row["rank"] == entry.get("rank")
                             and row["query"] == entry.get("query")
                             and entry.get("name", "") in row["text"]
                             and entry.get("definition", "")[:160] in row["text"]
                             for entry, row in zip(entries[:8], shown)))
            return response, shown, bound

        first, first_shown, first_bound = await manual_query("leukocyte")
        second, second_shown, second_bound = await manual_query("platelet")
        third, third_shown, third_bound = await manual_query("leukocyte")
        html = await b.page.inner_html("#evidence")
        query = await b.page.input_value("#query")
        first_entries = (first or {}).get("entries", [])
        second_entries = (second or {}).get("entries", [])
        third_entries = (third or {}).get("entries", [])
        # Compare the complete content fields while ignoring only the query
        # string: two different echoed labels are not proof that retrieval
        # used the input.  The returned term identities and definitions must
        # change with the query.
        def content(response):
            return {"mode": response.get("mode"), "entries": [
                {key: entry.get(key) for key in ("go_id", "name", "definition", "rank", "mode")}
                for entry in response.get("entries", [])]}

        content_differs = canonical(content(first)) != canonical(content(second))
        same_query_content = canonical(first) == canonical(third)
        ids_differ = {entry.get("go_id") for entry in first_entries} != {
            entry.get("go_id") for entry in second_entries}
        manual_ids = [entry.get("go_id") for entry in second_entries]
        causal = (first and second and third
                  and first.get("query") == "leukocyte"
                  and second.get("query") == "platelet"
                  and first_entries and second_entries
                  and all(entry.get("query") == "leukocyte" for entry in first_entries)
                  and all(entry.get("query") == "platelet" for entry in second_entries)
                  and all(entry.get("query") == "leukocyte" for entry in third_entries)
                  and all(entry.get("go_id") and entry.get("name") and entry.get("definition")
                          for entry in [*first_entries, *second_entries, *third_entries])
                  and content_differs
                  and same_query_content
                  and ids_differ
                  and set(manual_ids) != set(automatic_ids)
                  and first_bound and second_bound and third_bound
                  and "automatic derivation" not in html)
        return [check("manual-results-replace-auto", causal,
                       {"automatic_ids": automatic_ids, "first": first,
                        "second": second, "first_shown": first_shown,
                        "second_shown": second_shown, "third": third,
                        "third_shown": third_shown, "content_differs": content_differs,
                        "same_query_content": same_query_content,
                        "ids_differ": ids_differ,
                        "html": html[:160]}),
                check("query-retained", query == "leukocyte"
                      and first.get("query") == "leukocyte"
                      and second.get("query") == "platelet"
                      and third.get("query") == "leukocyte", query)]

    @S("R05-retry-same", "retrieval", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_r05")
        first = await b.page.evaluate(
            """async id => { const r = await fetch(`/api/v1/retrieve/${id}`, {method: 'POST'});
                return {status: r.status, body: await r.json()}; }""", sid)
        second = await b.page.evaluate(
            """async id => { const r = await fetch(`/api/v1/retrieve/${id}`, {method: 'POST'});
                return {status: r.status, body: await r.json()}; }""", sid)
        same = canonical(stable_retrieval(first.get("body"))) == canonical(stable_retrieval(second.get("body")))
        return [check("first-retrieve-ok", first["status"] == 200 and first["body"].get("evidence", 0) > 0
                      and first["body"].get("sets"), first),
                check("retry-retrieve-identical", second["status"] == 200 and same
                      and second["body"].get("sets"),
                      {"first": first, "second": second})]

    @S("R06-unconfirmed-blocked", "retrieval", 1.5)
    async def _(b):
        sid = await import_and_select(b, IMG["txl_r06"], "txl_r06")
        analysis = await analyze_current(b)
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#findings .ev').length > 0",
            timeout=240000)
        states = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/findings/${id}`)).json()).findings
                .map(f => ({id: f.finding_id, state: f.review_state}))""", sid)
        if not states:
            raise RuntimeError("unconfirmed fixture analysis produced no findings")
        all_unconfirmed = all(f["state"] == "unreviewed" for f in states)
        if not all_unconfirmed:
            raise RuntimeError(f"fixture did not start with unreviewed findings: {states}")
        await b.page.evaluate("() => { S.v1JobIds = []; }")
        await b.page.click("#synthesize")
        await b.page.wait_for_function(
            "() => /v1 failed|failed:/.test(document.querySelector('#job').textContent || '')",
            timeout=30000)
        txt = await b.job_text()
        submitted = await b.page.evaluate("() => S.v1JobIds || []")
        return [check("analysis-done", "analyze: done" in analysis, analysis),
                check("findings-unconfirmed", len(states) > 0 and all_unconfirmed,
                      states),
                check("blocked-by-confirmation-guard",
                      "v1 failed" in txt and "409" in txt and "no confirmed findings" in txt,
                      f"{sid}: {txt[:120]}"),
                check("synthesis-not-submitted", submitted == [], submitted)]

    @S("R07-rapid-double-retrieve", "retrieval", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_r07")
        responses = await b.page.evaluate(
            """async id => await Promise.all([1, 2].map(async () => {
                const r = await fetch(`/api/v1/retrieve/${id}`, {method: 'POST'});
                return {status: r.status, body: await r.json()};
            }))""", sid)
        same = (len(responses) == 2 and
                canonical(stable_retrieval(responses[0].get("body"))) ==
                canonical(stable_retrieval(responses[1].get("body"))))
        return [check("both-retrieves-ok", len(responses) == 2 and all(r["status"] == 200 for r in responses)
                      and all(r["body"].get("sets") for r in responses), responses),
                check("same-retrieval-content", same and responses[0]["body"].get("evidence", 0) > 0,
                      {"responses": responses})]

    # ---- evidence (6) ----
    @S("E01-query-origin-badges", "evidence", 2)
    async def _(b):
        await prepared_v1(b, "txl_e01")
        html = await b.page.inner_html("#evidence")
        return [check("content-present", len(html) > 50 and "GO:" in html, len(html)),
                check("origin-badge", "[auto]" in html and "automatic derivation" in html, html[:120])]

    @S("E02-exclusion", "evidence", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_e02")
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#evidence input[data-v1-evidence]').length > 0",
            timeout=30000)
        boxes = await b.page.query_selector_all("#evidence input[data-v1-evidence]")
        if not boxes:
            raise RuntimeError("automatic retrieval returned no evidence checkboxes")
        box = boxes[0]
        fid = await box.get_attribute("data-fid")
        eid = await box.get_attribute("data-eid")
        before = await box.is_checked()
        if not fid or not eid or not before:
            raise RuntimeError(f"evidence exclusion target is not an included automatic item: {(fid, eid, before)}")
        await box.click()
        await b.page.wait_for_function(
            """target => fetch(`/api/v1/export/${target.sid}`).then(r => r.json()).then(bundle =>
                !!bundle.evidence?.[target.fid]?.excluded?.some(e => e.evidence_id === target.eid))""",
            arg={"sid": sid, "fid": fid, "eid": eid}, timeout=30000)
        persisted_before = await b.page.evaluate(
            """target => fetch(`/api/v1/export/${target.sid}`).then(r => r.json()).then(bundle =>
                bundle.evidence?.[target.fid]?.excluded?.some(e => e.evidence_id === target.eid) || false)""",
            {"sid": sid, "fid": fid, "eid": eid})
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_selector("#specimens")
        await select_by_value(b, sid)
        await b.page.evaluate("id => v1RetrieveAll(id)", sid)
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#evidence input[data-v1-evidence]').length > 0",
            timeout=30000)
        after_box = b.page.locator(f"#evidence input[data-v1-evidence][data-fid='{fid}'][data-eid='{eid}']")
        after = await after_box.is_checked()
        persisted_after = await b.page.evaluate(
            """target => fetch(`/api/v1/export/${target.sid}`).then(r => r.json()).then(bundle =>
                bundle.evidence?.[target.fid]?.excluded?.some(e => e.evidence_id === target.eid) || false)""",
            {"sid": sid, "fid": fid, "eid": eid})
        return [check("toggle-requested-exclusion", before and not after, (before, after, fid, eid)),
                check("exclusion-persisted", persisted_before and persisted_after,
                      (persisted_before, persisted_after, fid, eid)),
                check("exclusion-restored-after-reload", not after and await after_box.count() == 1,
                      (await after_box.count(), after))]

    @S("E03-finding-linkage", "evidence", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_e03")
        links = await b.page.eval_on_selector_all(
            "#evidence .ev[data-fid]", "els => els.map(e => e.dataset.fid || '')")
        entries = await b.page.eval_on_selector_all(
            "#evidence input[data-v1-evidence]",
            "els => els.map(e => ({fid: e.dataset.fid || '', eid: e.dataset.eid || '', text: e.parentElement?.textContent || ''}))")
        findings = await v1_findings(b, sid)
        bundle = await v1_export(b, sid)
        expected_fids = [f["finding_id"] for f in findings]
        expected_entries = [(fid, entry.get("evidence_id"))
                            for fid, group in bundle.get("evidence", {}).items()
                            for entry in group.get("evidence", [])]
        actual_entries = [(entry["fid"], entry["eid"]) for entry in entries]
        linkage = (bool(expected_fids) and sorted(links) == sorted(expected_fids)
                   and len(links) == len(set(links))
                   and all(fid and eid for fid, eid in actual_entries)
                   and len(actual_entries) == len(set(actual_entries))
                   and sorted(actual_entries) == sorted(expected_entries)
                   and all(entry["eid"] in entry["text"] for entry in entries))
        return [check("linkage-matches-findings-and-evidence", linkage,
                      {"links": links, "expected_fids": expected_fids,
                       "entries": entries, "expected_entries": expected_entries})]

    @S("E04-refresh-persists", "evidence", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_e04")
        exclusion = b.page.locator("#evidence input[data-v1-evidence]").first
        if await exclusion.count() == 0 or not await exclusion.is_checked():
            raise RuntimeError("prepared evidence has no included item to persist")
        exclusion_fid = await exclusion.get_attribute("data-fid")
        exclusion_eid = await exclusion.get_attribute("data-eid")
        if not exclusion_fid or not exclusion_eid:
            raise RuntimeError("prepared evidence exclusion target has no durable identity")
        await exclusion.click()
        await b.page.wait_for_function(
            "target => fetch(`/api/v1/export/${target.sid}`).then(r => r.json()).then(bundle => "
            "!!bundle.evidence?.[target.fid]?.excluded?.some(e => e.evidence_id === target.eid))",
            arg={"sid": sid, "fid": exclusion_fid, "eid": exclusion_eid}, timeout=30000)
        bundle_before = await v1_export(b, sid)
        rendered_before = await rendered_v1_evidence(b)
        expected_before = exported_v1_evidence_rows(bundle_before)
        actual_before = [{key: row[key] for key in ("fid", "eid", "go_id", "origin", "rank", "query", "checked")}
                         for row in rendered_before["entries"]]
        expected_projection_before = [
            {key: row[key] for key in ("fid", "eid", "go_id", "origin", "rank", "query", "checked")}
            for row in expected_before]
        row_order = lambda row: (row["fid"], row["eid"])
        if sorted(actual_before, key=row_order) != sorted(expected_projection_before, key=row_order):
            raise RuntimeError("prepared evidence DOM is not bound to its persisted export")
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        await select_by_value(b, sid)
        bundle_after = await v1_export(b, sid)
        expected_after = exported_v1_evidence_rows(bundle_after)
        await b.page.wait_for_function(
            "expected => document.querySelectorAll('#evidence input[data-v1-evidence]').length === expected",
            arg=len(expected_after), timeout=30000)
        rendered_after = await rendered_v1_evidence(b)
        actual_after = [{key: row[key] for key in ("fid", "eid", "go_id", "origin", "rank", "query", "checked")}
                        for row in rendered_after["entries"]]
        groups_expected = [{"fid": fid, "query": group.get("query", "")}
                           for fid, group in bundle_after.get("evidence", {}).items()]
        options = await b.page.eval_on_selector_all("#specimens option", "els => els.length")
        return [check("reloaded", options > 5, options),
                check("evidence-persisted", expected_after and
                      bundle_before.get("specimen_id") == sid
                      and bundle_after.get("specimen_id") == sid
                      and canonical(bundle_before.get("evidence")) == canonical(bundle_after.get("evidence")),
                      {"before": expected_before, "after": expected_after, "sid": sid}),
                check("evidence-render-restored", sorted(actual_after, key=row_order) == sorted([
                    {key: row[key] for key in ("fid", "eid", "go_id", "origin", "rank", "query", "checked")}
                    for row in expected_after], key=row_order)
                      and rendered_after["groups"] == groups_expected
                      and all(row["name"] in next(actual["text"] for actual in rendered_after["entries"]
                                                    if actual["fid"] == row["fid"] and actual["eid"] == row["eid"])
                              for row in expected_after),
                      {"expected": expected_after, "actual": rendered_after})]

    # ---- synthesis (10) ----
    @S("Y01-normal-note", "synthesis", 2.5)
    async def _(b):
        await b.goto_app()
        await prepared_v1(b, "txl_y01")
        job, note = await synth_done(b)
        return [check("job-done", "done" in job, job[:100]),
                check("note-rendered", len(note) > 100, len(note))]

    @S("Y02-server-down-clean", "synthesis", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_y02")
        pre_findings = await b.findings_count()
        pre_evidence = await b.page.inner_html("#evidence")
        confirmed = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/findings/${id}`)).json()).findings
                .filter(f => f.review_state === 'confirmed' || f.review_state === 'corrected').length""", sid)
        if pre_findings <= 0 or confirmed <= 0 or "GO:" not in pre_evidence:
            raise RuntimeError(f"synthesis-down precondition not prepared: {(pre_findings, confirmed, pre_evidence[:80])}")
        stopped = False
        down_seen = False
        try:
            stop = await b.page.evaluate(
                """async () => { const r = await fetch('/api/server/stop', {method: 'POST'});
                    return {status: r.status, body: await r.json()}; }""")
            await b.page.wait_for_function(
                "() => (document.querySelector('#server-label').textContent || '').includes('down')",
                timeout=30000)
            down_seen = True
            stopped = True
            await b.page.evaluate("() => { S.v1JobIds = []; }")
            await b.page.click("#synthesize")
            await b.page.wait_for_function(
                "() => (S.v1JobIds || []).length > 0 || /v1 failed|failed:/.test(document.querySelector('#job').textContent || '')",
                timeout=30000)
            submitted = await b.page.evaluate("() => S.v1JobIds || []")
            if not submitted:
                raise RuntimeError("server-down synthesis did not submit a v1 job")
            await b.page.wait_for_function(
                "() => /failed:|v1 failed/.test(document.querySelector('#job').textContent || '')",
                timeout=240000)
            failed_job = await b.job_text()
            actual_failure = "synthesis transport failed" in failed_job.lower()
            await b.page.evaluate(
                """async () => { const r = await fetch('/api/server/start', {method: 'POST'});
                    if (!r.ok) throw new Error(`server restart ${r.status}`); }""")
            await b.page.wait_for_function(
                "() => (document.querySelector('#server-label').textContent || '').includes('up')",
                timeout=120000)
            stopped = False
            retry_job, retry_note = await synth_done(b)
            return [check("prepared-confirmed", pre_findings > 0 and confirmed > 0 and "GO:" in pre_evidence,
                          (sid, pre_findings, confirmed)),
                    check("server-stopped", stop["status"] == 200 and down_seen,
                          (stop, failed_job[:100])),
                    check("synthesis-transport-failed", actual_failure and "failed" in failed_job.lower(),
                          failed_job[:160]),
                    check("recovered-after-restart", "done (100%)" in retry_job and len(retry_note) > 100,
                          (retry_job, len(retry_note)))]
        finally:
            if stopped:
                try:
                    await b.page.evaluate("() => fetch('/api/server/start', {method: 'POST'})")
                    await b.page.wait_for_function(
                        "() => (document.querySelector('#server-label').textContent || '').includes('up')",
                        timeout=120000)
                except Exception:  # noqa: BLE001 - runner teardown records any residual outage
                    pass

    @S("Y03-repeat-works", "synthesis", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_y03")
        job1, note1 = await synth_done(b)
        ids_after_first = await b.page.evaluate("() => [...(S.v1JobIds || [])]")
        job2, note2 = await synth_done(b)
        ids = await b.page.evaluate("() => [...(S.v1JobIds || [])]")
        states = await b.page.evaluate(
            """async ids => await Promise.all(ids.map(async id => await (await fetch(`/api/jobs/${id}`)).json()))""", ids)
        packets = [job.get("result", {}).get("packet") for job in states]
        repeated = (len(ids_after_first) == 1 and len(ids) == 2 and ids[0] != ids[1]
                    and all(job.get("state") == "done" for job in states)
                    and packets[0] is not None and packets[0] == packets[1])
        return [check("first-synthesis-done", "done" in job1 and len(note1) > 100,
                      (job1, len(note1))),
                check("second-synthesis-done", "done" in job2 and len(note2) > 100,
                      (job2, len(note2))),
                check("real-repeat-with-stable-packet", repeated,
                      {"sid": sid, "ids": ids, "states": states})]

    @S("Y04-double-click", "synthesis", 1)
    async def _(b):
        await prepared_v1(b, "txl_y04")
        await b.page.evaluate("() => { S.v1JobIds = []; }")
        await b.page.click("#synthesize")
        await b.page.click("#synthesize")
        await b.page.wait_for_function(
            r"() => /done \(100%\)|failed:|cancelled:/.test(document.querySelector('#job').textContent || '')",
            timeout=240000)
        job = await b.job_text()
        note = await b.note_text()
        ids = await b.page.evaluate("() => S.v1JobIds || []")
        return [check("double-click-settled", "done (100%)" in job and len(note) > 100, (job, len(note))),
                check("identical-job-reused", len(ids) == 2 and len(set(ids)) == 1, ids)]

    @S("Y05-refresh-during-job", "synthesis", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_y05")
        await b.page.evaluate("() => { S.v1JobIds = []; }")
        await b.page.click("#synthesize")
        await b.page.wait_for_function(
            "() => (S.v1JobIds || []).length > 0",
            timeout=30000)
        job_ids = await b.page.evaluate("() => S.v1JobIds || []")
        if not job_ids:
            raise RuntimeError("synthesis did not submit before refresh")
        job_id = job_ids[-1]
        job_before_refresh = await b.page.evaluate(
            """async id => await (await fetch(`/api/jobs/${id}`)).json()""", job_id)
        nonterminal_before_refresh = job_before_refresh.get("state") in ("queued", "running")
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_selector("#synthesize")
        await select_by_value(b, sid)
        await b.page.wait_for_function(
            r"""id => /done \(100%\)/.test(document.querySelector('#job').textContent || '')
                && (document.querySelector('#note').textContent || '').length > 100
                && document.querySelector('#job').dataset.jobId === id""",
            arg=job_id, timeout=240000)
        state = await b.page.evaluate(
            """async id => await (await fetch(`/api/jobs/${id}`)).json()""", job_id)
        ui = await b.page.evaluate(
            """() => ({selected: document.querySelector('#specimens').value,
                job: document.querySelector('#job').textContent || '',
                job_id: document.querySelector('#job').dataset.jobId || '',
                note: document.querySelector('#note').textContent || '',
                state_note: S.note || null})""")
        ui_recovered = (ui["selected"] == sid and ui["job_id"] == job_id
                        and "done (100%)" in ui["job"] and len(ui["note"]) > 100
                        and ui["state_note"] == state.get("result", {}).get("note"))
        return [check("job-nonterminal-before-refresh", nonterminal_before_refresh,
                      job_before_refresh),
                check("recovered", await b.page.input_value("#specimens") == sid, sid),
                check("browser-ui-recovered", ui_recovered, {"ui": ui, "job": state}),
                check("backend-job-settled", state.get("state") == "done"
                      and state.get("result", {}).get("note") == ui["note"], state)]

    @S("Y06-fail-then-retry", "synthesis", 1)
    async def _(b):
        await prepared_v1(b, "txl_y06")
        await b.page.evaluate("() => fetch('/api/server/stop', {method: 'POST'})")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('down')",
            timeout=30000)
        failed_job, _ = await synth_done(b)
        failed = "failed" in failed_job.lower() and "synthesis" in failed_job.lower()
        await b.page.evaluate("() => fetch('/api/server/start', {method: 'POST'})")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('up')",
            timeout=120000)
        retry_job, retry_note = await synth_done(b)
        return [check("failure-observed", failed, failed_job[:120]),
                check("retry-ok", "done (100%)" in retry_job and len(retry_note) > 100,
                      (retry_job, len(retry_note)))]

    @S("Y07-stale-draft-invalidated", "synthesis", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_y07")
        job0, _ = await synth_done(b)
        first_packet = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/export/${id}`)).json()).synthesis.packet""",
            await b.page.input_value("#specimens"))
        btns = b.page.locator("#findings button[data-act='confirm']")
        if await btns.count():
            target = btns.first
            fid = await target.get_attribute("data-fid")
            if not fid:
                raise RuntimeError("finding action has no finding id")
            # Change an existing reviewed finding in the append-only review log.
            # A correction keeps the finding identity in the packet while
            # changing its reviewed qualifier, so the second synthesis must
            # validate a genuinely new upstream packet rather than relying on
            # a model response for a sparse F2.. sequence.
            correction = await b.page.evaluate(
                """async ({sid, fid}) => {
                    const r = await fetch('/api/v1/reviews', {
                      method: 'POST', headers: {'Content-Type': 'application/json'},
                      body: JSON.stringify({specimen_id: sid, finding_id: fid,
                        action: 'correct', changes: {qualifier: 'uncertain'},
                        reviewer: 'workstation', reason: 'stale-draft regression'})
                    });
                    return {status: r.status, body: await r.json()};
                }""", {"sid": sid, "fid": fid})
            if correction["status"] != 200:
                raise RuntimeError(f"finding correction failed: {correction}")
            await b.page.wait_for_function(
                "async ({sid, fid}) => { const r = await fetch(`/api/v1/findings/${sid}`); "
                "if (!r.ok) return false; const body = await r.json(); "
                "const f = body.findings.find(item => item.finding_id === fid); "
                "return f && f.review_state === 'corrected' && f.qualifier === 'uncertain'; }",
                arg={"sid": sid, "fid": fid}, timeout=30000)
        else:
            raise RuntimeError("prepared fixture has no finding to invalidate")
        job1, _ = await synth_done(b)
        second_packet = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/export/${id}`)).json()).synthesis.packet""",
            await b.page.input_value("#specimens"))
        first_finding = next((f for f in first_packet.get("findings", [])
                              if f.get("finding_id") == fid), {})
        second_finding = next((f for f in second_packet.get("findings", [])
                               if f.get("finding_id") == fid), {})
        changed = first_packet != second_packet
        return [check("re-synth-runs", "done" in job1, job1[:80]),
                check("draft-not-blindly-reused",
                      changed and first_finding.get("qualifier") != second_finding.get("qualifier")
                      and second_finding.get("qualifier") == "uncertain",
                      (changed, first_finding.get("qualifier"),
                       second_finding.get("qualifier")))]

    # ---- provenance (6) ----
    @S("P01-claim-click-region", "provenance", 2)
    async def _(b):
        await prepared_v1(b, "txl_p01")
        note = await b.note_text()
        if len(note) < 100 or "[C1]" not in note:
            job, note = await synth_done(b)
        if "[C1]" not in note:
            raise RuntimeError("synthesized note contains no claim anchor")
        hit = b.page.locator("#note .claim-hit[data-claim-id='C1']")
        if await hit.count() != 1:
            raise RuntimeError("claim C1 was not rendered as a hit region")
        expected = await b.page.evaluate(
            """() => {
              const claim = (S.validated?.claims || []).find(c => c.claim_id === 'C1');
              const box = claim && S.boxes.find(item => item.finding_id === claim.finding_ids?.[0]);
              return box?.bbox || null;
            }""")
        if not expected:
            raise RuntimeError("claim C1 has no corresponding finding box")
        before = await b.page.evaluate("() => S.highlight || null")
        await hit.click()
        clicked = await b.page.evaluate("() => S.highlight || null")
        return [check("claim-hit-region", before is None and clicked == expected,
                      {"before": before, "after": clicked, "expected": expected}),
                check("corresponding-highlight", clicked == expected,
                      {"claim": "C1", "highlight": clicked, "expected": expected})]

    @S("P02-origin-rank-shown", "provenance", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_p02")
        html = await b.page.inner_html("#evidence")
        observed = await b.page.evaluate(
            """async id => {
              const bundle = await (await fetch(`/api/v1/export/${id}`)).json();
              const expected = Object.entries(bundle.evidence || {}).flatMap(([fid, group]) =>
                (group.evidence || []).map(e => ({fid, eid: e.evidence_id,
                  origin: e.origin, rank: Number(e.rank)})));
              const shown = [...document.querySelectorAll('#evidence input[data-v1-evidence]')]
                .map(input => ({fid: input.dataset.fid, eid: input.dataset.eid,
                  origin: input.dataset.origin, rank: Number(input.dataset.rank),
                  text: input.closest('label')?.textContent || ''}));
              const key = value => `${value.fid}:${value.eid}:${value.origin}:${value.rank}`;
              const expectedKeys = expected.map(key).sort();
              const shownKeys = shown.map(key).sort();
              return {expected: expectedKeys, shown: shownKeys, rows: shown};
            }""", sid)
        rendered = (observed["rows"] and observed["shown"] == observed["expected"]
                    and all("[auto]" in row["text"]
                            and "origin:" in row["text"]
                            and "rank:" in row["text"]
                            and row["origin"].startswith("auto-")
                            and row["rank"] >= 1 for row in observed["rows"]))
        first = observed["rows"][0] if observed["rows"] else {}
        render_detail = {"equal": observed["shown"] == observed["expected"],
                         "count": len(observed["rows"]),
                         "auto": "[auto]" in first.get("text", ""),
                         "origin_label": "origin:" in first.get("text", ""),
                         "auto_origin": first.get("origin", "").startswith("auto-"),
                         "rank_label": "rank:" in first.get("text", ""),
                         "rank_positive": first.get("rank", 0) >= 1,
                         "origin": first.get("origin"), "rank": first.get("rank")}
        return [check("evidence-visible", len(html) > 50 and "GO:" in html, len(html)),
                check("origin-rank-visible", rendered, render_detail)]

    @S("P03-digests-shown", "provenance", 1.5)
    async def _(b):
        await prepared_v1(b, "txl_p03")
        await b.page.click("#sec-prov h2")
        await b.page.wait_for_function(
            "() => (document.querySelector('#provenance').textContent || '').includes('phase5')",
            timeout=15000)
        expected = independent_sealed_digests()
        api_digests = await b.page.evaluate("async () => (await (await fetch('/api/provenance')).json()).sealed")
        displayed = await b.page.eval_on_selector_all(
            "#provenance .prov-file",
            "els => els.map(e => ({path: e.dataset.path, digest: e.dataset.digest, text: e.textContent || ''}))")
        display_map = {row["path"]: row["digest"] for row in displayed}
        exact = (api_digests == expected and display_map == expected
                 and len(displayed) == len(expected)
                 and all(row["digest"] == expected.get(row["path"])
                         and row["digest"] in row["text"] for row in displayed))
        return [check("provenance-rendered", len(displayed) == len(expected)
                       and set(display_map) == set(expected), displayed),
                check("named-digests-match-sealed-files", exact,
                      {"expected": expected, "api": api_digests, "displayed": displayed})]

    @S("P04-no-stale-after-switch", "provenance", 1)
    async def _(b):
        await prepared_v1(b, "txl_p04")
        old = await b.page.input_value("#specimens")
        new = await import_and_select(b, IMG["r06_switch"], "r06_switch")
        if old == new:
            raise RuntimeError("switch fixture did not produce a distinct specimen")
        label = await b.page.text_content("#specimen-label")
        note = await b.note_text()
        evidence = await b.page.inner_html("#evidence")
        return [check("switched", label.strip() == new, (old, new, label[:60])),
                check("stale-output-cleared", not note.strip() and not evidence.strip(),
                      (len(note), len(evidence)))]

    # ---- review (8) ----
    @S("W01-signoff", "review", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_w01")
        job, note = await synth_done(b)
        if "done (100%)" not in job or len(note) <= 100:
            raise RuntimeError("signoff precondition synthesis did not complete")
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('signed off')",
            timeout=30000)
        txt = await b.page.text_content("#review-out")
        bundle = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/export/${id}`)).json()).signoff""", sid)
        synthesis = await b.page.evaluate(
            """async id => (await (await fetch(`/api/v1/export/${id}`)).json()).synthesis""", sid)
        packet_sha = synthesis.get("packet_sha256")
        note_sha = hashlib.sha256(synthesis["note"].encode()).hexdigest()
        return [check("signed", "signed off" in txt.lower(), txt[:100]),
                check("signoff-persisted", bundle and bundle.get("specimen_id") == sid, bundle),
                check("signoff-binds-exact-synthesis", bool(packet_sha)
                      and bundle.get("packet_sha256") == packet_sha
                      and bundle.get("note_sha256") == note_sha,
                      {"packet_sha": packet_sha,
                       "actual_packet": bundle.get("packet_sha256"),
                       "note_matches": bundle.get("note_sha256") == note_sha,
                       "actual_note": bundle.get("note_sha256"),
                       "expected_note": note_sha,
                       "synthesis_note_len": len(synthesis.get("note", ""))})]

    @S("W02-sign-before-valid", "review", 2)
    async def _(b):
        sid = await import_and_select(b, IMG["w02"], "w02")
        before = await v1_export(b, sid)
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => /sign-off failed/.test(document.querySelector('#review-out').textContent || '')",
            timeout=30000)
        txt = await b.page.text_content("#review-out")
        after = await v1_export(b, sid)
        return [check("blocked-with-msg", "sign-off failed" in txt.lower() and "409" in txt and "nothing synthesized" in txt.lower(),
                      f"{sid}: {txt[:100]}"),
                check("no-signoff-record-created", "signoff" not in after,
                      {"before": before, "after": after})]

    @S("W03-upstream-change-blocks", "review", 2)
    async def _(b):
        await prepared_v1(b, "txl_w03")
        job, note = await synth_done(b)
        if "done (100%)" not in job or len(note) <= 100:
            raise RuntimeError("upstream-change precondition synthesis did not complete")
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('signed off')",
            timeout=30000)
        first = await b.page.text_content("#review-out")
        btns = b.page.locator("#findings button[data-act='reject']")
        if await btns.count() == 0:
            raise RuntimeError("prepared fixture has no finding to change")
        await btns.first.click()
        await b.page.wait_for_timeout(2000)
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('sign-off failed')",
            timeout=30000)
        second = await b.page.text_content("#review-out")
        return [check("first-signed", "signed off" in first.lower(), first[:80]),
                check("stale-blocked", "sign-off failed" in second.lower() and "409" in second, second[:100])]

    @S("W04-repeat-sign", "review", 1)
    async def _(b):
        await prepared_v1(b, "txl_w04")
        job, note = await synth_done(b)
        sid = await b.page.input_value("#specimens")
        signoff_url = f"/api/v1/signoff/{sid}"
        async with b.page.expect_response(
                lambda response: signoff_url in response.url
                and response.request.method == "POST" and response.status == 200,
                timeout=30000) as first_response:
            await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('signed off')",
            timeout=30000)
        first = await b.page.text_content("#review-out")
        first_record = await (await first_response.value).json()
        export_before_repeat = await v1_export(b, sid)
        if not export_before_repeat.get("signoffs"):
            raise RuntimeError("first sign-off was not durably recorded")
        async with b.page.expect_response(
                lambda response: signoff_url in response.url
                and response.request.method == "POST" and response.status == 200,
                timeout=30000) as second_response:
            await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('signed off')",
            timeout=30000)
        second = await b.page.text_content("#review-out")
        second_record = await (await second_response.value).json()
        export_after_repeat = await v1_export(b, sid)
        history = export_after_repeat.get("signoffs", [])
        stable_replay = (first_record.get("signoff_id")
                         and first_record == second_record
                         and export_before_repeat.get("signoff") == export_after_repeat.get("signoff")
                         and len(history) == 1
                         and history[0] == first_record
                         and history[0].get("packet_sha256") == first_record.get("packet_sha256")
                         and history[0].get("note_sha256") == first_record.get("note_sha256"))
        return [check("first-sign-succeeds", "signed off" in first.lower(), first[:80]),
                check("repeat-signoff-idempotent", "signed off" in second.lower()
                      and "failed" not in second.lower() and stable_replay,
                      {"first": first_record, "second": second_record,
                       "before": export_before_repeat.get("signoffs"),
                       "after": history})]

    @S("W05-provisional-marking", "review", 1)
    async def _(b):
        await prepared_v1(b, "txl_w05")
        job, note = await synth_done(b)
        cls = await b.page.get_attribute("#note", "class")
        return [check("note-provisional", "provisional" in (cls or "") and len(note) > 100,
                      (cls, len(note), job))]

    # ---- async (6) ----
    @S("A01-double-approve-race", "async", 2)
    async def _(b):
        await review_fixture(b, "txl_async")
        before = await v1_export(b, await b.page.input_value("#specimens"))
        sid = before.get("specimen_id")
        btn = b.page.locator("#findings button[data-act='confirm']").first
        if await btn.count() == 0:
            raise RuntimeError("review fixture has no confirm action")
        fid = await btn.get_attribute("data-fid")
        before_target_reviews = [r for r in before.get("reviews", []) if r.get("finding_id") == fid]
        if before_target_reviews:
            raise RuntimeError("double-approve fixture already has a target review")
        await btn.dblclick()
        await b.page.wait_for_timeout(2500)
        await b.page.wait_for_function(
            "() => (document.querySelector('#evidence').textContent || '').includes('GO:')",
            timeout=30000)
        row = b.page.locator(f"#findings .ev[data-fid='{fid}']")
        state = await row.text_content() if await row.count() else ""
        after = await v1_export(b, sid)
        target_reviews = [r for r in after.get("reviews", []) if r.get("finding_id") == fid]
        target_effects = [f for f in await v1_findings(b, sid) if f.get("finding_id") == fid]
        durable_once = (len(after.get("reviews", [])) == len(before.get("reviews", [])) + 1
                        and len(target_reviews) == 1
                        and target_reviews[0].get("action") == "confirm"
                        and target_reviews[0].get("review_id")
                        and len(target_effects) == 1
                        and target_effects[0].get("review_state") == "confirmed")
        return [check("double-approve-settled", await row.count() == 1 and "confirmed" in state.lower(), state[:120]),
                check("finding-still-visible", await b.findings_count() > 0, await b.findings_count()),
                check("single-durable-review-effect", durable_once,
                      {"sid": sid, "fid": fid, "before": before_target_reviews,
                       "after": target_reviews, "effective": target_effects})]

    @S("A02-navigate-mid-analysis", "async", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_a02")
        await b.page.evaluate("() => { S.analysisJobIds = []; }")
        await b.page.click("#analyze")
        await b.page.wait_for_function(
            "() => (S.analysisJobIds || []).length > 0", timeout=30000)
        job_ids = await b.page.evaluate("() => S.analysisJobIds || []")
        if not job_ids:
            raise RuntimeError("analysis did not submit before navigation")
        job_before_navigation = await b.page.evaluate(
            """async id => await (await fetch(`/api/jobs/${id}`)).json()""", job_ids[-1])
        nonterminal_before_navigation = job_before_navigation.get("state") in ("queued", "running")
        options = await b.page.evaluate(
            "(excluded) => [...document.querySelectorAll('#specimens option')].map(o => o.value).filter(v => v !== excluded && v.startsWith('txl-'))",
            sid)
        target = options[0] if options else None
        if not target:
            raise RuntimeError("no alternate specimen available for analysis navigation")
        await select_by_value(b, target)
        await b.page.wait_for_function(
            "() => !(document.querySelector('#findings').textContent || '').includes('loading findings')",
            timeout=30000)
        target_before = await selection_snapshot(b)
        target_expected = await b.page.evaluate(
            """async id => await (await fetch(`/api/specimens/${id}/overlays`)).json()""", target)
        errs = [e for e in b.ctx.console_errors if "uncaught" in e.lower()]
        state = await b.page.evaluate(
            """async id => { for (let i = 0; i < 240; i++) {
                const r = await fetch(`/api/jobs/${id}`); const j = await r.json();
                if (["done", "failed", "cancelled"].includes(j.state)) return j.state;
                await new Promise(resolve => setTimeout(resolve, 1000));
            } return "timeout"; }""", job_ids[-1])
        await b.page.wait_for_timeout(1000)
        target_after = await selection_snapshot(b)
        old_job_id = str(job_ids[-1])
        uncontaminated = (target_before == target_after and target_after["selected"] == target
                          and json.loads(target_after["counts"] or "{}") == target_expected.get("counts", {})
                          and target_before["note"] == "" and target_before["evidence"] == ""
                          and old_job_id not in target_after["job"]
                          and old_job_id not in target_after["note"]
                          and "analyze: done" not in target_after["st_job"])
        return [check("navigated", await b.page.input_value("#specimens") == target, target),
                check("job-nonterminal-before-navigation", nonterminal_before_navigation,
                      job_before_navigation),
                check("no-uncached-crash", not errs, errs[:1]),
                check("job-settled", state in ("done", "failed", "cancelled"), state),
                check("new-selection-uncontaminated", uncontaminated,
                      {"before": target_before, "after": target_after, "old_job": old_job_id})]

    @S("A03-rapid-mode-switch", "async", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_a03")
        if await b.page.locator("#mode-manual").count() == 0 or await b.page.locator("#mode-auto").count() == 0:
            raise RuntimeError("prepared fixture has no review mode controls")
        seen = []
        for _ in range(4):
            await b.page.click("#mode-manual", timeout=3000)
            await b.page.wait_for_function("() => S.reviewMode === 'manual'", timeout=15000)
            seen.append(await b.page.evaluate("() => S.reviewMode"))
            await b.page.click("#mode-auto", timeout=3000)
            await b.page.wait_for_function("() => S.reviewMode === 'auto'", timeout=15000)
            seen.append(await b.page.evaluate("() => S.reviewMode"))
        mode = await b.page.evaluate("() => S.reviewMode")
        findings = await b.findings_count()
        ids = [f["finding_id"] for f in await v1_findings(b, sid)]
        dom_ids = await b.page.eval_on_selector_all(
            "#findings .ev[data-fid]", "els => els.map(e => e.dataset.fid)")
        return [check("stable", seen == ["manual", "auto"] * 4 and mode == "auto",
                      {"seen": seen, "mode": mode}),
                check("findings-intact", findings > 0 and dom_ids == ids,
                      {"findings": findings, "ids": ids, "dom_ids": dom_ids})]

    # ---- recovery (6) ----
    @S("C01-corrupt-then-valid", "recovery", 1.5)
    async def _(b):
        from bench import import_file
        before_reject = await catalog_snapshot(b)
        await import_file(b.page, IMG["corrupt"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.length > 0", timeout=30000)
        n0 = await b.page.text_content("#import-note")
        after_reject = await catalog_snapshot(b)
        await import_file(b.page, IMG["valid_c01"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.includes('imported')", timeout=30000)
        sid = await import_note_id(b)
        after_valid = await catalog_snapshot(b)
        return [check("failed-then-ok", "fail" in n0.lower() or "reject" in n0.lower(), n0[:60]),
                check("rejected-record-absent", before_reject == after_reject,
                      (len(before_reject), len(after_reject))),
                check("valid-recovered", bool(sid) and sid in {item["id"] for item in after_valid},
                      (sid, len(after_valid)))]

    @S("C02-server-toggle", "recovery", 2)
    async def _(b):
        await b.page.evaluate("() => fetch('/api/server/start', {method: 'POST'})")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('up')",
            timeout=120000)
        await b.page.click("#server-label")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('down')",
            timeout=30000)
        txt = await b.page.text_content("#server-label")
        await b.page.click("#server-label")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('up')",
            timeout=120000)
        final = await b.page.text_content("#server-label")
        return [check("toggle-down", "down" in txt.lower(), txt[:60]),
                check("toggle-up", "up" in final.lower(), final[:60])]

    @S("C03-backend-restart", "recovery", 1.5)
    async def _(b):
        before = await catalog_snapshot(b)
        await b.page.evaluate(
            """async () => { const r = await fetch('/api/server/stop', {method: 'POST'});
                if (!r.ok) throw new Error(`server stop ${r.status}`); }""")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('down')",
            timeout=30000)
        restart = await b.restart_backend()
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        await b.page.evaluate(
            """async () => { const r = await fetch('/api/server/start', {method: 'POST'});
                if (!r.ok) throw new Error(`server start ${r.status}`); }""")
        await b.page.wait_for_function(
            "() => (document.querySelector('#server-label').textContent || '').includes('up')",
            timeout=120000)
        status = await b.page.evaluate("() => fetch('/api/health').then(r => r.status)")
        after = await catalog_snapshot(b)
        return [check("backend-restarted", restart.get("ready") and restart.get("old_exit_code") is not None
                       and restart.get("new_pid"), restart),
                check("state-persists", before == after and len(after) > 5,
                      (before[:3], after[:3], len(before), len(after))),
                check("recovered", status == 200 and "up" in (await b.page.text_content("#server-label")).lower(), status)]

    @S("C04-invalid-id-handled", "recovery", 1)
    async def _(b):
        code = await b.page.evaluate(
            "() => fetch('/api/specimens/nope/image').then(r => r.status).catch(e => 'err')")
        return [check("404-shown", code == 404, code)]

    # ---- edge (4) ----
    @S("G01-huge-rejected", "edge", 1)
    async def _(b):
        from bench import import_file
        before = await catalog_snapshot(b)
        await import_file(b.page, IMG["huge"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.length > 0", timeout=60000)
        note = await b.page.text_content("#import-note")
        low = note.lower()
        after = await catalog_snapshot(b)
        return [check("rejected", "fail" in low or "reject" in low or "large" in low, note[:80]),
                check("no-record-created", before == after, (len(before), len(after)))]

    @S("G02-keyboard-nav", "edge", 1)
    async def _(b):
        before = await b.page.input_value("#specimens")
        await b.page.keyboard.press(".")
        await b.page.wait_for_timeout(800)
        after = await b.page.input_value("#specimens")
        label = await b.page.text_content("#specimen-label")
        return [check("navigated", before != after and after == label.strip(),
                      (before, after, label[:30]))]

    @S("G03-dialog-opens", "edge", 0.5)
    async def _(b):
        try:
            async with b.page.expect_file_chooser(timeout=8000) as fc:
                await b.page.click("#import-btn", timeout=8000)
            ch = await fc.value
            await ch.set_files(str(IMG["valid_g03"]))
            await b.page.wait_for_function(
                "() => (document.querySelector('#import-note').textContent || '').length > 0",
                timeout=30000)
            sid = await import_note_id(b)
            catalog = await catalog_snapshot(b)
            opened = bool(sid) and sid in {item["id"] for item in catalog}
        except Exception:  # noqa: BLE001 - native dialogs are environment-sensitive
            opened = False
        return [check("dialog-opens", opened, f"opened={opened}")]

    @S("G04-overlay-toggle", "edge", 0.5)
    async def _(b):
        options = await b.page.eval_on_selector_all(
            "#specimens option", "els => els.map(e => e.value).filter(v => v.startsWith('txl-'))")
        if not options:
            raise RuntimeError("overlay toggle has no labeled specimen target")
        sid = options[0]
        await select_by_value(b, sid)
        await b.page.wait_for_function(
            "() => S.img && S.boxes && S.boxes.length > 0 && document.querySelector('#canvas').width > 0",
            timeout=30000)
        if not await b.page.evaluate("() => S.showOverlay"):
            await b.page.click("#overlay-toggle")
            await b.page.wait_for_function("() => S.showOverlay === true")
        digest = """() => {
            const c = document.querySelector('#canvas');
            const data = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let hash = 2166136261;
            for (let i = 0; i < data.length; i += 4) {
                hash ^= data[i]; hash = Math.imul(hash, 16777619);
                hash ^= data[i + 1]; hash = Math.imul(hash, 16777619);
                hash ^= data[i + 2]; hash = Math.imul(hash, 16777619);
            }
            return hash >>> 0;
        }"""
        on = await b.page.evaluate("() => ({shown: S.showOverlay, boxes: S.boxes.length})")
        on["digest"] = await b.page.evaluate(digest)
        await b.page.click("#overlay-toggle")
        await b.page.wait_for_function("() => S.showOverlay === false")
        off = await b.page.evaluate("() => ({shown: S.showOverlay, boxes: S.boxes.length})")
        off["digest"] = await b.page.evaluate(digest)
        await b.page.click("#overlay-toggle")
        await b.page.wait_for_function("() => S.showOverlay === true")
        restored = await b.page.evaluate("() => ({shown: S.showOverlay, boxes: S.boxes.length})")
        restored["digest"] = await b.page.evaluate(digest)
        rendered_change = (on["shown"] and not off["shown"] and restored["shown"]
                           and on["boxes"] > 0 and off["boxes"] == on["boxes"]
                           and on["digest"] != off["digest"]
                           and restored["digest"] == on["digest"])
        return [check("overlay-render-toggles", rendered_change,
                      {"sid": sid, "on": on, "off": off, "restored": restored})]

    @S("G05-bad-api-id", "edge", 1)
    async def _(b):
        code = await b.page.evaluate(
            "() => fetch('/api/v1/findings/000000000000').then(r => r.status).catch(e => 'err')")
        return [check("handled", code in (404, 409), code)]

    # ---- stability repeats (2) ----
    @S("T01-happy-path-repeat", "stability", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_t01")
        job1, note1 = await synth_done(b)
        job2, note2 = await synth_done(b)
        ids = await b.page.evaluate("() => [...(S.v1JobIds || [])]")
        states = await b.page.evaluate(
            """async ids => await Promise.all(ids.map(async id => await (await fetch(`/api/jobs/${id}`)).json()))""", ids)
        packets = [job.get("result", {}).get("packet") for job in states]
        return [check("first-repeat-done", "done" in job1 and len(note1) > 100,
                      (job1, len(note1))),
                check("second-repeat-done", "done" in job2 and len(note2) > 100,
                      (job2, len(note2))),
                check("repeat-jobs-are-real", len(ids) == 2 and ids[0] != ids[1]
                      and all(job.get("state") == "done" for job in states)
                      and packets[0] == packets[1],
                      {"sid": sid, "ids": ids, "states": states})]

    @S("T02-bulk-review-repeat", "stability", 0.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_t02")
        n = await b.findings_count()
        evidence = await b.page.eval_on_selector_all("#evidence .ev[data-fid]", "els => els.length")
        before = await v1_export(b, sid)
        repeat = await b.page.evaluate(
            """async id => { const r = await fetch('/api/v1/reviews/bulk', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({specimen_id: id, action: 'confirm', reviewer: 'workstation'})
              }); return {status: r.status, body: await r.json()}; }""", sid)
        after = await v1_export(b, sid)
        return [check("repeat-list", n > 0, n),
                check("repeat-evidence", evidence == n and evidence > 0, (n, evidence)),
                check("bulk-repeat-idempotent", repeat["status"] == 200
                      and repeat["body"].get("confirmed") == []
                      and before.get("reviews") == after.get("reviews"),
                      {"repeat": repeat, "before": before.get("reviews"),
                       "after": after.get("reviews")})]

    @S("T03-refresh-recovery-repeat", "stability", 0.5)
    async def _(b):
        before = await catalog_snapshot(b)
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#specimens option').length > 5")
        after = await catalog_snapshot(b)
        return [check("repeat-reloaded", before == after and len(after) > 5,
                       (before[:3], after[:3], len(before), len(after)))]

    # ---- offline (2) ----
    @S("O01-loopback-only", "offline", 2)
    async def _(b):
        bad = [h for h in b.hosts if h not in ("127.0.0.1", "localhost", "")]
        return [check("no-remote", not bad, bad[:3])]
