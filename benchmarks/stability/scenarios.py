"""Scenario definitions. Each returns ordered checks sharing the weight."""
from __future__ import annotations

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
        import re
        m = re.search(r"imported\s+([0-9a-f]{12})", txt or "")
        return m.group(1) if m else None

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
        sid = await b.page.evaluate("() => (window.__prepSid || null)")
        if sid:
            # Other scenarios deliberately switch specimens (including the
            # zero-finding fixture).  A cached preparation id is only useful
            # after making it the active selection again and waiting for its
            # findings render to settle.
            if await b.page.input_value("#specimens") != sid:
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
        await b.page.evaluate(f"() => window.__prepSid = '{sid}'")
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
        await b.page.wait_for_function(
            "() => { const t = document.querySelector('#job').textContent || ''; return /done|failed|cancelled|v1 failed/.test(t); }",
            timeout=timeout)
        return await b.job_text(), await b.note_text()

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
        before = await b.specimen_ids()
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        after = await b.specimen_ids()
        return [check("same-list", before == after, (len(before), len(after))),
                check("canvas-present", await b.page.is_visible("#canvas"))]

    @S("S03-dead-backend", "startup", 1)
    async def _(b):
        b.ctx.expect_failed_request("127.0.0.1:8099")
        try:
            await b.page.goto("http://127.0.0.1:8099/", timeout=8000)
            return [check("connection-refused", False, "dead backend unexpectedly accepted navigation")]
        except Exception as e:  # noqa: BLE001 - the refusal itself is the assertion
            message = str(e)
            return [check("connection-refused", "CONNECTION_REFUSED" in message.upper(), message[:80])]

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

    @S("I01-valid-png", "import", 2)
    async def _(b):
        note = await _note(b, IMG["valid"])
        return [check("imported-msg", "imported" in note, note)]

    @S("I02-duplicate-import", "import", 1.5)
    async def _(b):
        n0 = len(await b.specimen_ids())
        note = await _note(b, IMG["valid"])
        n1 = len(await b.specimen_ids())
        sid = await import_note_id(b)
        return [check("stable-list", n1 == n0, (n0, n1, note[:60])),
                check("same-specimen", bool(sid), sid or "missing id")]

    @S("I03-corrupt-rejected", "import", 1.5)
    async def _(b):
        note = await _note(b, IMG["corrupt"])
        return [check("rejected-msg", "fail" in note.lower() or "reject" in note.lower(), note)]

    @S("I04-wrong-extension", "import", 1)
    async def _(b):
        note = await _note(b, IMG["textpng"])
        return [check("rejected-msg", "fail" in note.lower() or "reject" in note.lower(), note)]

    @S("I05-rgba-accepted", "import", 1)
    async def _(b):
        note = await _note(b, IMG["rgba"])
        return [check("accepted", "imported" in note, note)]

    @S("I06-grayscale-accepted", "import", 1)
    async def _(b):
        note = await _note(b, IMG["gray"])
        return [check("accepted", "imported" in note, note)]

    @S("I07-oversized-rejected", "import", 1)
    async def _(b):
        note = await _note(b, IMG["huge"], timeout=60000)
        low = note.lower()
        return [check("rejected-msg", "fail" in low or "reject" in low or "large" in low, note[:80])]

    @S("I08-repeated-imports", "import", 1)
    async def _(b):
        ids = []
        for _ in range(3):
            note = await _note(b, IMG["valid"])
            sid = await import_note_id(b)
            if not sid:
                raise RuntimeError(f"repeated import did not return an id: {note[:80]}")
            ids.append(sid)
        return [check("same-specimen", len(set(ids)) == 1, ids),
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
        await prepared_v1(b, "txl_v03")
        n0 = await b.findings_count()
        st = await analyze_current(b)
        n1 = await b.findings_count()
        return [check("job-done", "analyze: done" in st, st),
                check("initial-findings", n0 > 0, n0),
                check("same-count", n0 == n1, (n0, n1))]

    @S("V04-refresh-persists", "vision", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_v04")
        n0 = await b.findings_count()
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        await select_by_value(b, sid)
        await ensure_findings(b)
        n1 = await b.findings_count()
        return [check("list-back", len(await b.specimen_ids()) > 0),
                check("same-selection", await b.page.input_value("#specimens") == sid, sid),
                check("count-persists", n0 == n1 and n1 > 0, (n0, n1))]

    @S("V05-navigate-mid-run", "vision", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_v05")
        await b.page.evaluate("() => { S.analysisJobIds = []; }")
        await b.page.click("#analyze")
        await b.page.wait_for_function(
            "() => (S.analysisJobIds || []).length > 0",
            timeout=30000)
        job_id = (await b.page.evaluate("() => S.analysisJobIds || []"))[-1]
        options = await b.page.eval_on_selector_all(
            "#specimens option", "function(els, excluded) { return els.map(e => e.value).filter(v => v !== excluded); }", sid)
        target = options[0] if options else None
        if not target:
            raise RuntimeError("no alternate specimen available for mid-run navigation")
        await select_by_value(b, target)
        errs = [e for e in b.ctx.console_errors if "uncaught" in e.lower()]
        state = await b.page.evaluate(
            """async id => { for (let i = 0; i < 240; i++) {
                const r = await fetch(`/api/jobs/${id}`); const j = await r.json();
                if (["done", "failed", "cancelled"].includes(j.state)) return j.state;
                await new Promise(resolve => setTimeout(resolve, 1000));
            } return "timeout"; }""", job_id)
        return [check("navigated", await b.page.input_value("#specimens") == target, target),
                check("no-uncached-crash", not errs, errs[:1]),
                check("analysis-job-settled", state in ("done", "failed", "cancelled"), state)]

    # ---- findings (12) ----
    @S("F01-approve-scrolled", "findings", 2)
    async def _(b):
        await b.goto_app()
        await b.page.set_viewport_size({"width": 1440, "height": 500})
        try:
            await review_fixture(b, "txl_review")
            n0 = await b.findings_count()
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
            return [check("had-scroll", y0 > 50, (y0, y1)),
                    check("confirm-changed-target", "confirmed" in state.lower(), state[:120]),
                    check("scroll-preserved", abs(y1 - y0) < 40, (y0, y1, n0, n1, mode))]
        finally:
            await b.page.set_viewport_size({"width": 1440, "height": 900})

    @S("F02-reject-scrolled", "findings", 1.5)
    async def _(b):
        await review_fixture(b, "txl_reject")
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
        return [check("had-scroll", y0 > 50, (y0, y1)),
                check("reject-changed-target", bool(fid) and fid not in remaining, (fid, remaining)),
                check("scroll-preserved", y1 > 50 and y1 <= y0, (y0, y1))]

    @S("F03-bulk-approve", "findings", 1.5)
    async def _(b):
        await import_and_select(b, IMG["txl_bulk"], "txl_bulk")
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
        return [check("bulk-ran", "auto-approved" in txt, txt[:80]),
                check("none-unreviewed", left == 0, left)]

    @S("F04-mode-switch", "findings", 1.5)
    async def _(b):
        await review_fixture(b, "txl_mode")
        states = []
        for _ in range(3):
            await b.page.click("#mode-manual")
            await b.page.wait_for_function(
                "() => S.reviewMode === 'manual' && document.querySelector('#mode-manual')?.classList.contains('primary')",
                timeout=15000)
            states.append("manual")
            await b.page.click("#mode-auto")
            await b.page.wait_for_function(
                "() => S.reviewMode === 'auto' && document.querySelector('#mode-auto')?.classList.contains('primary')",
                timeout=15000)
            states.append("auto")
        n = await b.findings_count()
        return [check("mode-toggled-and-settled", states == ["manual", "auto"] * 3, states),
                check("list-intact", n > 0, n)]

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
        await review_fixture(b, "txl_double")
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
        return [check("action-settled", "confirmed" in state.lower(), state[:120]),
                check("single-row-remains", n > 0 and await row.count() == 1, n)]

    @S("F07-rows-have-ids", "findings", 1)
    async def _(b):
        await prepared_v1(b, "txl_f07")
        n = await b.page.eval_on_selector_all("#findings .ev[data-fid]", "e=>e.length")
        btns = await b.page.eval_on_selector_all("#findings button[data-act]", "e=>e.length")
        return [check("rows-identified", n > 0, n),
                check("actions-present", btns >= n * 2 and btns > 0, btns)]

    @S("F08-many-usable", "findings", 1)
    async def _(b):
        await prepared_v1(b, "txl_f08")
        n = await b.findings_count()
        side_h = await b.page.evaluate("document.querySelector('#side').clientHeight")
        side_scroll = await b.page.evaluate("document.querySelector('#side').scrollHeight")
        return [check("rows-present", n > 0, n),
                check("scrollable", side_scroll > side_h and side_scroll > 100, (side_scroll, side_h))]

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
        await prepared_v1(b, "txl_r01")
        await b.page.fill("#query", "")
        html = await b.page.inner_html("#evidence")
        typed = await b.page.input_value("#query")
        findings = await b.findings_count()
        return [check("evidence-present", "GO:" in html and "automatic derivation" in html, html[:100]),
                check("findings-grounded", findings > 0 and html.count("data-fid") >= findings, (findings, html.count("data-fid"))),
                check("no-typing-needed", typed == "", typed)]

    @S("R02-multiple-findings", "retrieval", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_r02")
        html = await b.page.inner_html("#evidence")
        fids = await b.page.eval_on_selector_all("#evidence [data-fid]", "e=>e.length")
        findings = await b.findings_count()
        return [check("evidence-rendered", len(html) > 50 and "GO:" in html, len(html)),
                check("finding-links", fids == findings and fids > 0, (sid, findings, fids))]

    @S("R03-rejected-excluded", "retrieval", 1.5)
    async def _(b):
        sid = await prepared_v1(b, "txl_r03")
        btn = b.page.locator("#findings button[data-act='reject']").first
        if await btn.count() == 0:
            raise RuntimeError("prepared fixture has no reject action")
        fid = await btn.get_attribute("data-fid")
        await btn.click()
        await b.page.wait_for_timeout(1500)
        await b.page.evaluate("id => v1RetrieveAll(id)", sid)
        await b.page.wait_for_function(
            "fid => ![...document.querySelectorAll('#evidence [data-fid]')].some(e => e.dataset.fid === fid)", arg=fid,
            timeout=30000)
        evidence_fids = await b.page.eval_on_selector_all("#evidence [data-fid]", "els => els.map(e => e.dataset.fid)")
        return [check("rejected-finding-excluded", fid not in evidence_fids, (fid, evidence_fids))]

    @S("R04-manual-override", "retrieval", 1.5)
    async def _(b):
        await prepared_v1(b, "txl_r04")
        await b.page.fill("#query", "leukocyte")
        await b.page.click("#retrieve")
        await b.page.wait_for_function(
            "() => /GO:|retrieval failed/.test(document.querySelector('#evidence').textContent || '')",
            timeout=30000)
        html = await b.page.inner_html("#evidence")
        query = await b.page.input_value("#query")
        return [check("manual-results", "GO:" in html, html[:100]),
                check("query-retained", query == "leukocyte", query)]

    @S("R05-retry-same", "retrieval", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_r05")
        first = await b.page.evaluate(
            """async id => { const r = await fetch(`/api/v1/retrieve/${id}`, {method: 'POST'});
                return {status: r.status, body: await r.json()}; }""", sid)
        second = await b.page.evaluate(
            """async id => { const r = await fetch(`/api/v1/retrieve/${id}`, {method: 'POST'});
                return {status: r.status, body: await r.json()}; }""", sid)
        return [check("first-retrieve-ok", first["status"] == 200 and first["body"].get("evidence", 0) > 0, first),
                check("retry-retrieve-ok", second["status"] == 200 and second["body"].get("evidence", 0) == first["body"].get("evidence", -1), second)]

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
        return [check("both-retrieves-ok", len(responses) == 2 and all(r["status"] == 200 for r in responses), responses),
                check("same-evidence-count", len({r["body"].get("evidence", -1) for r in responses}) == 1 and responses[0]["body"].get("evidence", 0) > 0,
                      [r["body"].get("evidence") for r in responses])]

    # ---- evidence (6) ----
    @S("E01-query-origin-badges", "evidence", 2)
    async def _(b):
        await prepared_v1(b, "txl_e01")
        html = await b.page.inner_html("#evidence")
        return [check("content-present", len(html) > 50 and "GO:" in html, len(html)),
                check("origin-badge", "[auto]" in html and "automatic derivation" in html, html[:120])]

    @S("E02-exclusion", "evidence", 1.5)
    async def _(b):
        await prepared_v1(b, "txl_e02")
        await b.page.fill("#query", "leukocyte")
        await b.page.click("#retrieve")
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#evidence input[type=checkbox]').length > 0",
            timeout=30000)
        boxes = await b.page.query_selector_all("#evidence input[type=checkbox]")
        if not boxes:
            raise RuntimeError("manual retrieval returned no evidence checkboxes")
        before = await boxes[0].is_checked()
        await boxes[0].click()
        await b.page.wait_for_timeout(500)
        after = await b.page.locator("#evidence input[type=checkbox]").first.is_checked()
        return [check("toggle-ok", before != after, (before, after))]

    @S("E03-finding-linkage", "evidence", 1.5)
    async def _(b):
        await prepared_v1(b, "txl_e03")
        html = await b.page.inner_html("#evidence")
        links = await b.page.eval_on_selector_all("#evidence [data-fid]", "els => els.map(e => e.dataset.fid)")
        return [check("linkage-visible", links and all(fid.startswith("F") for fid in links), links)]

    @S("E04-refresh-persists", "evidence", 1)
    async def _(b):
        sid = await prepared_v1(b, "txl_e04")
        before = await b.page.eval_on_selector_all("#evidence [data-fid]", "els => els.length")
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function("() => document.querySelectorAll('#specimens option').length > 5")
        bundle = await b.page.evaluate(
            """async id => { const r = await fetch(`/api/v1/export/${id}`); return await r.json(); }""", sid)
        after = sum(len(group.get("evidence", [])) for group in bundle.get("evidence", {}).values())
        options = await b.page.eval_on_selector_all("#specimens option", "els => els.length")
        return [check("reloaded", options > 5, options),
                check("evidence-persisted", before > 0 and after > 0, (before, after))]

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
        await prepared_v1(b, "txl_y03")
        job, note = await synth_done(b)
        retried = False
        if "failed: SynthesisError: synthesis transport failed" in job:
            # One recorded retry after re-verifying server health: distinguishes
            # a flapped server process from a systematic synthesis failure.
            await b.page.wait_for_timeout(15000)
            job, note = await synth_done(b)
            retried = True
        return [check("done-again", "done" in job, job[:80]),
                check("note-again", len(note) > 100, len(note)),
                check("no-retry-needed", not retried, f"retried={retried}")]

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
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_selector("#synthesize")
        await select_by_value(b, sid)
        state = await b.page.evaluate(
            """async id => { for (let i = 0; i < 240; i++) {
                const r = await fetch(`/api/jobs/${id}`); const j = await r.json();
                if (["done", "failed", "cancelled"].includes(j.state)) return j.state;
                await new Promise(resolve => setTimeout(resolve, 1000));
            } return "timeout"; }""", job_id)
        return [check("recovered", await b.page.input_value("#specimens") == sid, sid),
                check("backend-job-settled", state == "done", state)]

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
        job0, note0 = await synth_done(b)
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
        job1, note1 = await synth_done(b)
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
                      and second_finding.get("qualifier") == "uncertain" and note0 != note1,
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
        clicked = await b.page.evaluate(
            """() => {
              const el = document.querySelector('#note');
              const r = el.getBoundingClientRect();
              el.dispatchEvent(new MouseEvent('click', {bubbles: true, clientX: r.x + 10, clientY: r.y + 10}));
              return (typeof S !== 'undefined' && S.highlight) ? JSON.stringify(S.highlight) : 'no-highlight';
            }""")
        return [check("click-handled", clicked != "no-highlight", str(clicked)[:100])]

    @S("P02-origin-rank-shown", "provenance", 1.5)
    async def _(b):
        await prepared_v1(b, "txl_p02")
        html = await b.page.inner_html("#evidence")
        return [check("evidence-visible", len(html) > 50 and "GO:" in html, len(html)),
                check("origin-rank-visible", "[auto]" in html and "query:" in html, html[:120])]

    @S("P03-digests-shown", "provenance", 1.5)
    async def _(b):
        await prepared_v1(b, "txl_p03")
        await b.page.click("#sec-prov h2")
        await b.page.wait_for_function(
            "() => (document.querySelector('#provenance').textContent || '').includes('phase5')",
            timeout=15000)
        html = await b.page.inner_html("#provenance")
        import re
        return [check("provenance-rendered", len(html) > 20 and "phase5" in html, html[:100]),
                check("digest-visible", bool(re.search(r"[0-9a-f]{20}", html)), html[:100])]

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
        return [check("signed", "signed off" in txt.lower(), txt[:100]),
                check("signoff-persisted", bundle and bundle.get("specimen_id") == sid, bundle)]

    @S("W02-sign-before-valid", "review", 2)
    async def _(b):
        sid = await import_and_select(b, IMG["w02"], "w02")
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => /sign-off failed/.test(document.querySelector('#review-out').textContent || '')",
            timeout=30000)
        txt = await b.page.text_content("#review-out")
        return [check("blocked-with-msg", "sign-off failed" in txt.lower() and "409" in txt and "nothing synthesized" in txt.lower(),
                      f"{sid}: {txt[:100]}")]

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
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('signed off')",
            timeout=30000)
        first = await b.page.text_content("#review-out")
        await b.page.click("#signoff")
        await b.page.wait_for_function(
            "() => (document.querySelector('#review-out').textContent || '').includes('signed off')",
            timeout=30000)
        second = await b.page.text_content("#review-out")
        return [check("first-sign-succeeds", "signed off" in first.lower(), first[:80]),
                check("repeat-sign-succeeds", "signed off" in second.lower() and "failed" not in second.lower(), second[:80])]

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
        return [check("double-approve-settled", await row.count() == 1 and "confirmed" in state.lower(), state[:120]),
                check("finding-still-visible", await b.findings_count() > 0, await b.findings_count())]

    @S("A02-navigate-mid-analysis", "async", 2)
    async def _(b):
        sid = await prepared_v1(b, "txl_a02")
        await b.page.evaluate("() => { S.analysisJobIds = []; }")
        await b.page.click("#analyze")
        await b.page.wait_for_timeout(2000)
        job_ids = await b.page.evaluate("() => S.analysisJobIds || []")
        if not job_ids:
            raise RuntimeError("analysis did not submit before navigation")
        options = await b.page.evaluate(
            "(excluded) => [...document.querySelectorAll('#specimens option')].map(o => o.value).filter(v => v !== excluded)",
            sid)
        target = options[0] if options else None
        if not target:
            raise RuntimeError("no alternate specimen available for analysis navigation")
        await select_by_value(b, target)
        errs = [e for e in b.ctx.console_errors if "uncaught" in e.lower()]
        state = await b.page.evaluate(
            """async id => { for (let i = 0; i < 240; i++) {
                const r = await fetch(`/api/jobs/${id}`); const j = await r.json();
                if (["done", "failed", "cancelled"].includes(j.state)) return j.state;
                await new Promise(resolve => setTimeout(resolve, 1000));
            } return "timeout"; }""", job_ids[-1])
        return [check("navigated", await b.page.input_value("#specimens") == target, target),
                check("no-uncached-crash", not errs, errs[:1]),
                check("job-settled", state in ("done", "failed", "cancelled"), state)]

    @S("A03-rapid-mode-switch", "async", 2)
    async def _(b):
        await prepared_v1(b, "txl_a03")
        if await b.page.locator("#mode-manual").count() == 0 or await b.page.locator("#mode-auto").count() == 0:
            raise RuntimeError("prepared fixture has no review mode controls")
        for _ in range(4):
            await b.page.click("#mode-manual", timeout=3000)
            await b.page.click("#mode-auto", timeout=3000)
            await b.page.wait_for_timeout(200)
        mode = await b.page.evaluate("() => S.reviewMode")
        findings = await b.findings_count()
        return [check("stable", mode == "auto" and findings > 0, (mode, findings))]

    # ---- recovery (6) ----
    @S("C01-corrupt-then-valid", "recovery", 1.5)
    async def _(b):
        from bench import import_file
        await import_file(b.page, IMG["corrupt"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.length > 0", timeout=30000)
        n0 = await b.page.text_content("#import-note")
        await import_file(b.page, IMG["valid"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.includes('imported')", timeout=30000)
        sid = await import_note_id(b)
        return [check("failed-then-ok", "fail" in n0.lower() or "reject" in n0.lower(), n0[:60]),
                check("valid-recovered", bool(sid), sid or "missing imported id")]

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
        before = await b.specimen_ids()
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
        after = await b.specimen_ids()
        return [check("backend-restarted", restart.get("ready") and restart.get("old_exit_code") is not None
                       and restart.get("new_pid"), restart),
                check("state-persists", before == after and len(after) > 5, (len(before), len(after))),
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
        await import_file(b.page, IMG["huge"])
        await b.page.wait_for_function(
            "() => document.querySelector('#import-note').textContent.length > 0", timeout=60000)
        note = await b.page.text_content("#import-note")
        low = note.lower()
        return [check("rejected", "fail" in low or "reject" in low or "large" in low, note[:80])]

    @S("G02-keyboard-nav", "edge", 1)
    async def _(b):
        before = await b.page.text_content("#specimen-label")
        await b.page.keyboard.press(".")
        await b.page.wait_for_timeout(800)
        after = await b.page.text_content("#specimen-label")
        return [check("navigated", before != after, (before[:30], after[:30]))]

    @S("G03-dialog-opens", "edge", 0.5)
    async def _(b):
        try:
            async with b.page.expect_file_chooser(timeout=8000) as fc:
                await b.page.click("#import-btn", timeout=8000)
            ch = await fc.value
            await ch.set_files(str(IMG["valid"]))
            await b.page.wait_for_function(
                "() => (document.querySelector('#import-note').textContent || '').length > 0",
                timeout=30000)
            opened = True
        except Exception:  # noqa: BLE001 - native dialogs are environment-sensitive
            opened = False
        return [check("dialog-opens", opened, f"opened={opened}")]

    @S("G04-overlay-toggle", "edge", 0.5)
    async def _(b):
        t0 = await b.page.text_content("#overlay-toggle")
        await b.page.click("#overlay-toggle")
        await b.page.wait_for_timeout(500)
        t1 = await b.page.text_content("#overlay-toggle")
        await b.page.click("#overlay-toggle")
        return [check("toggled", t0 != t1, (t0, t1))]

    @S("G05-bad-api-id", "edge", 1)
    async def _(b):
        code = await b.page.evaluate(
            "() => fetch('/api/v1/findings/000000000000').then(r => r.status).catch(e => 'err')")
        return [check("handled", code in (404, 409), code)]

    # ---- stability repeats (2) ----
    @S("T01-happy-path-repeat", "stability", 1)
    async def _(b):
        await prepared_v1(b, "txl_t01")
        job, note = await synth_done(b)
        return [check("repeat-done", "done" in job, job[:80]),
                check("repeat-note", len(note) > 100, len(note))]

    @S("T02-bulk-review-repeat", "stability", 0.5)
    async def _(b):
        await prepared_v1(b, "txl_t02")
        n = await b.findings_count()
        evidence = await b.page.eval_on_selector_all("#evidence [data-fid]", "els => els.length")
        return [check("repeat-list", n > 0, n),
                check("repeat-evidence", evidence == n and evidence > 0, (n, evidence))]

    @S("T03-refresh-recovery-repeat", "stability", 0.5)
    async def _(b):
        before = await b.specimen_ids()
        await b.page.reload(wait_until="networkidle")
        await b.page.wait_for_function(
            "() => document.querySelectorAll('#specimens option').length > 5")
        after = await b.specimen_ids()
        return [check("repeat-reloaded", before == after and len(after) > 5, (len(before), len(after)))]

    # ---- offline (2) ----
    @S("O01-loopback-only", "offline", 2)
    async def _(b):
        bad = [h for h in b.hosts if h not in ("127.0.0.1", "localhost", "")]
        return [check("no-remote", not bad, bad[:3])]
