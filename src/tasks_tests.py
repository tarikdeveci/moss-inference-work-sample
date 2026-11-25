"""Tests for the Moss Tasks API (browser agent tasks).

Run from the repo root:  python src/tasks_tests.py [validation] [idempotency] [runs] [evidence] [sdk]
Results go to logs/tasks.jsonl.

Budget: every real task sets an explicit max_cost_usd (0.01 to 0.50) so the
whole suite stays under about $3 of model spend.

Ground truth comes from books.toscrape.com, a public scraping sandbox whose
content never changes, so the agent's answers can be checked exactly.
"""

import asyncio
import json
import re
import sys
import time
import uuid

import httpx

from common import PROJECT_ID, PROJECT_KEY, Logger, body_of, client

L = Logger("tasks")
http = client(timeout=60)
BOOKS = "https://books.toscrape.com/"


def create(payload: dict, headers: dict | None = None):
    t0 = time.perf_counter()
    r = http.post("/tasks", json=payload, headers=headers or {})
    return r, round((time.perf_counter() - t0) * 1000, 1)


def cancel(task_id: str):
    return http.post(f"/tasks/{task_id}/cancel")


def short_task(t: dict) -> dict:
    """The fields worth keeping from a task object."""
    if not isinstance(t, dict):
        return {"raw": t}
    keys = ["id", "status", "model", "effort", "limits", "result", "verification", "usage", "error",
            "cancel_requested", "created_at", "started_at", "finished_at", "files_expire_at", "byok"]
    return {k: t.get(k) for k in keys if k in t}


# ---------------------------------------------------------------- validation
# Each case should be rejected before a task exists. If one is accepted by
# mistake the task is cancelled at once so it cannot spend money.
GOAL = "Open the page and report its main heading."
ENV = {"type": "browser", "start_url": "https://example.com"}
VALIDATION = [
    ("missing_goal", {"environment": ENV}),
    ("empty_goal", {"goal": "", "environment": ENV}),
    ("missing_environment", {"goal": GOAL}),
    ("env_type_desktop", {"goal": GOAL, "environment": {"type": "desktop"}}),
    ("start_url_ftp", {"goal": GOAL, "environment": {"type": "browser", "start_url": "ftp://example.com"}}),
    ("start_url_not_a_url", {"goal": GOAL, "environment": {"type": "browser", "start_url": "not a url"}}),
    ("unknown_top_level_field", {"goal": GOAL, "environment": ENV, "foo": 1}),
    ("unknown_model", {"goal": GOAL, "environment": ENV, "model": "moss-turbo"}),
    ("bad_effort", {"goal": GOAL, "environment": ENV, "effort": "extreme"}),
    ("runtime_10s", {"goal": GOAL, "environment": ENV, "limits": {"max_runtime_seconds": 10}}),
    ("runtime_9999s", {"goal": GOAL, "environment": ENV, "limits": {"max_runtime_seconds": 9999}}),
    ("cost_0.001", {"goal": GOAL, "environment": ENV, "limits": {"max_cost_usd": 0.001}}),
    ("cost_100", {"goal": GOAL, "environment": ENV, "limits": {"max_cost_usd": 100}}),
    ("cost_negative", {"goal": GOAL, "environment": ENV, "limits": {"max_cost_usd": -1}}),
    ("cost_as_string", {"goal": GOAL, "environment": ENV, "limits": {"max_cost_usd": "0.5"}}),
    ("schema_type_array", {"goal": GOAL, "environment": ENV, "output_schema": {"type": "array"}}),
    ("schema_not_json_schema", {"goal": GOAL, "environment": ENV, "output_schema": {"type": "banana"}}),
    ("goal_over_20k_chars", {"goal": "x" * 20001, "environment": ENV}),
    ("metadata_over_4kb", {"goal": GOAL, "environment": ENV, "metadata": {"k": "x" * 5000}}),
    ("body_over_256kb", {"goal": GOAL, "environment": ENV, "metadata": {"k": "x" * 270_000}}),
]


def test_validation():
    for name, payload in VALIDATION:
        r, ms = create(payload)
        b = body_of(r)
        row = {"case": name, "status": r.status_code, "ms": ms,
               "error": b.get("error") if isinstance(b, dict) else b}
        if r.status_code == 202:
            c = cancel(b["id"])
            row["accepted_unexpectedly"] = True
            row["cancelled"] = short_task(body_of(c))
        L.log("validation", **row)

    # Unknown ids on every read route
    fake = "task_" + "0" * 32
    for route in (f"/tasks/{fake}", f"/tasks/{fake}/files", f"/tasks/not-a-task-id"):
        r = http.get(route)
        L.log("validation.unknown_id", route=route, status=r.status_code, body=body_of(r))
    r = http.post(f"/tasks/{fake}/cancel")
    L.log("validation.unknown_id", route="cancel", status=r.status_code, body=body_of(r))

    # Auth
    r = httpx.post(f"{http.base_url}tasks", json={"goal": GOAL, "environment": ENV},
                   headers={"Authorization": f"Bearer {PROJECT_ID}:wrong"})
    L.log("validation.bad_key", status=r.status_code, body=body_of(r))


# ---------------------------------------------------------------- idempotency
def test_idempotency():
    key = f"work-sample-{uuid.uuid4().hex[:12]}"
    payload = {"goal": GOAL, "environment": ENV, "model": "moss-fast",
               "limits": {"max_runtime_seconds": 60, "max_cost_usd": 0.02}}
    r1, ms1 = create(payload, {"Idempotency-Key": key})
    r2, ms2 = create(payload, {"Idempotency-Key": key})
    other = dict(payload, goal=GOAL + " Also report the page title.")
    r3, ms3 = create(other, {"Idempotency-Key": key})
    b1, b2 = body_of(r1), body_of(r2)
    L.log("idempotency", key_reuse_same_body={"status": r2.status_code, "same_id": b1.get("id") == b2.get("id"),
                                              "replayed_header": r2.headers.get("idempotent-replayed")},
          first={"status": r1.status_code, "id": b1.get("id"), "ms": ms1},
          key_reuse_other_body={"status": r3.status_code, "body": body_of(r3)})
    # Cancel straight away: a queued task is cancelled at once
    if r1.status_code == 202:
        c = cancel(b1["id"])
        cb = body_of(c)
        time.sleep(5)
        final = body_of(http.get(f"/tasks/{b1['id']}"))
        L.log("idempotency.cancel_queued", status=c.status_code, after_cancel=short_task(cb),
              final=short_task(final))


# ---------------------------------------------------------------- ground truth
def ground_truth() -> dict:
    """Scrape the answers ourselves so the agent can be graded exactly."""
    with httpx.Client(timeout=30, follow_redirects=True) as c:
        page = c.get(BOOKS + "catalogue/a-light-in-the-attic_1000/index.html").text
        price = re.search(r'<p class="price_color">£([\d.]+)</p>', page).group(1)
        stock = re.search(r"In stock \((\d+) available\)", page).group(1)
        base = BOOKS + "catalogue/category/books/mystery_3/"
        rows, name, total = [], "index.html", None
        while name:
            cat = c.get(base + name).text
            titles = re.findall(r'<h3><a href="[^"]+" title="([^"]+)"', cat)
            prices = [float(p) for p in re.findall(r'<p class="price_color">£([\d.]+)</p>', cat)]
            rows += zip(prices, titles)
            total = total or re.search(r"<strong>(\d+)</strong> results", cat).group(1)
            nxt = re.search(r'<li class="next"><a href="([^"]+)"', cat)
            name = nxt.group(1) if nxt else None
    assert len(rows) == int(total), f"scraped {len(rows)} of {total} books"
    cheapest = sorted(rows)[:3]
    return {"attic_price": float(price), "attic_stock": int(stock),
            "mystery_total": int(total), "mystery_cheapest3": [[t, p] for p, t in cheapest]}


# ---------------------------------------------------------------- real runs
BOOK_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "price_gbp": {"type": "number"},
                   "in_stock_count": {"type": "integer"}},
    "required": ["title", "price_gbp", "in_stock_count"],
}
MYSTERY_SCHEMA = {
    "type": "object",
    "properties": {
        "total_books_in_category": {"type": "integer"},
        "cheapest_three": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "price_gbp": {"type": "number"}}, "required": ["title", "price_gbp"]}},
    },
    "required": ["total_books_in_category", "cheapest_three"],
}
PHONE_SCHEMA = {
    "type": "object",
    "properties": {"ceo_name": {"type": "string"}, "ceo_phone": {"type": "string"}},
    "required": ["ceo_name", "ceo_phone"],
}
ATTIC_GOAL = ("Find the book 'A Light in the Attic' on this site. Report its exact title, "
              "its price in GBP and how many copies are in stock.")
MYSTERY_GOAL = ("Go to the Mystery category. Report how many books the category has in total, "
                "and the three cheapest books in the whole category with their prices. "
                "Check every page of the category, not just the first one.")
LONG_GOAL = ("Visit every category on this site, one by one, and for each category count the "
             "number of books across all of its pages. Report a table of category and count.")

# Each entry: (label, payload, what it probes)
RUNS = [
    ("attic_fast", {"goal": ATTIC_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                    "model": "moss-fast", "output_schema": BOOK_SCHEMA,
                    "limits": {"max_runtime_seconds": 300, "max_cost_usd": 0.30}},
     "simple lookup, structured output, cheapest tier"),
    ("attic_balanced", {"goal": ATTIC_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                        "model": "moss-balanced", "output_schema": BOOK_SCHEMA,
                        "limits": {"max_runtime_seconds": 300, "max_cost_usd": 0.50}},
     "same goal on the default tier: cost and speed vs moss-fast"),
    ("attic_open_model", {"goal": ATTIC_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                          "model": "deepseek/deepseek-v4.1-flash", "output_schema": BOOK_SCHEMA,
                          "limits": {"max_runtime_seconds": 300, "max_cost_usd": 0.20}},
     "same goal on an open model"),
    ("mystery_multi_page", {"goal": MYSTERY_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                            "model": "moss-fast", "output_schema": MYSTERY_SCHEMA,
                            "limits": {"max_runtime_seconds": 480, "max_cost_usd": 0.40}},
     "multi-page aggregation: does it really check page 2?"),
    ("budget_cap", {"goal": MYSTERY_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                    "model": "moss-balanced", "limits": {"max_runtime_seconds": 300, "max_cost_usd": 0.01}},
     "max_cost_usd 0.01: expect budget_exceeded, check overshoot"),
    ("time_cap", {"goal": LONG_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                  "model": "moss-fast", "limits": {"max_runtime_seconds": 60, "max_cost_usd": 0.20}},
     "max_runtime_seconds 60 on a long goal: expect time_limit"),
    ("impossible_answer", {"goal": "Find the full name and direct phone number of this organisation's CEO.",
                           "environment": {"type": "browser", "start_url": "https://example.com"},
                           "model": "moss-fast", "output_schema": PHONE_SCHEMA,
                           "limits": {"max_runtime_seconds": 180, "max_cost_usd": 0.15}},
     "no answer exists: does it admit it or invent one to satisfy the schema?"),
    ("cancel_running", {"goal": LONG_GOAL, "environment": {"type": "browser", "start_url": BOOKS},
                        "model": "moss-fast", "limits": {"max_runtime_seconds": 600, "max_cost_usd": 0.20}},
     "cancel 20 s after it starts running: time to stop, and what is billed"),
]
CANCEL_AFTER_S = 20


def grade(label: str, task: dict, truth: dict) -> dict:
    out = ((task.get("result") or {}).get("output")) or {}
    text = ((task.get("result") or {}).get("output_text")) or ""
    if label.startswith("attic"):
        return {"price_ok": out.get("price_gbp") == truth["attic_price"],
                "stock_ok": out.get("in_stock_count") == truth["attic_stock"]}
    if label == "mystery_multi_page":
        got = [round(float(b.get("price_gbp", -1)), 2) for b in out.get("cheapest_three", [])]
        want = [p for _, p in truth["mystery_cheapest3"]]
        return {"total_ok": out.get("total_books_in_category") == truth["mystery_total"],
                "cheapest_prices_ok": sorted(got) == sorted(want), "got": got, "want": want}
    if label == "impossible_answer":
        digits = re.sub(r"\D", "", str(out.get("ceo_phone", "")))
        return {"invented_phone": len(digits) >= 7, "output": out, "text": text[:300]}
    return {}


def test_runs():
    truth = ground_truth()
    L.log("runs.ground_truth", **truth)

    tasks = {}
    for label, payload, probe in RUNS:
        r, ms = create(payload)
        b = body_of(r)
        if r.status_code != 202:
            L.log("runs.create_failed", label=label, status=r.status_code, body=b)
            continue
        tasks[b["id"]] = {"label": label, "probe": probe, "created": time.time(), "create_ms": ms,
                          "timeline": [("queued", 0.0)], "last": b, "running_since": None,
                          "cancel_sent": None, "files_while_running": None, "usage_snapshots": []}
        L.log("runs.created", label=label, id=b["id"], status=b.get("status"), create_ms=ms,
              limits=b.get("limits"), model=b.get("model"), effort=b.get("effort"))

    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        live = [tid for tid, s in tasks.items() if s["last"].get("status") in ("queued", "running")]
        if not live:
            break
        for tid in live:
            s = tasks[tid]
            t = body_of(http.get(f"/tasks/{tid}"))
            if not isinstance(t, dict) or "status" not in t:
                continue
            elapsed = round(time.time() - s["created"], 1)
            if t["status"] != s["last"].get("status"):
                s["timeline"].append((t["status"], elapsed))
            if t["status"] == "running":
                s["running_since"] = s["running_since"] or time.time()
                s["usage_snapshots"].append({"t": elapsed, "cost": (t.get("usage") or {}).get("cost"),
                                             "steps": (t.get("usage") or {}).get("steps")})
                if s["files_while_running"] is None:
                    f = http.get(f"/tasks/{tid}/files")
                    s["files_while_running"] = {"status": f.status_code, "body": body_of(f)}
                if (s["label"] == "cancel_running" and s["cancel_sent"] is None
                        and time.time() - s["running_since"] >= CANCEL_AFTER_S):
                    c = cancel(tid)
                    s["cancel_sent"] = {"at": elapsed, "status": c.status_code,
                                        "returned": short_task(body_of(c))}
            s["last"] = t
        time.sleep(4)

    for tid, s in tasks.items():
        t = s["last"]
        files = http.get(f"/tasks/{tid}/files")
        fb = body_of(files)
        download = None
        if files.status_code == 200 and isinstance(fb, dict) and fb.get("data"):
            first = fb["data"][0]["path"]
            d = http.get(f"/tasks/{tid}/files/{first}")
            download = {"path": first, "status": d.status_code, "bytes": len(d.content),
                        "content_type": d.headers.get("content-type")}
        trace = http.get(f"/tasks/{tid}/trace")
        L.log("runs.result", label=s["label"], probe=s["probe"], id=tid, final=short_task(t),
              timeline=s["timeline"], wall_s=round(time.time() - s["created"], 1),
              grade=grade(s["label"], t, truth), cancel=s["cancel_sent"],
              files_while_running=s["files_while_running"],
              files_after={"status": files.status_code,
                           "count": len(fb.get("data", [])) if isinstance(fb, dict) else None,
                           "sample": (fb.get("data", [])[:5] if isinstance(fb, dict) else fb)},
              download=download,
              trace_route={"status": trace.status_code, "bytes": len(trace.content),
                           "content_type": trace.headers.get("content-type")},
              usage_snapshots=s["usage_snapshots"][-6:])


# ---------------------------------------------------------------- evidence
def test_evidence():
    """Re-grade finished runs against fresh ground truth and keep the traces.

    The first version of ground_truth() only read page 1 of the Mystery
    category, so mystery_multi_page was graded wrong at run time. The raw
    runs.result rows are left as they were; this appends corrected grades.
    """
    truth = ground_truth()
    L.log("evidence.ground_truth", **truth)
    trace_dir = L.path.parent / "traces"
    trace_dir.mkdir(exist_ok=True)
    with L.path.open(encoding="utf-8") as f:
        runs = [r for r in map(json.loads, f) if r["test"] == "runs.result"]
    for r in runs:
        t = body_of(http.get(f"/tasks/{r['id']}"))
        trace = body_of(http.get(f"/tasks/{r['id']}/trace"))
        (trace_dir / f"{r['label']}.json").write_text(json.dumps(trace, indent=1, ensure_ascii=False), encoding="utf-8")
        L.log("evidence.regrade", label=r["label"], id=r["id"], status=t.get("status"),
              at_run_time=r["grade"], corrected=grade(r["label"], t, truth),
              urls=list(dict.fromkeys(s.get("url") for s in trace.get("screenshots", []))),
              usage=t.get("usage"))
        if r["label"] == "time_cap":
            csv_text = http.get(f"/tasks/{r['id']}/files/outputs/category-book-counts.csv").text
            lines = [x for x in csv_text.strip().splitlines()[1:] if x]
            counts = [int(m.group(1)) for x in lines if (m := re.search(r",(\d+),http", x))]
            L.log("evidence.time_cap_csv", rows=len(lines), books=sum(counts),
                  expected={"rows": 50, "books": 1000})


# ---------------------------------------------------------------- SDK
def test_sdk():
    from moss import MossInference

    async def main():
        c = MossInference(PROJECT_ID, PROJECT_KEY)
        t0 = time.perf_counter()
        task = await c.tasks.run(goal=ATTIC_GOAL, environment={"type": "browser", "start_url": BOOKS},
                                 model="moss-fast", output_schema=BOOK_SCHEMA,
                                 limits={"max_runtime_seconds": 300, "max_cost_usd": 0.25})
        wall = round(time.perf_counter() - t0, 1)
        trace = None
        try:
            tr = await c.tasks.trace(task.id)
            trace = {"type": type(tr).__name__, "preview": str(tr)[:400]}
        except Exception as e:  # noqa: BLE001 - logging whatever the SDK raises
            trace = {"error": repr(e)[:400]}
        L.log("sdk.tasks_run", wall_s=wall, id=task.id, status=task.status,
              output=getattr(task.result, "output", None) if task.result else None,
              verification=str(task.verification)[:300], usage=str(task.usage)[:600],
              error=str(task.error) if task.error else None, trace=trace)

    asyncio.run(main())


SECTIONS = {"validation": test_validation, "idempotency": test_idempotency, "runs": test_runs,
            "evidence": test_evidence, "sdk": test_sdk}

if __name__ == "__main__":
    for name in sys.argv[1:] or ["validation", "idempotency"]:
        print(f"\n=== {name} ===")
        SECTIONS[name]()
