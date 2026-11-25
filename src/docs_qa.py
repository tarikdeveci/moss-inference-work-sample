"""Ask the Moss docs: a small CLI that answers questions about Moss using
Moss itself (a Moss index of the docs plus grounded fastlane-1 streaming).

    python src/docs_qa.py build              # fetch docs, chunk them, (re)create the index
    python src/docs_qa.py ask "question"     # stream a grounded answer with source links
    python src/docs_qa.py eval               # graded questions, grounded vs ungrounded
    python src/docs_qa.py rank               # where the answer passage ranks in a direct search
    python src/docs_qa.py drop               # delete the index

Every eval run is logged to logs/demo.jsonl.
"""

import asyncio
import json
import re
import statistics
import sys
import time

import httpx

from common import PROJECT_ID, PROJECT_KEY, Logger, client

DOCS_URL = "https://docs.moss.dev/llms-full.txt"
INDEX = "moss-docs"
MODEL = "moss/fastlane-1"
CHUNK_CHARS = 1200
L = Logger("demo")
http = client(timeout=120)

SYSTEM = ("You answer questions about Moss using only the documentation passages you are given. "
          "Each passage starts with its page title and Source URL. Answer in at most three sentences, "
          "then add a line 'Source: <url>' for the passage you used. If the passages do not contain "
          "the answer, say exactly: Not in the Moss docs.")


# ---------------------------------------------------------------- build
def chunk_docs(text: str) -> list[tuple[str, str]]:
    """Split llms-full.txt into pages ('# Title' + 'Source: url'), then into
    paragraph-aligned chunks. Every chunk carries its page title and URL so
    the model can cite it."""
    pages = re.split(r"\n(?=# [^\n]+\nSource: )", text)
    out = []
    for page in pages:
        m = re.match(r"# ([^\n]+)\nSource: (\S+)", page)
        if not m:
            continue
        title, url = m.group(1).strip(), m.group(2).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", url.split("/docs/")[-1].lower()).strip("-") or "root"
        body = page[m.end():].strip()
        # Moss rejects chunks over 2,000 characters (the embedding model's input
        # limit), so long paragraphs are cut into CHUNK_CHARS pieces first.
        pieces = [p[i:i + CHUNK_CHARS] for p in re.split(r"\n\s*\n", body)
                  for i in range(0, len(p), CHUNK_CHARS)]
        buf, n = "", 0
        for para in pieces:
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                out.append((f"{slug}-{n}", f"{title}\nSource: {url}\n\n{buf.strip()}"))
                buf, n = "", n + 1
            buf += para + "\n\n"
        if buf.strip():
            out.append((f"{slug}-{n}", f"{title}\nSource: {url}\n\n{buf.strip()}"))
    return out


async def _create(chunks):
    from moss import DocumentInfo, MossClient
    mc = MossClient(PROJECT_ID, PROJECT_KEY)
    try:
        await mc.delete_index(INDEX)
    except Exception:  # noqa: BLE001 - fine if it does not exist yet
        pass
    docs = [DocumentInfo(id=i, text=t) for i, t in chunks]
    return await mc.create_index(INDEX, docs, "moss-minilm")


def build():
    t0 = time.perf_counter()
    text = httpx.get(DOCS_URL, timeout=60, follow_redirects=True).text
    chunks = chunk_docs(text)
    fetch_s = round(time.perf_counter() - t0, 1)
    t1 = time.perf_counter()
    res = asyncio.run(_create(chunks))
    L.log("demo.build", source=DOCS_URL, bytes=len(text), chunks=len(chunks),
          pages=len({c[1].split("\n")[1] for c in chunks}), fetch_s=fetch_s,
          index_s=round(time.perf_counter() - t1, 1), result=str(res)[:200])


def drop():
    from moss import MossClient
    print(asyncio.run(MossClient(PROJECT_ID, PROJECT_KEY).delete_index(INDEX)))


# ---------------------------------------------------------------- ask
def ask(question: str, grounded: bool = True, echo: bool = True) -> dict:
    headers = {"X-Moss-Cache": "off"}
    if grounded:
        headers["X-Moss-Indexes"] = INDEX
    payload = {"model": MODEL, "stream": True, "max_output_tokens": 220,
               "input": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]}
    t0 = time.perf_counter()
    ttft, parts, usage, status, retrieve_ms = None, [], None, None, None
    with http.stream("POST", "/responses", json=payload, headers=headers) as r:
        status, retrieve_ms = r.status_code, r.headers.get("x-moss-retrieve-ms")
        if status != 200:
            r.read()
            return {"status": status, "error": r.text[:300]}
        for line in r.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("type") == "response.output_text.delta":
                ttft = ttft or (time.perf_counter() - t0) * 1000
                parts.append(ev["delta"])
                if echo:
                    print(ev["delta"], end="", flush=True)
            elif ev.get("type") == "response.completed":
                usage = (ev.get("response") or {}).get("usage")
    if echo:
        print()
    return {"status": status, "ttft_ms": round(ttft, 1) if ttft else None,
            "total_ms": round((time.perf_counter() - t0) * 1000, 1), "retrieve_ms": retrieve_ms,
            "usage": usage, "text": "".join(parts)}


# ---------------------------------------------------------------- eval
# Answers were checked by hand against the docs on 25 Nov 2026.
# (question, regex that a correct answer must match; None = not in the docs)
EVAL = [
    ("Which request header turns off the reply cache?", r"X-Moss-Cache"),
    ("Which header picks the indexes a response is grounded in?", r"X-Moss-Indexes"),
    ("What range does limits.max_cost_usd accept on a task?", r"0\.01.*50"),
    ("How many days are a task's files kept after it ends?", r"\b7\b|seven"),
    ("What error code do the task files routes return before the task ends?", r"409|task_not_finished"),
    ("How many tasks can a project have queued before it gets a 429?", r"\b20\b"),
    ("Which environment types can a task use?", r"browser"),
    ("What is the default model of a task in the Tasks API?", r"moss-balanced"),
    ("What does a failed task's error code say when it ran out of time?", r"time_limit"),
    ("Who is the CEO of Moss?", None),
    ("What is the phone number of Moss customer support?", None),
]


def is_correct(text: str, pattern: str | None) -> bool:
    if pattern is None:
        return "not in the moss docs" in text.lower()
    return re.search(pattern, text, re.I | re.S) is not None


def evaluate():
    rows = []
    for q, pattern in EVAL:
        row = {"question": q}
        for mode, grounded in (("grounded", True), ("ungrounded", False)):
            r = ask(q, grounded=grounded, echo=False)
            r["correct"] = is_correct(r.get("text", ""), pattern)
            r["cites_source"] = "docs.moss.dev" in r.get("text", "")
            row[mode] = r
        rows.append(row)
        L.log("demo.eval", **row)
    summary = {}
    for mode in ("grounded", "ungrounded"):
        rs = [r[mode] for r in rows]
        tt = [r["ttft_ms"] for r in rs if r.get("ttft_ms")]
        summary[mode] = {"correct": sum(r["correct"] for r in rs), "of": len(rs),
                         "cites_source": sum(r["cites_source"] for r in rs),
                         "ttft_median_ms": statistics.median(tt) if tt else None,
                         "input_tokens_median": statistics.median(
                             [(r.get("usage") or {}).get("input_tokens", 0) for r in rs])}
    L.log("demo.eval_summary", **summary)


# ---------------------------------------------------------------- rank
# The response does not say which passages grounding used, so this queries
# the same index directly (locally, through the SDK) to see where the passage
# holding the answer ranks for each eval question.
async def _ranks(top_k: int):
    from moss import MossClient, QueryOptions
    mc = MossClient(PROJECT_ID, PROJECT_KEY)
    await mc.load_index(INDEX)
    for q, pattern in EVAL:
        if pattern is None:
            continue
        docs = (await mc.query(INDEX, q, QueryOptions(top_k=top_k))).docs
        hits = [i + 1 for i, d in enumerate(docs) if re.search(pattern, d.text, re.I | re.S)]
        L.log("demo.retrieval_rank", question=q, top_k=top_k, answer_regex=pattern,
              first_rank=hits[0] if hits else None, ranks=hits, top3=[d.id for d in docs[:3]])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "build":
        build()
    elif cmd == "ask":
        ask(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        evaluate()
    elif cmd == "rank":
        asyncio.run(_ranks(10))
    elif cmd == "drop":
        drop()
    else:
        sys.exit(__doc__)
