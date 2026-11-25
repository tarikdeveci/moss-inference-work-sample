"""Shared helpers: auth headers, a raw HTTP client and JSONL logging.

Raw HTTP is used on purpose for most tests so the exact status codes,
headers and bodies the API returns end up in the logs unchanged.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

BASE_URL = os.environ.get("MOSS_BASE_URL", "https://inference.moss.dev/v1")
PROJECT_ID = os.environ["MOSS_PROJECT_ID"]
PROJECT_KEY = os.environ["MOSS_PROJECT_KEY"]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def auth_headers(project_id: str = PROJECT_ID, key: str = PROJECT_KEY) -> dict:
    return {"Authorization": f"Bearer {project_id}:{key}"}


def client(timeout: float = 120.0) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers=auth_headers(), timeout=timeout)


class Logger:
    """Appends one JSON object per line to logs/<name>.jsonl."""

    def __init__(self, name: str):
        self.path = LOG_DIR / f"{name}.jsonl"

    def log(self, test: str, **fields):
        row = {"ts": datetime.now(timezone.utc).isoformat(), "test": test, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        short = {k: v for k, v in fields.items() if k not in ("body", "response", "raw")}
        print(f"[{test}] {json.dumps(short, ensure_ascii=False)[:300]}")
        return row


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round((time.perf_counter() - t0) * 1000, 1)


def body_of(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError:
        return resp.text[:2000]


def output_text(resp_json: dict) -> str:
    """Extract text from an OpenAI-style Response object."""
    parts = []
    for item in resp_json.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "".join(parts)
