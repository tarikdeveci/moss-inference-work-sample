# Moss Inference work sample

Scripts, raw logs and notes from testing the Moss Inference API and the Moss Tasks API on 25 November 2026. The write-up is in [docs/one-pager.md](docs/one-pager.md).

## Setup

Python 3.13 was used.

```bash
python -m venv .venv
.venv/bin/pip install moss==1.26.0 openai python-dotenv httpx   # Windows: .venv\Scripts\pip
```

Create `.env` in the repo root with your own values. It is in `.gitignore`.

```
MOSS_PROJECT_ID=
MOSS_PROJECT_KEY=
```

## Run

```bash
python src/inference_tests.py                 # every section
python src/inference_tests.py stream cache    # chosen sections
python src/tasks_tests.py validation idempotency
python src/tasks_tests.py runs                # 8 real browser tasks, costs money
python src/tasks_tests.py evidence sdk
python src/docs_qa.py build                  # index the Moss docs (about 75 s)
python src/docs_qa.py ask "How do I turn off the reply cache?"
python src/docs_qa.py eval                   # 11 graded questions, with and without the index
python src/docs_qa.py rank                   # rank of the answer passage in a direct search, no inference cost
python src/docs_qa.py drop                   # delete the index
```

Every real task sets `max_cost_usd`. One batch of the 8 tasks in `runs` costs about $0.15; the logs hold 3 batches ($0.44). `evidence` re-reads the tasks logged by `runs` and skips the ones it already graded, so it only works while those tasks still exist in the project. The `docs_qa.py eval` run cost a few cents.

## Layout

| Path | Content |
|---|---|
| `src/common.py` | auth header, HTTP client, JSONL logger |
| `src/inference_tests.py` | models, latency, streaming, cache, headers, grounding, errors, OpenAI SDK, tools, concurrency |
| `src/tasks_tests.py` | validation, idempotency, real tasks, evidence, SDK |
| `src/docs_qa.py` | small app: questions about Moss answered from a Moss index of its own docs |
| `logs/inference.jsonl`, `logs/tasks.jsonl`, `logs/demo.jsonl` | one JSON object per result |
| `logs/traces/` | saved task traces, `<label>_<batch>.json` |

Each log row has `ts` (UTC), `test` (section and case) and the fields of that case: status code, time in ms, response body or task object.

## Findings

| Area | Finding | Log rows |
|---|---|---|
| Inference | Reply cache on by default, same text 5 of 5 at temperature 1.5 | `cache.same_prompt_temp_1.5` |
| Inference | Cache off: 4 requests, 4 different outputs | `headers.cache_off` |
| Inference | First token about 200 ms median on fastlane without an index, 284 ms with one, 2 of 7 grounded runs over 1 s; `GET /models` answered in 139 ms | `stream.ttft_cache_off`, `grounded_ttft.*` |
| Inference | Empty input returns 200 and invented text | `errors.empty_input`, `headers.empty_input_no_system_prompt` |
| Inference | Grounding: 5 of 5 graded questions right, the sixth checked by reading | `grounding.qa` |
| Inference | `max_output_tokens: 0` on /responses is reported as `max_tokens` | `errors.max_output_tokens_0` |
| Tasks | 20 of 20 invalid requests rejected | `validation` |
| Tasks | Idempotency: replay on same body, 422 on another | `idempotency` |
| Tasks | Lookups and the two-page comparison right in 3 of 3 batches on every model | `evidence.regrade` |
| Tasks | 60 s time cap: 1 of 3 finished; 2 failed with a null result, one with a complete CSV already written, one with no CSV | `runs.result`, `evidence.time_cap_csv` |
| Tasks | $0.01 cap ended at $0.024 in 3 of 3 | `runs.result` |
| Tasks | Cancel: stopped about 12 s after the request in 3 of 3; a new tool step after cancel in 2 of 3 | `runs.result`, `logs/traces/cancel_running_*.json` |
| Tasks | No invented phone number when no answer exists, 3 of 3 | `runs.result`, `evidence.regrade` |
| Demo | Moss docs Q&A: 9 of 11 right with the index, 2 of 11 without | `demo.eval`, `demo.eval_summary` |
| Demo | Direct search of the same index: the `time_limit` passage ranks 4th, the default model passage 5th (answered right anyway) | `demo.retrieval_rank` |

## Reading the logs

- `latency.non_stream` and `stream.ttft` were measured with the reply cache on and are not model latency. Use the rows that end in `_cache_off`.
- `cache.same_prompt_temp_1.5` for Claude Haiku holds only errors: Moss does not accept temperature on Anthropic models.
- The first grade of `mystery_multi_page` in `runs.result` is wrong because the grader read only page 1 of the category. The corrected grade is in `evidence.regrade`. The original row is unchanged.
- `wall_s` in `runs.result` is the time until the whole batch finished, not one task. A task's own run time is `final.finished_at` minus `final.started_at`.
- `grounded_ttft.stream` interleaves runs with and without an index on `moss/fastlane-1`, cache off, a new question each run. `grounded_ttft.network_baseline` is `GET /models` from the same connection.
- In `demo.eval`, `correct` is a regex check against an answer I looked up in the docs. For the two questions the docs do not answer, correct means the model said "Not in the Moss docs". `cites_source` only checks that the answer contains `docs.moss.dev`.
- `demo.retrieval_rank` runs the eval questions as a plain SDK query (top 10) and lists the ranks of passages that match the answer regex. For the environment question the regex is just `browser`, which many passages match, so that row says little.
- The project ID in one 429 message was replaced with `<project-id>`. No other row was edited.
