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
```

Every real task sets `max_cost_usd`. The 8 tasks in `runs` cost $0.15 in total. `evidence` re-reads the tasks logged by `runs`, so it only works while those tasks still exist in the project.

## Layout

| Path | Content |
|---|---|
| `src/common.py` | auth header, HTTP client, JSONL logger |
| `src/inference_tests.py` | models, latency, streaming, cache, headers, grounding, errors, OpenAI SDK, tools, concurrency |
| `src/tasks_tests.py` | validation, idempotency, real tasks, evidence, SDK |
| `logs/inference.jsonl`, `logs/tasks.jsonl` | one JSON object per result |
| `logs/traces/` | saved task traces |

Each log row has `ts` (UTC), `test` (section and case) and the fields of that case: status code, time in ms, response body or task object.

## Findings

| Area | Finding | Log rows |
|---|---|---|
| Inference | Reply cache on by default, same text 5 of 5 at temperature 1.5 | `cache.same_prompt_temp_1.5` |
| Inference | Cache off: 4 requests, 4 different outputs | `headers.cache_off` |
| Inference | First token 196 ms median on fastlane, not grounded | `stream.ttft_cache_off` |
| Inference | Empty input returns 200 and invented text | `errors.empty_input`, `headers.empty_input_no_system_prompt` |
| Inference | Grounding: 5 of 5 graded questions right, the sixth checked by reading | `grounding.qa` |
| Tasks | 20 of 20 invalid requests rejected | `validation` |
| Tasks | Idempotency: replay on same body, 422 on another | `idempotency` |
| Tasks | Time-capped task failed with null result, its CSV was complete | `runs.result`, `evidence.time_cap_csv` |
| Tasks | $0.01 cap ended at $0.024 | `runs.result` |
| Tasks | No invented phone number when no answer exists | `runs.result`, `evidence.regrade` |

## Reading the logs

- `latency.non_stream` and `stream.ttft` were measured with the reply cache on and are not model latency. Use the rows that end in `_cache_off`.
- `cache.same_prompt_temp_1.5` for Claude Haiku holds only errors: Moss does not accept temperature on Anthropic models.
- The first grade of `mystery_multi_page` in `runs.result` is wrong because the grader read only page 1 of the category. The corrected grade is in `evidence.regrade`. The original row is unchanged.
- The project ID in one 429 message was replaced with `<project-id>`. No other row was edited.
