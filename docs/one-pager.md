# Moss Inference work sample

Tarık Deveci, 25 November 2026. Code and raw logs: https://github.com/tarikdeveci/moss-inference-work-sample

Samples are small (1 to 5 runs per case), so the numbers are observations, not benchmarks.

## 1. Portal

- **Liked:** clean UI, GitHub sign-up, and a sample search that came back in 1 ms.
- **Didn't like:** in onboarding step 4, "Keep sample data for now" did not move me forward, so I opened /inference by hand. The Inference quickstart says there is no included allowance; the pricing page says $10 at signup.
- **Would improve:** fix that step, say the same thing about credits everywhere, and end onboarding with a working first Inference call.

## 2. Inference API

- **Liked:** the OpenAI SDK worked with just a new base URL and key. Errors name the parameter and the allowed range. `moss/fastlane-1` streamed its first token in 196 ms (median of 5, cache off, no index) against 509 for Claude Haiku. Grounding got 5 of 5 graded questions right, preferred the newer of two conflicting documents, ignored a prompt injection, and admitted what was not in the index.
- **Didn't like:** the reply cache is on by default, so five identical requests at temperature 1.5 gave the same name five times. The only sign I found was `input_tokens: 0`, and my first latency numbers (about 140 ms) were just the cache. Empty input `""` returns 200 with invented text, once about the weather in Chinese; missing input returns a clear 400.
- **Would improve:** send a cache-hit header, skip the cache at high temperature, reject empty input, and show the burst limit of 10 in the rate-limit headers (they show only 120).

## 3. Tasks API

**Tested:** 20 invalid requests, idempotency, limits, files, traces, the Python SDK, and 8 parallel browser tasks (one run each), mostly on books.toscrape.com. All 8 started within 18 ms of creation.

| Task | Result | Time | Cost |
|---|---|---|---|
| Lookup, moss-fast | correct | 32 s | $0.009 |
| Same, moss-balanced | correct | 21 s | $0.059 |
| Same, deepseek-v4.1-flash | correct, 14 steps | 30 s | $0.014 |
| Two-page comparison, moss-fast | correct, opened page 2 | 32 s | $0.009 |
| No answer exists | said "Not found", invented nothing | 57 s | $0.013 |
| $0.01 cost cap | `budget_exceeded` | 7 s | $0.024 |
| 60 s time cap | `time_limit`, null result | 61 s | $0.013 |
| Cancel while running | stopped about 12 s after the request | 34 s | $0.010 |

- **Liked:** all 20 bad requests were rejected correctly, idempotency keys worked, and moss-fast got the lookup right for about one seventh of the moss-balanced cost. Traces include screenshots and a timeline.
- **Where it broke:** the time-capped task reported failed with a null result, but the CSV it wrote was complete (50 categories, 1,000 books, matching the site totals). The $0.01 cap ended at $0.024; the docs say running calls finish and bill. After cancel, the trace shows a new tool step about 10 s later.
- **Didn't like:** `verification.passed: true` only means the output fits the schema. `usage.cost_usd` is model cost only; the total is in `usage.cost.total_usd`.
- **Would improve:** point to files already written when a task hits `time_limit`, stop before a call that would pass the cost cap, and name the check `schema_valid`.
- **My own mistake:** my grader read only page 1 and failed a correct answer. Moss was right; both grades are in the logs.
