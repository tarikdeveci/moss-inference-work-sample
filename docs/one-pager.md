# Moss Inference work sample

Tarık Deveci, 25 November 2026. Code and raw logs: https://github.com/tarikdeveci/moss-inference-work-sample

All run from home in Turkey (`GET /models` took 139 ms), 1 to 7 runs per case: what I saw, not benchmarks.

## 1. Portal

- **Liked:** It looks clean, GitHub sign-up was quick, and the sample search came back in 1 ms.
- **Didn't like:** In onboarding step 4, "Keep sample data for now" did nothing, so I typed the Inference page URL myself. The quickstart says there is no included allowance, but pricing says $10 at signup.
- **Would improve:** Fix that step, tell one story about credits, and end onboarding with a first Inference call.

## 2. Inference API

- **Liked:** I used the OpenAI SDK, changed only the base URL and key, and it worked. Most errors name the parameter and its range. With the cache off, `moss/fastlane-1` gave the first token in about 200 ms (median); Claude Haiku took 509 ms. Grounding got all 5 of my graded questions right. It picked the newer of two conflicting documents, ignored a prompt injection, and said so when the index had no answer.
- **Didn't like:** The reply cache is on by default. Same prompt five times at temperature 1.5, same name five times. My first latency numbers (about 140 ms) were really the cache, and the only hint was `input_tokens: 0`. Empty input `""` returns 200 with made-up text. With an index the first token took 284 ms median (docs: under 200 ms), against 200 without, and 2 of 7 grounded runs took over 1 s.
- **Would improve:** Add a header that says a reply came from cache, skip the cache at high temperature, reject empty input, and name the parameter I sent (`max_output_tokens: 0` errors as `max_tokens`).

## 3. Tasks API

**What I tested:** 20 invalid requests, idempotency, limits, files, traces, the Python SDK, and 8 browser tasks in parallel, 3 times (24 tasks, $0.44). My script graded the answers against books.toscrape.com. Every task started within 17 ms.

| Task (3 runs each) | Result | Median time | Median cost |
|---|---|---|---|
| Lookup, moss-fast | 3/3 correct | 21 s | $0.007 |
| Same, moss-balanced | 3/3 correct | 21 s | $0.060 |
| Same, deepseek-v4.1-flash | 3/3 correct, 6 to 14 steps | 25 s | $0.011 |
| Two-page comparison, moss-fast | 3/3 correct | 33 s | $0.009 |
| No answer exists | 3/3 "Not found", nothing invented | 63 s | $0.014 |
| $0.01 cost cap | 3/3 `budget_exceeded` | 7 s | $0.024 |
| 60 s time cap, long job | 1 finished, 2 `time_limit` | 62 s | $0.011 |
| Cancel while running | 3/3 stopped about 12 s after the request | 35 s | $0.010 |

- **Liked:** Clear validation errors, working idempotency, and moss-fast right 3/3 at 1/8 of moss-balanced cost.
- **Where it broke:** When a task hits `time_limit` the result is null, even if the work is done. One of those runs had already written the full CSV (50 categories, 1,000 books), the other had no CSV. After cancel, 2 of 3 traces still show a new tool step about 10 s later.
- **Didn't like:** My $0.01 cap ended at $0.024 all three times. Documented, but 2.4 times the cap is a lot. `verification.passed: true` only means the output fits the schema. `usage.cost_usd` leaves out the browser cost.
- **Would improve:** On `time_limit`, point to the files already written. Stop before a call that passes the cap. Call the check `schema_valid`.
- **My own mistake:** My first grader only read page 1 and failed a correct answer; both grades are logged.

## 4. What I built with it

I wanted to use Moss on something real, so I built `src/docs_qa.py`: it answers questions about Moss from its own docs (1,455 chunks) with a source link. I wrote 11 questions and looked up the answers myself. With the index it got 9 right (one with a wrong extra clause), without it 2, inventing an `X-Moss-Grounding-Index` header. Both misses said "Not in the Moss docs" instead of guessing. I couldn't see which passages it used or choose how many, so I couldn't tell why. In a direct search one miss ranked 4th, yet one ranked 5th came back right.
