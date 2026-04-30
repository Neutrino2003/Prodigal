# ProdigalPay Collection Agent

A conversational AI agent that guides customers through a complete payment collection
flow — account lookup → identity verification → balance disclosure → payment collection —
using a **LangGraph**-orchestrated graph with deterministic guardrails around every
LLM decision.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Environment Variables](#2-environment-variables)
3. [Running the Agent](#3-running-the-agent)
4. [Running Tests](#4-running-tests)
5. [Running the Evaluation](#5-running-the-evaluation)
6. [Architecture](#6-architecture)
7. [File Guide](#7-file-guide)
8. [Sample Conversations](#8-sample-conversations)
9. [Token Optimisation — State-Specific Prompts](#9-token-optimisation--state-specific-prompts)
10. [Further Meaningful Improvements](#10-further-meaningful-improvements)

---

## 1. Quick Start

```bash
# 1. Activate the environment
conda activate Ml           # or: pip install -r requirements.txt

# 2. Copy and fill in the env file
cp .env.example .env
# edit .env — see §2 below

# 3. Run interactively
python cli.py
```

---

## 2. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | **Yes** | — | API key for NVIDIA NIM / OpenAI-compatible endpoint |
| `NVIDIA_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | LLM base URL |
| `LLM_MODEL` | No | `meta/llama-3.3-70b-instruct` | Model name |
| `PAYMENT_API_BASE_URL` | **Yes** | — | Base URL for the Prodigal payment backend |
| `MAX_VERIFICATION_ATTEMPTS` | No | `3` | Max identity verification retries |
| `REQUEST_TIMEOUT_SECONDS` | No | `10` | Timeout for account lookup requests |
| `PAYMENT_TIMEOUT_SECONDS` | No | `15` | Timeout for payment processing requests |
| `LOOKUP_NETWORK_RETRIES` | No | `1` | Extra retries on network error for lookup |

**.env.example:**

```dotenv
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.3-70b-instruct
PAYMENT_API_BASE_URL=https://payments.internal.prodigal.ai
MAX_VERIFICATION_ATTEMPTS=3
REQUEST_TIMEOUT_SECONDS=10
PAYMENT_TIMEOUT_SECONDS=15
LOOKUP_NETWORK_RETRIES=1
```

---

## 3. Running the Agent

### CLI (interactive terminal)

```bash
python cli.py
```

Starts a Rich-styled terminal session. Type your messages at the `You ›` prompt.
The session shows live state (verified, session closed, transaction ID) after every turn.
Type `exit` or `quit` to end.

### Python API (evaluator interface)

```python
from payment_agent import Agent

agent = Agent()
agent.next("Hi")                        # {"message": "Hello! Please share your account ID..."}
agent.next("ACC1001")                   # {"message": "Got it. Could you confirm your full name?"}
agent.next("Nithin Jain")              # {"message": "Thanks. DOB, Aadhaar last 4, or pincode?"}
agent.next("DOB is 1990-05-14")        # {"message": "Identity verified. Balance ₹1,250.75..."}
```

`Agent.next()` always returns `{"message": str}` — state is maintained internally between calls.

---

## 4. Running Tests

```bash
# All unit + integration tests (fast — fully mocked, no API calls)
conda activate Ml
python -m pytest tests/ -v
```

Expected result: **27 passed** in ~13 seconds.

| Test file | Coverage |
|---|---|
| `tests/test_basic.py` | Imports, validators (Luhn, DOB, expiry, card type, amount), `ToolContext` defaults, `next()` contract |
| `tests/test_integration.py` | All 4 tools with mocked API — lookup (success/not-found/switch/same-account), verify (success/failure/exhausted/empty-fields/zero-balance), validate_card (success/bad-Luhn/not-verified), process_payment (success/failure/exhausted), `next()` interface |

---

## 5. Running the Evaluation

```bash
# Run all 12 scripted personas against the live API (~12–15 minutes)
python -m eval.runner

# Print the latest result report
python -m eval.metrics
```

Results are saved to `eval/results/eval_<timestamp>.json`.

### Evaluation Personas

| Persona | Account | Scenario |
|---|---|---|
| ACC1001–ACC1004 | All four | Happy-path: verify + pay |
| Happy Path – ACC1001 | ACC1001 | Duplicate control run |
| Failed Verification – ACC1002 | ACC1002 | 3 wrong DOBs → session closed |
| Wrong Account ID | INVALID999 | API 404 → session closed |
| Zero Balance – ACC1003 | ACC1003 | Verified, no payment needed |
| Leap Year DOB – ACC1004 | ACC1004 | DOB 1988-02-29 (valid leap year) |
| Invalid Card – ACC1001 | ACC1001 | Bad Luhn → valid card retry |
| Expired Card – ACC1001 | ACC1001 | Expired 01/2020 → rejected |
| Insufficient Balance – ACC1001 | ACC1001 | Amount > balance → corrected |

### Latest Results

| Metric | Value |
|---|---|
| Total scenarios | 12 |
| Identity verified | 10 / 12 |
| Payment completed | 7 / 12 |
| Sessions correctly closed | 10 / 12 |
| Unit + integration tests | 27 / 27 |

---

## 6. Architecture

```
CLI / Eval Runner
    └─▶ Agent.next(user_input) → {"message": str}
            │
            ├─▶ ToolContext  (mutable business state — verified, balance, card_attempts …)
            │
            └─▶ LangGraph Graph
                    │
                    ├─ input_guard node  ← deterministic pre-validation (dates, expiry)
                    │
                    ├─ assistant node   ← LLM with bound tools (meta/llama-3.3-70b-instruct)
                    │       │
                    │       └─ tool_node ← executes tools + provenance guards
                    │               │
                    │               ├─ lookup_account()   → HTTP POST /api/lookup-account
                    │               ├─ verify_identity()  → domain/verification.py (pure Python)
                    │               ├─ validate_card()    → domain/validators.py  (pure Python)
                    │               └─ process_payment()  → HTTP POST /api/process-payment
                    │
                    └─▶ END (when LLM produces a plain text reply with no tool calls)
```

The LLM handles natural-language understanding and decides *when* to call each tool.
All verification, validation, and state transitions are deterministic Python — the LLM
cannot alter them through prompt manipulation.

**Graph flow per turn:**

```
START → input_guard → assistant ──[tool calls?]──▶ tool_node → assistant
                                └──[no tool calls]──▶ END
```

See [DESIGN.md](./DESIGN.md) for the full architecture deep-dive with Mermaid diagrams.

---

## 7. File Guide

### Root

| File | Purpose |
|---|---|
| `agent.py` *(entry)* | — |
| `cli.py` | Rich-styled interactive terminal |
| `requirements.txt` | Runtime, CLI, test, and eval dependencies |
| `.env.example` | Template for environment variables |
| `README.md` | This file |
| `DESIGN.md` | Architecture decisions, Mermaid diagrams, tradeoffs |
| `conftest.py` | Pytest root configuration (minimal) |
| `test_api.py` | Quick smoke-test for live API connectivity |

### `payment_agent/`

| File | Purpose |
|---|---|
| `__init__.py` | Exports `Agent` |
| `agent.py` | `Agent` class — `next()` interface, response extraction, state management |
| `graph.py` | LangGraph graph: `input_guard`, `assistant`, `tool_node`; all guardrail logic |
| `tools.py` | `ToolContext` dataclass; `make_tools()` factory (4 tools with business logic) |
| `prompts.py` | `SYSTEM_PROMPT` — Aria persona, flow instructions, guardrails |
| `state.py` | `AgentState` TypedDict (`messages` with `add_messages` reducer) |
| `config.py` | `Settings` dataclass loaded from environment via `python-dotenv` |

### `payment_agent/domain/`

| File | Purpose |
|---|---|
| `validators.py` | Luhn check, card type detection, expiry/amount/DOB/CVV validation |
| `verification.py` | `verify_identity()`: factor parsing → structural validation → strict name+factor matching |

### `payment_agent/services/`

| File | Purpose |
|---|---|
| `tools.py` | `lookup_account()` and `process_payment()` HTTP wrappers; Pydantic models; result dataclasses |

### `eval/`

| File | Purpose |
|---|---|
| `runner.py` | Drives scripted personas through `Agent`, saves JSON results |
| `personas.py` | 12 persona definitions with conversation scripts (synced to live API data) |
| `judge.py` | LLM-based conversation scorer with deterministic fallback |
| `metrics.py` | Loads and pretty-prints latest `eval/results/*.json` |

### `tests/`

| File | Purpose |
|---|---|
| `test_basic.py` | Unit tests: validators, `ToolContext`, `Agent` interface |
| `test_integration.py` | Integration tests: all 4 tools with mocked API responses |

---

## 8. Sample Conversations

*(Refer to the `eval/results/` folder for more complete conversation examples and evaluation logs.)*

### A. Successful End-to-End Payment

```
Agent  › Hello! I'm Aria from ProdigalPay. Could you please share your account ID?
User   › ACC1001
Agent  › Thank you. To verify your identity, could you please share your full name?
User   › Nithin Jain
Agent  › Thanks, Nithin. Could you share your date of birth (YYYY-MM-DD),
         Aadhaar last 4 digits, or pincode?
User   › DOB is 1990-05-14
Agent  › Identity verified! Your outstanding balance is ₹1,250.75.
         How much would you like to pay today?
User   › 500
Agent  › Please provide your card number, CVV, expiry date, and the cardholder name.
User   › 4532015112830366 123 12/2027 Nithin Jain
Agent  › Payment successful! Transaction ID: txn_1777564053497_uy3trkr.
         Thank you for paying ₹500.00. Have a wonderful day!
```

### B. Verification Failure (3 Wrong Attempts)

```
Agent  › Could you share your account ID?
User   › ACC1002
Agent  › Please share your full name and a verification factor.
User   › Rajarajeswari Balasubramaniam / DOB 1990-01-01
Agent  › I'm sorry, the details don't match. This was attempt 1 of 3. 2 remaining.
User   › Rajarajeswari Balasubramaniam / DOB 1990-01-02
Agent  › I'm sorry, the details don't match. This was attempt 2 of 3. 1 remaining.
User   › Rajarajeswari Balasubramaniam / DOB 1990-01-03
Agent  › I'm sorry, I was unable to verify your identity after 3 attempts.
         For your security, please visit your nearest branch or call support.
```

### C. Payment Failure — Expired Card

```
Agent  › Identity verified. Balance ₹1,250.75. How much to pay today?
User   › 500
Agent  › Please provide your card details.
User   › 4532015112830366 123 01/2020 Nithin Jain
Agent  › I'm sorry, but that card expired in January 2020.
         Could you provide a valid card?
```

### D. Edge Case — Zero Balance

```
Agent  › Identity verified.
         I can see your account currently has no outstanding balance.
         There's nothing to pay at this time. Thank you for reaching out!
```

---

## 9. Token Optimisation — State-Specific Prompts

### The Current Approach

The agent currently sends the **full `SYSTEM_PROMPT`** (~700 tokens) on every single turn —
greeting, account lookup, verification, payment collection, and close — regardless of which
step the conversation is actually at. This is simple and robust, but wasteful.

At each turn, the LLM reads rules for steps it has already completed and for steps that
cannot logically happen yet (e.g. the card collection instructions are read even on the
greeting turn). In a high-volume deployment, these redundant tokens directly translate to
cost and latency.

### State-Specific Prompts

When the scope widens to production-grade throughput, the system prompt could be **scoped
to the current conversation phase**, derived from the `ToolContext`. By only feeding the LLM the rules for the specific state it is currently in (e.g., greeting vs. card collection), redundant tokens can be eliminated.

### Token Savings Estimate

| Phase | Full prompt | Phase-scoped | Saving |
|---|---|---|---|
| Greeting | ~700 tokens | ~60 tokens | **~91%** |
| Verification | ~700 tokens | ~120 tokens | **~83%** |
| Payment amount | ~700 tokens | ~80 tokens | **~89%** |
| Card collection | ~700 tokens | ~100 tokens | **~86%** |
| Closing | ~700 tokens | ~50 tokens | **~93%** |

In a conversation with 8 turns, the current approach costs ≈ 5,600 prompt tokens in system
messages alone. State-scoped prompts would bring this down to ≈ 410 tokens — a **~93%
reduction in system-prompt token spend** per conversation.

### Why This Isn't Done Yet

This was not implemented because the current requirements are not overly complicated, and passing the full prompt ensures maximum reliability and context retention during the initial build. However, as the complexity of the agent or the use cases increases, transitioning to state-specific prompts will be a highly effective strategy for token optimization and cost reduction.
