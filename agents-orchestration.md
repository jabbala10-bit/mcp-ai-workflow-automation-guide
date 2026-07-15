# AI Agents & Agent Orchestration

> A hands-on reference module. Every concept is paired with runnable code and a real-time use case, with recurring framing for BFSI / regulated-industry deployment (auditability, determinism boundaries, human-in-the-loop, cost governance).

**Contents**
1. [What is an Agent?](#1-what-is-an-agent)
2. [Tool Calling](#2-tool-calling)
3. [The ReAct Pattern](#3-the-react-pattern)
4. [Agentic Design Patterns](#4-agentic-design-patterns)
5. [Agent Memory](#5-agent-memory)
6. [Agent Observability](#6-agent-observability)
7. [Lab: Research Agent](#7-lab-research-agent-full-build)
8. [From Single Agent to Orchestration](#8-from-single-agent-to-orchestration)
9. [Production Hardening Checklist](#9-production-hardening-checklist-bfsi-lens)

---

## 1. What is an Agent?

**Definition:** An agent is an LLM placed inside a *loop* where it can choose actions (tools), observe their results, and decide what to do next — until a goal is met or a stop condition fires.

```
Agent = LLM (reasoning) + Tools (actions) + Memory (state) + Decision Loop (control)
```

The key distinction from a plain LLM call: a plain call is **one-shot** (prompt in, text out). An agent is **iterative and stateful** — it can call a tool, look at the result, and call another tool based on what it learned.

### The anatomy, concretely

| Component | What it does | Real example |
|-----------|--------------|--------------|
| **LLM (the brain)** | Decides *what* to do next | "I need the customer's account balance before I can answer" |
| **Tools (the hands)** | Deterministic functions the LLM can invoke | `get_account_balance(account_id)` → hits a real API |
| **Memory (the notebook)** | Carries state across steps | Remembers the account_id extracted three steps ago |
| **Decision loop (the controller)** | Runs the reason→act→observe cycle | `while not done and steps < MAX_STEPS` |

### The decision loop (the thing that makes it an "agent")

```python
def agent_loop(goal: str, tools: dict, max_steps: int = 8):
    """Minimal but honest agent loop."""
    scratchpad = []                       # short-term memory
    for step in range(max_steps):
        decision = llm_decide(goal, scratchpad, tools)   # LLM picks next action

        if decision.type == "final_answer":
            return decision.content        # goal met -> exit

        # Act
        tool = tools[decision.tool_name]
        observation = tool(**decision.args)

        # Observe -> feed back into memory
        scratchpad.append({
            "thought": decision.reasoning,
            "action": decision.tool_name,
            "args": decision.args,
            "observation": observation,
        })
    return "Stopped: reached max steps without a final answer."
```

### When to use an agent vs. when NOT to

This is the single most important architectural decision, and where most teams over-engineer.

**Use an agent when:**
- The number of steps is **unknown ahead of time** (e.g., "keep searching until you find the counterparty's registered address").
- The path **branches based on intermediate results** (e.g., if KYC fails, escalate; if it passes, continue).
- The task genuinely needs **tools** (live data, calculations, writes).

**Do NOT use an agent when:**
- A single prompt or a fixed pipeline (chain) does the job. A deterministic chain is cheaper, faster, and *auditable*. In BFSI, "the LLM decided to loop 6 times" is a compliance liability if a 2-step chain would have sufficed.
- Latency is hard-capped (agents are inherently multi-round-trip).

> **FDE heuristic:** Start with the simplest thing that works — prompt → chain → agent → multi-agent. Only climb the ladder when the rung below genuinely can't hold the requirement. Every rung up adds latency, cost, and non-determinism.

### Real-time use cases

- **Loan pre-qualification assistant (BFSI):** extracts applicant details from a chat, calls a credit-bureau tool, a DTI-calculator tool, and a policy-rules tool, then produces a decision *with* the trace of which rules fired — essential for lending audit (RBI Digital Lending Directions).
- **Incident triage bot (SRE):** reads an alert, queries logs, checks a runbook, and either self-remediates or pages a human.
- **Research agent (this module's lab):** searches, reads, synthesizes, and writes a report.

---

## 2. Tool Calling

Tool calling is the mechanism by which an LLM stops "hallucinating an answer" and instead says *"call this function with these arguments."* You describe your tools as JSON Schema; the model returns a structured request to invoke one; your code executes it and returns the result.

### Defining a tool (JSON schema)

The schema is a contract. The `description` fields are not documentation — they are **prompt engineering**. The model routes on them.

```python
tools = [
    {
        "name": "get_account_balance",
        "description": (
            "Retrieve the current available balance for a customer's account. "
            "Use ONLY when the user asks about their balance or funds available. "
            "Requires a validated 10-12 digit account number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "The customer's account number, digits only.",
                    "pattern": "^[0-9]{10,12}$"
                },
                "currency": {
                    "type": "string",
                    "enum": ["INR", "EUR", "USD"],
                    "description": "ISO currency code for the returned balance."
                }
            },
            "required": ["account_id"]
        }
    }
]
```

### The full round-trip (Anthropic Messages API)

```python
import anthropic
client = anthropic.Anthropic()

def get_account_balance(account_id: str, currency: str = "INR") -> dict:
    # In reality: authenticated call to core banking. Here, a stub.
    return {"account_id": account_id, "available": 154200.50, "currency": currency}

TOOL_REGISTRY = {"get_account_balance": get_account_balance}

messages = [{"role": "user", "content": "What's the balance on account 900012345678?"}]

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=messages,
)

# The model responds with a tool_use block instead of prose
if resp.stop_reason == "tool_use":
    tool_call = next(b for b in resp.content if b.type == "tool_use")
    result = TOOL_REGISTRY[tool_call.name](**tool_call.input)   # YOU execute it

    # Send the result back so the model can compose the final answer
    messages.append({"role": "assistant", "content": resp.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": str(result),
        }]
    })
    final = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024, tools=tools, messages=messages
    )
    print(final.content[0].text)
    # -> "Your account 900012345678 has ₹1,54,200.50 available."
```

### What actually happens (the mental model)

1. You send the user message **plus the tool schemas**.
2. The model decides: answer directly, or request a tool.
3. If it requests a tool, **your code runs it** — the model never executes anything. This is your security boundary.
4. You return the result; the model incorporates it.

> **Security note (critical for regulated deployments):** The LLM only *proposes* a call. Your executor is where you enforce authorization, rate limits, input validation, and PII redaction. Never let the model's proposed `account_id` bypass the same authz checks a human request would face. Treat tool inputs as untrusted.

### Common failure modes & fixes

| Failure | Cause | Fix |
|---------|-------|-----|
| Model invents an argument | Vague schema description | Tighten `description`, add `enum`/`pattern` |
| Model calls the wrong tool | Overlapping descriptions | Make each tool's "use when / don't use when" explicit |
| Model loops calling the same tool | It can't see the result changed nothing | Return a clear error string it can reason about |
| Args pass schema but fail business rules | Schema ≠ business validation | Validate again in the executor; return a correction hint |

### Real-time use cases
- **Payment ops:** `initiate_transfer`, `check_sanctions_list`, `get_fx_rate` — the model orchestrates a compliant transfer flow.
- **Customer support:** `lookup_order`, `issue_refund` (gated behind human approval), `create_ticket`.
- **Data eng:** `run_sql` (read-only role), `profile_table`, `check_freshness`.

---

## 3. The ReAct Pattern

**ReAct = Reasoning + Acting.** Instead of the model silently jumping to a tool, it *interleaves* explicit reasoning ("Thought") with actions. The loop is: **Reason → Act → Observe → Repeat.**

```
Thought:  I need to know if this counterparty is on a sanctions list.
Action:   check_sanctions_list(name="Acme Holdings Ltd")
Observation: {"match": false, "checked_lists": ["OFAC", "EU", "UN"]}
Thought:  Clear. Now I need the FX rate to quote the transfer.
Action:   get_fx_rate(from="EUR", to="INR")
Observation: {"rate": 90.42}
Thought:  I have everything. I can compose the answer.
Final Answer: The transfer is cleared and will convert at ₹90.42/EUR.
```

### Why the explicit "Thought" matters

Two reasons, both practical:
1. **Better decisions.** Forcing the model to reason in text before acting measurably improves tool selection — it "thinks out loud" and catches its own mistakes.
2. **Auditability.** In BFSI you can log every Thought. When a regulator asks *"why did the system escalate this application?"*, you have the reasoning trace, not a black box.

### ReAct with native tool calling (modern approach)

Classic ReAct parsed a text format (`Thought:/Action:`). Modern APIs give you structured tool calls, so ReAct becomes: *let the model emit reasoning text alongside tool_use blocks, and loop.*

```python
def react_agent(question: str, tools, registry, max_steps=6):
    messages = [{"role": "user", "content": question}]
    trace = []

    for step in range(max_steps):
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            tools=tools, messages=messages,
        )

        # Capture reasoning (text blocks) for the audit trail
        thoughts = [b.text for b in resp.content if b.type == "text"]
        trace.append({"step": step, "thoughts": thoughts})

        if resp.stop_reason != "tool_use":
            return resp.content[0].text, trace   # Final Answer

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                try:
                    obs = registry[block.name](**block.input)
                except Exception as e:
                    obs = f"ERROR: {e}"          # let the model recover
                trace.append({"action": block.name, "args": block.input, "observation": obs})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(obs),
                })
        messages.append({"role": "user", "content": tool_results})

    return "Reached step limit.", trace
```

### ReAct vs. plain tool calling
They're the same machinery; ReAct is the *discipline* of keeping reasoning explicit and the loop observable. Every production agent you build is effectively ReAct + memory + guardrails.

### Real-time use case
**Fraud investigation copilot:** analyst asks *"Is transaction TXN-8842 suspicious?"* The agent reasons about which signals to check, pulls the device fingerprint, the velocity stats, and the geo-history, observes each, and concludes — leaving a Thought/Action/Observation trail the analyst can defend.

---

## 4. Agentic Design Patterns

Four patterns compose into most real systems. You rarely use one alone.

### 4.1 Planning
The agent first produces a **plan** (list of subtasks) before executing. Decouples *deciding what to do* from *doing it*.

```python
PLAN_PROMPT = """You are a planning module. Break the goal into an ordered list
of atomic, tool-executable steps. Return JSON: {"steps": ["...", "..."]}.
Do not execute anything. Goal: {goal}"""

def make_plan(goal):
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512,
        messages=[{"role": "user", "content": PLAN_PROMPT.format(goal=goal)}],
    )
    return json.loads(resp.content[0].text)["steps"]

# Goal: "Onboard vendor Acme for co-lending"
# -> ["Verify GST registration", "Run sanctions check", "Fetch credit rating",
#     "Validate bank account", "Draft onboarding summary"]
```

**Use case:** multi-step vendor onboarding where the steps must be *shown to a compliance officer for approval before execution* (plan-then-approve-then-execute).

### 4.2 Reflection (self-critique)
The agent reviews its own output against criteria and revises. Dramatically improves quality on generation tasks.

```python
def reflect_and_revise(draft, rubric):
    critique = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=512,
        messages=[{"role": "user", "content":
            f"Critique this draft against the rubric. List concrete defects only.\n"
            f"RUBRIC:\n{rubric}\n\nDRAFT:\n{draft}"}],
    ).content[0].text

    revised = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1024,
        messages=[{"role": "user", "content":
            f"Revise the draft to fix every defect.\nDEFECTS:\n{critique}\n\nDRAFT:\n{draft}"}],
    ).content[0].text
    return revised, critique
```

**Use case:** generating a customer-facing loan rejection letter that must be empathetic, legally precise, and free of prohibited language — reflect against a compliance rubric before sending.

### 4.3 Tool Use
Covered in §2. As a *pattern*, the discipline is: keep tools small, single-purpose, and well-described; prefer many narrow tools over one god-tool with a `mode` parameter.

### 4.4 Multi-step Reasoning / Decomposition
For hard analytical questions, force intermediate steps rather than a leap to the answer. Overlaps with Chain-of-Thought but applied to *tool-grounded* problems.

### How they compose (a real orchestration)
```
Plan  ──▶ for each step: ReAct(tool use) ──▶ Reflect on step output
                                                   │
                                          (fail) ◀─┴─▶ (pass) next step
```

**Composed use case — Sona-style loan memo:** Plan the memo sections → for each section, ReAct over data tools (bureau, bank statements, GST) → Reflect each section against RBI disclosure rules → assemble → final Reflection pass on the whole memo.

---

## 5. Agent Memory

Memory is *what state persists and for how long*. Three tiers, each with a different store and lifecycle.

| Tier | Scope | Store | Example |
|------|-------|-------|---------|
| **Short-term** | Within one run | In-context (the message list / scratchpad) | The account_id extracted 3 turns ago |
| **Long-term** | Across runs, semantic | Vector DB + metadata | "This customer prefers EUR statements" |
| **Episodic** | Across runs, by event | Structured log / DB rows | "On 2026-03-01 the agent escalated app #4471" |

### 5.1 Short-term (working memory)
It's just the conversation/scratchpad you carry in the loop. The constraint is the **context window** and cost. Techniques when it gets long:
- **Summarize-and-drop:** compress old turns into a running summary.
- **Sliding window:** keep the last N turns verbatim + a summary of the rest.

```python
def compress_memory(scratchpad, keep_last=4):
    if len(scratchpad) <= keep_last:
        return scratchpad
    old, recent = scratchpad[:-keep_last], scratchpad[-keep_last:]
    summary = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=256,
        messages=[{"role": "user",
                   "content": f"Summarize these agent steps into 3 bullet facts:\n{old}"}],
    ).content[0].text
    return [{"role": "summary", "content": summary}] + recent
```

### 5.2 Long-term (semantic memory via vector DB)
Store facts as embeddings; retrieve by similarity when relevant. This is RAG applied to *memory*.

```python
# Pseudocode with a vector store (Chroma / pgvector / Qdrant)
def remember(text, metadata):
    vec = embed(text)
    vectorstore.add(vector=vec, document=text, metadata=metadata)

def recall(query, k=5, filters=None):
    return vectorstore.query(embed(query), top_k=k, where=filters)

# On each run, inject relevant long-term memories:
memories = recall(user_message, filters={"customer_id": cid})
context = "\n".join(m.document for m in memories)
```

> **Regulated-industry caveat:** long-term memory is **stored personal data**. It falls under DPDP/GDPR — you need retention limits, a deletion path (right-to-erasure must purge the vector rows *and* their embeddings), and access controls scoped per customer via metadata filters. Don't let one customer's memory leak into another's retrieval.

### 5.3 Episodic (past runs)
A structured record of *what the agent did*, queryable for learning and audit. Distinct from semantic memory: it's the event log, not the fact store.

```python
episode = {
    "run_id": "run_4471",
    "timestamp": "2026-03-01T09:12:00Z",
    "goal": "Assess loan application #4471",
    "actions": [...],            # full ReAct trace
    "outcome": "escalated_to_human",
    "reason": "DTI 0.58 exceeds policy 0.50",
    "tokens": 8432, "cost_usd": 0.041,
}
episodic_db.insert(episode)
# Later: "Show all applications escalated for DTI in March" -> analytics + few-shot examples
```

**Use case:** feed successful past episodes back as **few-shot examples** so the agent improves at recurring tasks — while the failures feed your eval set.

---

## 6. Agent Observability

You cannot operate an agent you cannot see. Because agents are non-deterministic and multi-step, observability isn't optional — it's the difference between "a demo" and "a system on-call at 2am."

### The three things you must capture

1. **Traces** — the full step-by-step: every Thought, Action, Args, Observation.
2. **Tool telemetry** — latency, success/failure, and the actual inputs/outputs of each tool call.
3. **Cost & tokens** — per step, per run, per user — because agent cost is unbounded by default.

### A lightweight tracing decorator

```python
import time, uuid, json, logging
logger = logging.getLogger("agent")

def traced(tool_name):
    def deco(fn):
        def wrapper(**kwargs):
            span_id = str(uuid.uuid4())[:8]
            t0 = time.perf_counter()
            try:
                result = fn(**kwargs)
                status = "ok"
                return result
            except Exception as e:
                status, result = "error", str(e)
                raise
            finally:
                logger.info(json.dumps({
                    "span_id": span_id,
                    "tool": tool_name,
                    "args": redact_pii(kwargs),      # never log raw PII
                    "status": status,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                }))
        return wrapper
    return deco

@traced("get_account_balance")
def get_account_balance(account_id, currency="INR"):
    ...
```

### Token & cost accounting

```python
class UsageMeter:
    # rough per-1M-token rates; keep these in config, not code
    RATES = {"input": 3.00, "output": 15.00}   # USD per 1M tokens (illustrative)

    def __init__(self):
        self.in_tok = self.out_tok = 0

    def add(self, resp):
        self.in_tok  += resp.usage.input_tokens
        self.out_tok += resp.usage.output_tokens

    @property
    def cost_usd(self):
        return (self.in_tok/1e6)*self.RATES["input"] + (self.out_tok/1e6)*self.RATES["output"]

# Enforce a per-run budget guardrail:
if meter.cost_usd > MAX_RUN_BUDGET:
    raise BudgetExceeded("Agent halted: run budget exceeded")
```

### Production tooling
Roll-your-own is fine for learning; in production wire into **OpenTelemetry** (vendor-neutral spans) and/or a purpose-built LLM observability platform (Langfuse, LangSmith, Arize Phoenix, Helicone). They give you trace waterfalls, replay, eval harnesses, and cost dashboards out of the box. The key is: emit OTel-compatible spans so you're not locked in.

### The four golden signals for agents
- **Step count distribution** — sudden rise = the agent is looping / confused.
- **Tool error rate** — a spiking tool poisons downstream reasoning.
- **Cost per successful outcome** — the real unit economics.
- **Human-escalation rate** — trending up may mean drift or a broken tool.

> **Audit angle:** persist the full trace + inputs/outputs immutably (e.g., append-only store) with the model version and prompt version stamped on each run. When outcomes are challenged, you can reproduce exactly what happened.

---

## 7. Lab: Research Agent (Full Build)

Build an agent that: (1) accepts a research question, (2) searches the web, (3) reads relevant pages, (4) synthesizes findings into a structured report, (5) saves the report to a file. This exercises the whole module: tool calling, the ReAct loop, memory, and observability.

### Architecture

```
                ┌─────────────────────────────────────────┐
   question ───▶│  ReAct Loop (LLM = Claude)               │
                │   Thought → Action → Observation → repeat│
                └───────┬───────────┬───────────┬──────────┘
                        │           │           │
                   web_search   read_url    write_file
                        │           │           │
                        └── Observability: trace + tokens + cost ──┘
```

### Full implementation

```python
"""
research_agent.py — a ReAct research agent with tracing.
Deps: anthropic, requests, beautifulsoup4  (swap web_search for your provider)
"""
import os, json, time, uuid, logging, requests
from bs4 import BeautifulSoup
import anthropic

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("research_agent")
client = anthropic.Anthropic()

# ---------- Observability -------------------------------------------------
class Trace:
    def __init__(self, goal):
        self.run_id = str(uuid.uuid4())[:8]
        self.goal = goal
        self.events = []
        self.in_tok = self.out_tok = 0
    def record(self, **kw):
        self.events.append({"t": time.time(), **kw})
        log.info(json.dumps({"run": self.run_id, **kw}, default=str)[:500])
    def meter(self, resp):
        self.in_tok += resp.usage.input_tokens
        self.out_tok += resp.usage.output_tokens
    @property
    def cost(self):  # illustrative rates
        return self.in_tok/1e6*3 + self.out_tok/1e6*15

# ---------- Tools ---------------------------------------------------------
def web_search(query: str) -> list:
    """Return a list of {title, url, snippet}. Plug in Brave/Tavily/SerpAPI here."""
    # Example with a generic search API (pseudo — replace with your key/provider):
    r = requests.get("https://api.search.example/search",
                     params={"q": query, "count": 5}, timeout=10)
    return [{"title": x["title"], "url": x["url"], "snippet": x["snippet"]}
            for x in r.json().get("results", [])]

def read_url(url: str) -> str:
    """Fetch a page and return cleaned text (first ~4000 chars)."""
    html = requests.get(url, timeout=15, headers={"User-Agent": "research-agent/1.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:4000]

def write_file(filename: str, content: str) -> str:
    """Persist the final report."""
    path = os.path.join("reports", filename)
    os.makedirs("reports", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved {len(content)} chars to {path}"

REGISTRY = {"web_search": web_search, "read_url": read_url, "write_file": write_file}

TOOLS = [
    {"name": "web_search",
     "description": "Search the web for a query. Use to discover sources. Returns title/url/snippet list.",
     "input_schema": {"type": "object",
        "properties": {"query": {"type": "string", "description": "Focused search query."}},
        "required": ["query"]}},
    {"name": "read_url",
     "description": "Fetch and read the main text of a specific URL. Use after web_search to read a promising source.",
     "input_schema": {"type": "object",
        "properties": {"url": {"type": "string", "description": "Full https URL to read."}},
        "required": ["url"]}},
    {"name": "write_file",
     "description": "Save the final structured report to a markdown file. Call this LAST, once, when the report is complete.",
     "input_schema": {"type": "object",
        "properties": {"filename": {"type": "string", "description": "e.g. report.md"},
                       "content":  {"type": "string", "description": "Full markdown report."}},
        "required": ["filename", "content"]}},
]

SYSTEM = """You are a rigorous research agent. Work in a Reason→Act→Observe loop.
Strategy:
1. Search for the question to find sources.
2. Read the 2-4 most relevant URLs.
3. Cross-check claims across sources; note disagreements.
4. Synthesize a structured markdown report:
   ## Question  ## Key Findings  ## Evidence (with source URLs)  ## Caveats
5. Call write_file exactly once with the final report, then give a one-line summary.
Reason explicitly before each action. Do not fabricate sources."""

# ---------- Agent loop ----------------------------------------------------
def research(question: str, max_steps: int = 10):
    tr = Trace(question)
    messages = [{"role": "user", "content": question}]

    for step in range(max_steps):
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=2048,
            system=SYSTEM, tools=TOOLS, messages=messages,
        )
        tr.meter(resp)

        for b in resp.content:                       # log reasoning
            if b.type == "text" and b.text.strip():
                tr.record(step=step, thought=b.text.strip())

        if resp.stop_reason != "tool_use":
            tr.record(step=step, final=resp.content[0].text, cost_usd=round(tr.cost, 4))
            return resp.content[0].text, tr

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                t0 = time.perf_counter()
                try:
                    obs = REGISTRY[b.name](**b.input)
                    status = "ok"
                except Exception as e:
                    obs, status = f"ERROR: {e}", "error"
                tr.record(step=step, action=b.name, args=b.input, status=status,
                          latency_ms=round((time.perf_counter()-t0)*1000, 1))
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": str(obs)[:6000]})
        messages.append({"role": "user", "content": results})

    tr.record(final="Stopped: max steps.", cost_usd=round(tr.cost, 4))
    return "Reached step limit without completing.", tr


if __name__ == "__main__":
    answer, trace = research(
        "What are the RBI Digital Lending Directions 2025 requirements for co-lending?"
    )
    print("\n=== ANSWER ===\n", answer)
    print(f"\n[run {trace.run_id}] tokens in/out={trace.in_tok}/{trace.out_tok} "
          f"cost=${trace.cost:.4f} steps={len({e.get('step') for e in trace.events})}")
```

### What to observe when you run it
- Watch the **Thoughts** — you'll see it decide *which* sources to read, not read all of them.
- Watch **tool latency** — `read_url` dominates; that's your optimization target (parallelize reads).
- Watch **cost** — long pages inflate input tokens; the `[:4000]` truncation is a deliberate cost guardrail.

### Extensions (turn the lab into a portfolio piece)
1. **Parallel reads** — issue multiple `read_url` calls concurrently (batch tool use) to cut wall-clock time.
2. **Reflection pass** — after drafting, add a self-critique step (§4.2) against a rubric ("every claim has a source URL").
3. **Long-term memory** — cache read pages in a vector store so repeat questions skip re-fetching (§5.2).
4. **Guardrails** — domain allow-list for `read_url`, per-run budget cap (§6), and PII redaction in logs.
5. **Human-in-the-loop** — before `write_file`, surface the report for approval (mandatory for anything customer-facing in BFSI).

---

## 8. From Single Agent to Orchestration

One agent hits limits: too many tools confuse routing, one context window can't hold everything, and a single loop can't parallelize. **Orchestration** coordinates multiple specialized agents.

### Common topologies

| Pattern | Shape | When to use |
|---------|-------|-------------|
| **Supervisor / Orchestrator-Worker** | A lead agent routes subtasks to specialist agents | Distinct skills: a "search agent", a "SQL agent", a "writer agent" |
| **Sequential pipeline** | Agent A → B → C, each transforms | Clear stages with handoffs (extract → validate → decide) |
| **Parallel fan-out / fan-in** | Split work, run concurrently, merge | Independent subtasks (research 5 competitors at once) |
| **Debate / critic** | One generates, another critiques, loop | High-stakes correctness (Reflection scaled to two agents) |

### Supervisor sketch

```python
def supervisor(goal):
    plan = make_plan(goal)                      # §4.1 Planning
    results = {}
    for task in plan:
        specialist = route(task)                # pick sql_agent / search_agent / writer_agent
        results[task] = specialist.run(task, context=results)   # shared memory
    return writer_agent.run("Compose final deliverable", context=results)
```

### When to choose orchestration vs. a single agent
- **Single agent** until: the tool count exceeds ~8–10, the context won't fit, or you need parallelism.
- **Orchestration** buys modularity and parallelism at the cost of more moving parts, more places to fail, and harder end-to-end tracing. Instrument the *handoffs* especially — that's where multi-agent systems silently lose information.

> **FDE reality check:** most "we need a multi-agent system" requirements are actually a single agent with better tools and a Reflection pass. Reach for orchestration when a specialist genuinely needs a *different* system prompt, tool set, or model — not just as architectural ambition.

---

## 9. Production Hardening Checklist (BFSI lens)

Before an agent touches a regulated workflow:

- [ ] **Determinism boundary defined** — which decisions the LLM may make vs. which are hard-coded rules. (An agent should *never* be the sole approver of a loan; it prepares, a rule or human decides.)
- [ ] **Every tool re-validates authz** independently of the model's proposed inputs.
- [ ] **Full trace persisted immutably** with model + prompt version stamps (reproducibility for audit).
- [ ] **PII redaction** in all logs; long-term memory honors DPDP/GDPR retention & erasure.
- [ ] **Per-run and per-user budget caps** enforced (cost is unbounded by default).
- [ ] **Max-step / loop-breaker** so a confused agent can't run forever.
- [ ] **Human-in-the-loop gates** on any irreversible or customer-facing action (transfers, letters, decisions).
- [ ] **Eval set** built from real episodes (§5.3) — success rate, cost per outcome, escalation rate tracked over time.
- [ ] **Tool failure isolation** — one failing tool returns a recoverable error, doesn't crash the run.
- [ ] **Prompt-injection defense** on any tool that reads untrusted content (`read_url`): treat fetched text as data, never as instructions.

---

### One-paragraph mental model to keep

An agent is an LLM in a loop that can act through tools and remember across steps. You climb from prompt → chain → agent → multi-agent only when the requirement forces it. Tool calling is where the model *proposes* and your code *disposes* (and enforces security). ReAct keeps reasoning explicit and therefore auditable. Memory decides what state survives and for how long. Observability is what makes the whole thing operable. And in a regulated setting, the agent prepares and reasons — but rules and humans make the decisions that carry consequences.
