# MCP, AI Workflows & Automation — Practical, Hands-On Reference

> A working engineer's guide, not a slide summary. Every concept below ships with runnable code and a real use case drawn from lending, compliance, and agent-trust systems.
>
> **Currency note (July 2026):** The stable MCP spec is **2025-11-25**. A stateless revision (**2026-07-28**) is in release-candidate and removes the `initialize` handshake for horizontal scaling. Code here targets the **production-stable SDK lines** — Python `mcp[cli]` 1.x (`from mcp.server.fastmcp import FastMCP`) and TypeScript `@modelcontextprotocol/sdk` 1.x (`McpServer` + `registerTool`). SDK **v2** packages (`@modelcontextprotocol/server`, Python `mcp==2.x`) implement 2026-07-28 but are pre-release — don't ship them yet. MCP is now governed by the **Agentic AI Foundation** under the Linux Foundation (donated Dec 2025, co-founded with Block and OpenAI).

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [MCP vs API](#2-mcp-vs-api)
3. [MCP Components: Host, Client, Server](#3-mcp-components-host-client-server)
4. [Building an MCP Server](#4-building-an-mcp-server)
5. [AI Workflow Design](#5-ai-workflow-design)
6. [Automation Patterns](#6-automation-patterns)
7. [Putting It Together: A Reference Architecture](#7-putting-it-together)

---

## 1. What is MCP?

### The one-line definition

MCP (Model Context Protocol) is an open, JSON-RPC 2.0 protocol that standardizes how an LLM application discovers and safely calls external tools, data, and prompts — the "USB-C port for AI." Instead of hand-writing a bespoke integration for every model × every tool, you write one MCP server and any MCP-capable host (Claude Desktop, an IDE, ChatGPT, your own agent) can use it.

### The problem it actually solves: M×N → M+N

Before MCP, if you had **M** AI applications and **N** tools/data sources, you needed roughly **M×N** custom integrations. Each app re-implemented auth, schema, and call semantics for each tool. Add a model, re-integrate everything.

MCP collapses this to **M+N**: each host speaks MCP once, each tool exposes MCP once, and they interoperate. This is the same architectural move the Language Server Protocol (LSP) made for editors and language tooling — MCP is explicitly modelled on LSP.

```
   Without MCP (M×N glue)              With MCP (M+N)

   Claude ─┬─► Postgres              Claude ─┐
           ├─► Gmail                 IDE    ─┼─► [ MCP ] ─┬─► Postgres server
   IDE   ──┼─► GitHub                Agent  ─┘            ├─► Gmail server
           └─► Jira                                       ├─► GitHub server
   Agent ──┬─► Postgres                                   └─► Jira server
           └─► ...                   one contract, both sides
```

### The three server primitives

An MCP server can expose three kinds of capability. The mental model maps cleanly onto HTTP verbs:

| Primitive | Analogy | Who controls invocation | Purpose |
|-----------|---------|-------------------------|---------|
| **Tools** | `POST` — side effects | **Model-controlled** (the LLM decides to call) | Execute code, mutate state, call APIs |
| **Resources** | `GET` — read-only | **App-controlled** (host loads into context) | Feed data/context to the model |
| **Prompts** | Saved template | **User-controlled** (user invokes) | Reusable, parameterized interaction patterns |

There are also **client-side** primitives the server can request: **sampling** (server asks the client's LLM to run a completion), **elicitation** (server asks the user for more input mid-call), and **roots** (client tells server which filesystem/URIs it may touch). These are what make MCP "AI-native" rather than just an RPC layer.

### Why "safely" is the load-bearing word

MCP grants LLMs arbitrary code execution and data access, so the spec bakes in a trust model:

- **Explicit user consent is required before any tool call.** Hosts must present a consent affordance — the model cannot silently invoke.
- **Tool descriptions and annotations are untrusted unless the server is trusted.** A malicious server can lie about what a tool does ("poisoned tool" attack).
- **Prompt injection through tool results is a real, documented threat** — data returned by one tool can carry instructions that hijack subsequent calls and exfiltrate through another connected tool.

If you build agent-trust infrastructure, this is the seam to instrument: every tool call is a policy decision point (who, what, on whose behalf, with what scope), and MCP gives you a uniform place to enforce it.

### Minimal runnable server (Python) — see the whole thing in 15 lines

```python
# server.py  —  install:  uv add "mcp[cli]"
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hello-mcp")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""          # docstring becomes the tool description
    return a + b                     # type hints become the JSON Schema

@mcp.resource("config://app")
def app_config() -> str:
    """Read-only app configuration the model can load."""
    return '{"env": "prod", "region": "eu-central-1"}'

if __name__ == "__main__":
    mcp.run()                        # defaults to stdio transport
```

Test it without any LLM using the official Inspector:

```bash
uv run mcp dev server.py     # opens the MCP Inspector web UI
# or point Claude Desktop / Cursor at it via config (shown in §3)
```

You wrote **no** JSON Schema, request parsing, validation, or protocol handling. `a: int, b: int` *is* the schema.

---

## 2. MCP vs API

The bullet "MCP is context-aware and AI-native; REST APIs are generic" is directionally right but often misunderstood. **MCP does not replace REST APIs — it wraps them for consumption by an LLM.** Your REST API stays exactly where it is; an MCP server sits in front of it and translates.

### What actually differs

| Dimension | REST / generic API | MCP |
|-----------|-------------------|-----|
| **Primary consumer** | A programmer writing code ahead of time | An LLM deciding at runtime |
| **Discovery** | Human reads OpenAPI docs, hardcodes calls | Model calls `tools/list` and reads descriptions live |
| **Schema role** | Validation + docs | Also **the prompt** — the model reasons over descriptions to choose tools |
| **Interaction shape** | Request → response | Bidirectional: server can call back for sampling/elicitation |
| **Consent / trust** | App-layer concern, ad hoc | First-class in the protocol (consent before invocation) |
| **Statefulness** | Usually stateless | Stateful today (2025-11-25); stateless in 2026-07-28 RC |
| **Granularity** | Endpoints designed for developers | Tools designed for *model reasoning* — small, sharply named |

### The design rule that trips people up

Models reason better about **three well-named tools than one tool with a 20-field input**. A REST endpoint `POST /loans` with a giant body is fine for a developer. For an LLM, split it: `check_eligibility`, `price_loan`, `create_application`. Smaller tools, sharper descriptions, fewer hallucinated arguments. This is the single most impactful thing you can do to make an MCP server "AI-native" rather than a thin REST passthrough.

### When to use which

- **Use a plain API/SDK call** when your *code* knows exactly what to call. Don't put MCP between two services you control that never involve a model.
- **Use MCP** when a *model/agent* needs to decide, at runtime, which capability to use — and when you want that decision gated by consent and audited uniformly.

### Hands-on: wrapping an existing REST API as an MCP server

Real use case — you already run a KYC/credit-bureau REST service. You want an underwriting agent to use it, safely and with an audit trail.

```python
# kyc_mcp.py  —  wraps an existing internal REST API for agent consumption
import os, httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kyc-bureau")
BASE = os.environ["KYC_API_BASE"]                       # your existing service
client = httpx.AsyncClient(base_url=BASE, timeout=10.0)

@mcp.tool()
async def credit_score(pan: str) -> dict:
    """Fetch the current credit score for an Indian PAN.
    Returns {score:int, bureau:str, as_of:str}. Read-only; safe to retry."""
    r = await client.get(f"/bureau/score", params={"pan": pan},
                         headers={"Authorization": f"Bearer {os.environ['KYC_TOKEN']}"})
    r.raise_for_status()
    return r.json()

@mcp.tool()
async def kyc_status(customer_id: str) -> dict:
    """Return KYC verification status for a customer.
    Never mutates state. Fields: {status, level, last_verified}."""
    r = await client.get(f"/kyc/{customer_id}")
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # remote-served for a hosted agent
```

Notice what the wrapper *adds* over the raw REST endpoint: LLM-legible descriptions, explicit "read-only / safe to retry" hints (which map to MCP tool annotations that hosts use to decide whether to auto-approve), and a single choke point where you can later insert auth-context propagation and per-call audit logging.

---

## 3. MCP Components: Host, Client, Server

Three roles, and the distinction between the first two matters when you debug.

```
┌─────────────────────────── HOST (e.g. Claude Desktop, IDE, your agent) ──────────────────────────┐
│  Owns the LLM conversation, enforces consent, decides which servers to connect                    │
│                                                                                                   │
│   ┌── MCP CLIENT ──┐        ┌── MCP CLIENT ──┐        ┌── MCP CLIENT ──┐                            │
│   │ 1:1 connection │        │ 1:1 connection │        │ 1:1 connection │                            │
│   └───────┬────────┘        └───────┬────────┘        └───────┬────────┘                            │
└───────────┼─────────────────────────┼─────────────────────────┼───────────────────────────────────┘
            │ JSON-RPC 2.0             │                          │
            ▼                          ▼                          ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │  MCP SERVER   │          │  MCP SERVER   │          │  MCP SERVER   │
    │  (Postgres)   │          │  (Filesystem) │          │  (KYC API)    │
    └───────────────┘          └───────────────┘          └───────────────┘
```

| Component | What it is | Concrete example | Responsibility |
|-----------|-----------|------------------|----------------|
| **Host** | The AI application the user interacts with | Claude Desktop, Cursor, ChatGPT desktop, your custom agent | Manages LLM, enforces **consent**, aggregates multiple servers, decides what to expose to the model |
| **Client** | A connector *inside* the host, one per server | The protocol object the SDK creates | Maintains a **1:1** stateful session with exactly one server; handles capability negotiation |
| **Server** | A process exposing tools/resources/prompts | Your `kyc_mcp.py`, a Postgres server | Provides capabilities; must not trust the client blindly either |

Key nuance: **one host spins up many clients, each bound to exactly one server.** When "the MCP connection dropped," you're debugging a specific client↔server pair, not the whole host.

### Message flow (lifecycle)

On the stable 2025-11-25 spec:

1. **Initialize** — client and server handshake, exchange protocol version + capabilities.
2. **Discovery** — client calls `tools/list`, `resources/list`, `prompts/list`.
3. **Use** — model chooses a tool; host gets consent; client sends `tools/call`; server executes and returns.
4. **Notifications** — server can push `notifications/tools/list_changed`, progress updates, logs.

> The **2026-07-28** revision removes the initialize handshake and protocol-level session so servers scale behind a round-robin load balancer with no sticky sessions. Design new servers to be as stateless as possible now to ease that migration.

### Transports — the two that matter

| Transport | Use when | How it works |
|-----------|----------|--------------|
| **stdio** | Local — host spawns the server as a child process | JSON-RPC over stdin/stdout. **Never `print()` to stdout** in a stdio server — it corrupts the JSON-RPC stream. Log to stderr. |
| **Streamable HTTP** | Remote / hosted server, multiple clients | JSON-RPC over HTTP POST to one `/mcp` endpoint; optional SSE streaming for server→client messages. Replaces the old HTTP+SSE transport (kept only for backward compat). |

### Wiring a server into Claude Desktop (stdio)

Real, copy-pasteable. Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "kyc-bureau": {
      "command": "uv",
      "args": ["run", "--with", "mcp[cli]", "python", "/abs/path/kyc_mcp.py"],
      "env": { "KYC_API_BASE": "https://internal.kyc.local", "KYC_TOKEN": "..." }
    }
  }
}
```

Restart the host; the tools appear and every call prompts for consent.

---

## 4. Building an MCP Server

The instruction is "expose any tool (DB, API, filesystem) via MCP." Below are all three, production-shaped, in Python and TypeScript. The running example is an **agentic lending back office** — the kind of system where an agent needs read access to a loan DB, write access to a decision log, and file access for uploaded documents, all under strict guardrails.

### 4a. Exposing a database (Postgres) — read-mostly, injection-safe

The cardinal rule: **never let the model write raw SQL that you `execute()` verbatim.** Expose *intent-shaped* tools with parameterized queries, and expose schema/metadata as **resources** so the model can reason without touching data.

```python
# lending_db_mcp.py
import os, asyncpg
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lending-db", instructions=(
    "Read-only loan analytics. Call list_tables and get_schema before querying. "
    "All amounts are in INR paise. Results are capped at 500 rows."
))
_pool: asyncpg.Pool | None = None

async def pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=5)
    return _pool

# --- RESOURCE: schema as read-only context (no data leaves here) ---
@mcp.resource("schema://loans")
async def loans_schema() -> str:
    """Column definitions for the loans table, for the model to reason over."""
    p = await pool()
    rows = await p.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'loans' ORDER BY ordinal_position")
    return "\n".join(f"{r['column_name']}: {r['data_type']}" for r in rows)

# --- TOOL: intent-shaped, parameterized, never raw SQL ---
@mcp.tool()
async def loans_by_status(status: str, limit: int = 50) -> list[dict]:
    """List loans filtered by status ('active','overdue','closed','disbursed').
    Read-only. limit is clamped to 500."""
    if status not in {"active", "overdue", "closed", "disbursed"}:
        raise ValueError(f"invalid status: {status}")
    limit = max(1, min(limit, 500))
    p = await pool()
    rows = await p.fetch(
        "SELECT loan_id, customer_id, principal_paise, status, dpd "
        "FROM loans WHERE status = $1 ORDER BY dpd DESC LIMIT $2", status, limit)
    return [dict(r) for r in rows]

@mcp.tool()
async def portfolio_at_risk(min_dpd: int = 30) -> dict:
    """Compute Portfolio-at-Risk: outstanding principal for loans past `min_dpd`
    days-past-due, as a fraction of total outstanding. Read-only."""
    p = await pool()
    row = await p.fetchrow(
        "SELECT COALESCE(SUM(principal_paise) FILTER (WHERE dpd >= $1),0)::float "
        "       / NULLIF(SUM(principal_paise),0) AS par, "
        "       COUNT(*) FILTER (WHERE dpd >= $1) AS at_risk_count "
        "FROM loans WHERE status IN ('active','overdue','disbursed')", min_dpd)
    return {"par_ratio": row["par"], "at_risk_count": row["at_risk_count"], "min_dpd": min_dpd}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

Guardrails demonstrated: an allow-list on `status`, a clamped `limit`, parameterized queries (`$1`, `$2`), schema exposed as a **resource** so the model never needs `SELECT *` to learn structure, and analytics pushed into SQL rather than pulling raw rows the model then sums (cheaper, safer, deterministic).

### 4b. Exposing a filesystem — sandboxed with `roots`

Documents (bank statements, uploaded KYC) live on disk. Expose read access but **jail every path** to a sandbox and resolve symlinks — path traversal is the classic exploit.

```python
# docs_fs_mcp.py
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("kyc-docs")
SANDBOX = Path(os.environ.get("DOC_ROOT", "/srv/kyc-uploads")).resolve()

def _safe(rel: str) -> Path:
    """Resolve `rel` under SANDBOX or raise. Blocks ../ traversal and symlink escapes."""
    p = (SANDBOX / rel).resolve()
    if not p.is_relative_to(SANDBOX):          # Python 3.9+
        raise ValueError("path escapes sandbox")
    return p

@mcp.tool()
def list_documents(customer_id: str) -> list[str]:
    """List uploaded document filenames for a customer. Read-only."""
    folder = _safe(customer_id)
    return [f.name for f in folder.iterdir()] if folder.exists() else []

@mcp.tool()
def read_document(customer_id: str, filename: str) -> str:
    """Read a text/markdown document for a customer. Read-only, sandboxed."""
    p = _safe(f"{customer_id}/{filename}")
    if p.suffix.lower() not in {".txt", ".md", ".csv"}:
        raise ValueError("only text documents may be read")
    return p.read_text(encoding="utf-8", errors="replace")[:200_000]

if __name__ == "__main__":
    mcp.run()
```

### 4c. Same idea in TypeScript (production-stable SDK v1.x)

```typescript
// server.ts  —  npm i @modelcontextprotocol/sdk zod
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer(
  { name: "lending-db", version: "1.0.0" },
  { instructions: "Read-only loan analytics. Amounts in INR paise. Max 500 rows." }
);

server.registerTool(
  "loans_by_status",
  {
    title: "Loans by status",
    description: "List loans filtered by status. Read-only.",
    inputSchema: {
      status: z.enum(["active", "overdue", "closed", "disbursed"]),
      limit: z.number().int().min(1).max(500).default(50),
    },
    outputSchema: { loans: z.array(z.object({ loanId: z.string(), dpd: z.number() })) },
    annotations: { readOnlyHint: true },     // host can auto-approve read-only tools
  },
  async ({ status, limit }) => {
    const loans = await db.loansByStatus(status, limit);  // your parameterized query
    return {
      content: [{ type: "text", text: JSON.stringify(loans) }],
      structuredContent: { loans },
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

For a **remote** TS server, swap the transport for `StreamableHTTPServerTransport` mounted on a single `/mcp` Express route, authenticate the bearer token in middleware, and **start stateless** (`sessionIdGenerator: undefined`) — add sessions only if you genuinely need resumability. Enable DNS-rebinding protection with an `allowedHosts` allow-list when serving over HTTP.

### Testing your server

```bash
# The Inspector is the fastest loop — no LLM needed
npx @modelcontextprotocol/inspector          # then connect to http://localhost:8000/mcp
```

Write in-memory tests too: call your handler functions directly with sample args and assert on the structured output. Tool logic is just functions — unit-test them like any other.

---

## 5. AI Workflow Design

"Directed graphs of AI steps + human checkpoints." The core insight: reliable AI systems are **not** one giant prompt. They're a graph where each node is a bounded, testable step, edges route based on results, and humans sit at the checkpoints that carry real risk.

### The distinction that governs everything: Workflow vs Agent

- **Workflow** — LLM steps orchestrated through **predefined code paths**. You control the graph. Predictable, debuggable, cheaper. *Default to this.*
- **Agent** — the LLM **dynamically directs its own process**, choosing tools and steps at runtime (this is where MCP shines). More flexible, less predictable, harder to bound.

Reach for an agent only when the task genuinely can't be expressed as a fixed graph. Most production "AI features" should be workflows.

### The five workflow patterns (composable building blocks)

```
1. CHAINING          A ─► B ─► C            each step's output feeds the next
                          │
                     [gate] optionally check between steps

2. ROUTING           ┌─► specialist_A
           classify ─┼─► specialist_B       classify first, then dispatch
                     └─► specialist_C

3. PARALLELIZATION   ┌─► worker_1 ─┐
              split ─┼─► worker_2 ─┼─► aggregate    (sectioning or voting)
                     └─► worker_3 ─┘

4. ORCHESTRATOR-     orchestrator ─► spawns subtasks dynamically ─► synthesize
   WORKERS           (worker count decided at runtime, not compile time)

5. EVALUATOR-        generator ⇄ evaluator     loop until the evaluator passes
   OPTIMIZER         (one LLM produces, another critiques, repeat)
```

### Human checkpoints (human-in-the-loop)

A checkpoint is a node that **pauses the graph** and requires human approval before proceeding. Put them where an action is:

- **Irreversible** (disbursing funds, sending money on-chain, deleting data)
- **High-value or regulated** (loan approval above a threshold, anything touching regulated capital)
- **Low-confidence** (the model's own confidence or an evaluator score is below a bar)

The pattern is: run cheap/reversible steps automatically, **stop at the risky edge**, surface the full context to a human, resume on approval. In MCP terms, **elicitation** is the native mechanism — a tool can pause and request human input mid-execution.

### Hands-on: a loan-underwriting workflow (routing + chaining + checkpoint)

This is a directed graph, not an autonomous agent. Each node is bounded and testable; the human checkpoint gates the irreversible step.

```python
# underwriting_workflow.py  —  a DAG of AI + code steps with a human gate
from dataclasses import dataclass
from enum import Enum

class Decision(Enum):
    AUTO_APPROVE = "auto_approve"
    AUTO_DECLINE = "auto_decline"
    NEEDS_HUMAN  = "needs_human"

@dataclass
class Application:
    customer_id: str
    amount_paise: int
    credit_score: int
    par_flag: bool          # is this customer in a risky cohort?

# --- NODE 1: deterministic routing (cheap code, no LLM) ---
def route(app: Application) -> Decision:
    if app.credit_score >= 780 and app.amount_paise <= 5_00_000_00 and not app.par_flag:
        return Decision.AUTO_APPROVE
    if app.credit_score < 600 or app.par_flag:
        return Decision.AUTO_DECLINE
    return Decision.NEEDS_HUMAN

# --- NODE 2: LLM step, only for the ambiguous middle (via MCP tools) ---
async def llm_assess(app: Application, mcp_client) -> dict:
    """Model pulls context via MCP (bureau, docs) and produces a structured
    recommendation. Bounded output — it recommends, it does not disburse."""
    # model calls credit_score(), read_document(), portfolio_at_risk() over MCP
    # returns e.g. {"recommendation": "approve", "confidence": 0.72, "rationale": "..."}
    ...

# --- NODE 3: HUMAN CHECKPOINT (the graph pauses here) ---
async def human_checkpoint(app: Application, assessment: dict) -> bool:
    """Blocks until a credit officer approves. Full context surfaced to them.
    Returns True to proceed to disbursal, False to decline."""
    return await request_officer_approval(app, assessment)   # your queue/UI

# --- NODE 4: irreversible action, ONLY reached past the gate ---
async def disburse(app: Application):
    ...  # move money — never auto-reached without a checkpoint upstream

# --- The orchestration graph ---
async def underwrite(app: Application, mcp_client):
    decision = route(app)                              # edge 1
    if decision is Decision.AUTO_DECLINE:
        return {"result": "declined", "path": "auto"}
    if decision is Decision.NEEDS_HUMAN:
        assessment = await llm_assess(app, mcp_client) # edge 2 (LLM)
        if not await human_checkpoint(app, assessment):  # gate
            return {"result": "declined", "path": "human"}
    await disburse(app)                                # only past the gate
    return {"result": "approved"}
```

Why this shape wins in production: the LLM only runs on the ~20% ambiguous cases (cost control), the deterministic router handles the clear-cut ones (predictable + auditable), and the irreversible edge is structurally unreachable without a human. That structural guarantee — "money cannot move without passing node 3" — is the kind of property you want provable, not hoped-for.

> **Framework note:** for real graphs, use an orchestration library (LangGraph, or a durable-execution engine like Temporal for long-running/retryable workflows) rather than hand-rolling. The pattern above is the mental model they implement — nodes, edges, checkpoints, and resumable state.

---

## 6. Automation Patterns

Three ways to *start* an AI workflow. Same graph from §5 — what differs is the trigger.

| Pattern | Fires when | Latency | Typical use |
|---------|-----------|---------|-------------|
| **Scheduled** | A clock/cron says so | Minutes–hours | Nightly reconciliation, daily risk report, weekly digest |
| **Event-driven** | A message arrives on a bus/queue | Seconds | Reactive pipelines, fan-out, decoupled microservices |
| **Trigger-based (webhook)** | An external system calls you | Sub-second | Real-time fraud checks, "new application submitted" |

### 6a. Scheduled — nightly portfolio risk report

```python
# scheduled_risk.py  —  run under cron/APScheduler/K8s CronJob
import asyncio
from datetime import date

async def nightly_par_report():
    """03:00 IST: pull PAR via the MCP lending-db server, have the LLM draft a
    commentary, and post to the risk channel. No human in the loop — read-only."""
    par = await mcp.call_tool("portfolio_at_risk", {"min_dpd": 30})   # MCP tool
    commentary = await llm_summarize(par)          # bounded LLM step
    await post_to_slack("#risk", f"PAR@30dpd {date.today()}: "
                                 f"{par['par_ratio']:.2%}\n{commentary}")

# crontab:  0 3 * * *  /usr/bin/python /app/scheduled_risk.py
```

Rule of thumb: schedule work that is **read-only or fully reversible**. Anything irreversible on a schedule needs a checkpoint or a dry-run + approval step.

### 6b. Event-driven — react to a domain event on a queue

Real use case: whenever a loan crosses 90 days-past-due, an event lands on the bus; a worker kicks off a collections-strategy workflow.

```python
# event_worker.py  —  consumes from Kafka/SQS/Redis Streams
async def on_loan_npa(event: dict):
    """Event: {'type':'loan.npa', 'loan_id':..., 'dpd':...}. Decoupled from the
    producer — the core ledger just emits; this worker reacts."""
    if event["type"] != "loan.npa":
        return
    ctx = await mcp.call_tool("loans_by_status", {"status": "overdue"})
    strategy = await llm_recommend_collections(event["loan_id"], ctx)
    # write to a queue for human review — collections actions are NOT auto-fired
    await enqueue_for_review(event["loan_id"], strategy)

async def main():
    async for msg in consume("loan-events"):       # your broker client
        await on_loan_npa(msg)
```

Event-driven shines when producers and consumers must scale and fail independently. The ledger doesn't know or care that an AI workflow exists downstream — it just emits `loan.npa`.

### 6c. Trigger-based (webhook) — real-time fraud screen

Real use case: a payments provider POSTs to your webhook the instant a transaction is initiated; you must return an allow/hold decision in well under a second.

```python
# webhook_trigger.py  —  FastAPI endpoint an external system calls
from fastapi import FastAPI, Request, HTTPException
import hmac, hashlib, os

app = FastAPI()
SECRET = os.environ["WEBHOOK_SECRET"].encode()

def verify(raw: bytes, sig: str) -> bool:
    expected = hmac.new(SECRET, raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)      # constant-time compare

@app.post("/webhook/txn")
async def on_transaction(request: Request):
    raw = await request.body()
    if not verify(raw, request.headers.get("X-Signature", "")):
        raise HTTPException(401, "bad signature")   # ALWAYS verify webhooks
    txn = await request.json()

    # Fast path: deterministic rules first (µs), LLM only on the gray zone
    if txn["amount_paise"] > 10_00_000_00 or txn["velocity_1h"] > 20:
        score = await llm_fraud_assess(txn)         # bounded, low-latency model call
        if score["risk"] > 0.8:
            return {"decision": "hold", "reason": score["reason"]}
    return {"decision": "allow"}
```

Three non-negotiables for triggers: **verify the signature** (HMAC, constant-time), **respond fast** (offload slow work to a queue and return `202` if needed), and **make it idempotent** (providers retry — dedupe on an event ID).

### Choosing a pattern

```
Does an external system need to notify you in real time?  ── yes ──►  Webhook trigger
                          │ no
Is the work reacting to internal domain events?           ── yes ──►  Event-driven
                          │ no
Is it periodic / batch?                                   ── yes ──►  Scheduled
```

They compose: a webhook can drop an event on a bus (decouple + absorb spikes), and a scheduled job can sweep for anything the real-time path missed (a safety net). Production systems usually run all three.

---

## 7. Putting It Together

A single coherent system — an **agentic lending back office** — using every concept above:

```
                          ┌──────────────── HOST (agent runtime) ────────────────┐
   Triggers               │  LLM + consent enforcement + audit of every tool call │
   ─────────              │                                                        │
   webhook  ──┐           │   MCP clients ──►  lending-db server   (§4a, DB)       │
   events   ──┼──► queue ─┼─►                  kyc-docs server     (§4b, files)    │
   cron     ──┘           │                    kyc-bureau server   (§2, API wrap)  │
                          └────────────────────────┬───────────────────────────────┘
                                                    │
                              WORKFLOW GRAPH (§5): route ─► assess ─► [HUMAN GATE] ─► disburse
                                                                          ▲
                                                    irreversible edge unreachable without approval
```

- **§1 MCP** gives every tool call one consent + audit seam.
- **§2** wraps your existing KYC REST API instead of rewriting it.
- **§3** the agent host runs three MCP clients, one per server.
- **§4** DB (intent-shaped, parameterized), filesystem (sandboxed), API (wrapped) — all exposed safely.
- **§5** a directed graph runs the LLM only on ambiguous cases and structurally gates disbursal behind a human.
- **§6** the same graph is kicked off by webhooks (real-time), events (reactive), and cron (batch + safety-net sweep).

### The through-line

MCP standardizes *the boundary* between models and the world; workflows impose *structure* on how models act; automation decides *when* they act. The safety properties you care about — consent before action, irreversible steps behind human gates, per-call audit, scoped least-privilege tools — live at the seams MCP exposes. Build those seams deliberately and the system is governable; skip them and you have an agent with unbounded reach and no paper trail.

---

### Quick-reference commands

```bash
# Python
uv add "mcp[cli]"                              # stable 1.x SDK + CLI
uv run mcp dev server.py                        # dev + Inspector
uv run mcp install server.py                    # register into Claude Desktop

# TypeScript
npm i @modelcontextprotocol/sdk zod             # stable 1.x SDK
npx @modelcontextprotocol/inspector             # Inspector

# Pin away from pre-release v2 in your manifests:
#   Python:  mcp>=1.27,<2
#   Node:    "@modelcontextprotocol/sdk": "^1"
```
