# Multi-Agent Systems & AI Frameworks — FDE Reference

> **Position in the library:** This sits on top of the *LLM service layer* principle from Project 5. Every agent, worker, and supervisor in this document should route its model calls through that single service layer — that's where your observability, cost accounting, evals, and guardrails already live. Multi-agent orchestration is a *control-flow* layer above it, not a replacement for it.
>
> **Versions this doc is verified against (July 2026):** `langgraph 1.2.7`, `langchain 1.3.11`, `langchain-core 1.4.8`. Every code block below was executed on these versions with deterministic mock models (no API keys) before inclusion. Where the real world needs an LLM, the mock stands in for the model's *decision* so the control flow is what's being proven.

---

## 0. The FDE lens: why this topic is a deal-maker

For a Forward-Deployed Engineer, multi-agent systems are where three of your competencies collide at once: **technical architecture** (topology, state, failure modes), **customer-facing judgment** (talking a customer *out* of a multi-agent system they don't need), and **eval-driven development** (proving the system works before it touches production data).

The single most valuable thing you can say in an FDE interview or a customer scoping call is **"you probably don't need multi-agent for this."** The frontier-lab consensus in 2026 — Anthropic's most explicitly — is that most production "agent" systems should be a workflow, or a single well-tooled agent, and that multi-agent is a *last* resort reserved for problems you genuinely cannot decompose ahead of time. Knowing *when the complexity is worth it* is the senior signal. Everything in this doc is organized around building that judgment, not just the mechanics.

Three numbers to anchor the economics, all from 2026 production write-ups:

- A supervisor architecture costs roughly **3× the tokens** of a single mega-agent, because every supervisor turn is a full LLM call.
- Multi-agent systems can **amplify errors** rather than reduce them — one widely-cited analysis put it at up to ~17× under naive designs, because an early mistake propagates through every downstream agent.
- But Anthropic's own research found that a well-coordinated multi-agent research system (parallel sub-agents under a lead planner) **outperformed** a single-agent baseline by a wide margin on breadth-heavy tasks. Both facts are true. The design is what decides which one you get.

---

## 1. Foundations: the augmented LLM, and agents vs. workflows

### 1.1 The atom: the augmented LLM

Before "multi-agent" means anything, get the atom right. The base unit of every agentic system is an **LLM augmented with three things**:

1. **Tools** — functions the model can call (search, DB query, code exec, API calls).
2. **Memory** — what it can read from and write to across turns (context, vector store, scratchpad).
3. **Retrieval** — the ability to pull in relevant external knowledge on demand (RAG).

Everything else — chains, supervisors, peer meshes — is just *wiring several augmented LLMs together*. If a single augmented LLM with good tools solves the problem, you are done. Ship it.

### 1.2 The distinction that governs every design decision

Anthropic draws a line that has become the industry's shared vocabulary. Internalize it:

| | **Workflow** | **Agent** |
|---|---|---|
| Control flow | Predefined in code — *you* own the path | The model chooses its next step from feedback — you own the *goal and guardrails*, not the branches |
| Predictability | High | Lower (trades predictability for flexibility) |
| Cost/latency | Bounded, known | Grows per autonomous turn |
| Failure mode | A step breaks (localizable) | Errors compound across turns |
| Use when | The path can be mapped ahead of time | You *cannot* hardcode the path but *can* verify progress |

The practical rule: **use the simplest thing that passes your eval.** Often that's a single LLM call. Sometimes it's a fixed workflow. Reserve true agency (and especially *multi*-agency) for cases where the path genuinely can't be predicted — coding across an unknown set of files, open-ended research, computer-use — but where you can still *check* whether the agent is making progress (tests pass, sources found, screenshot matches goal).

### 1.3 Frameworks hide the prompts — a warning worth repeating to customers

Frameworks (LangChain, CrewAI, etc.) make it fast to start by abstracting away LLM calls, tool parsing, and chaining. The cost is that they add layers that obscure the actual prompts and responses, which makes debugging harder and tempts teams to add complexity a simpler setup would have avoided. The frontier-lab advice: **start with direct API calls** — many of these patterns are a few lines of code — and if you do adopt a framework, make sure you understand what it's doing underneath. "Incorrect assumptions about what's under the hood" is a top source of customer error. As an FDE, you will repeatedly be the person who has to open the black box.

---

## 2. The five workflow patterns (learn these before any multi-agent topology)

These are the building blocks. Multi-agent topologies are mostly compositions of these. All five are workflows — deterministic control flow — which is exactly why they're reliable enough to reach for first.

### 2.1 Prompt chaining
Decompose a task into a fixed sequence of LLM calls, each operating on the previous output, optionally with programmatic gates ("checks") between steps.
**Use when:** the task cleanly splits into fixed subtasks and you're trading latency for accuracy by making each call easier.
**Example:** write a marketing outline → validate it meets criteria → expand to full copy → translate.

### 2.2 Routing
Classify the input, then send it to a specialized downstream handler.
**Use when:** inputs fall into distinct categories that are better served by different prompts/models, and classification is reliable.
**Example:** support triage — refunds go to a billing prompt, outages to a technical prompt, cheap queries to a small model and hard ones to a frontier model.
**Verified implementation** (§8.3 below).

### 2.3 Parallelization
Run independent subtasks concurrently, then merge. Two sub-flavors:
- **Sectioning** — split the work into independent pieces (e.g., three researchers on three sources).
- **Voting** — run the *same* task several times and aggregate for confidence (e.g., three safety checks, block if any flags).

**Use when:** speed matters, or you want multiple perspectives on one input. **Verified implementation** (§8.4).

### 2.4 Orchestrator–workers (a.k.a. supervisor–worker)
A central LLM **dynamically** decomposes the task, delegates to workers, and synthesizes results. The key difference from parallelization: subtasks are **not predefined** — the orchestrator decides them at runtime based on the input.
**Use when:** you can't predict the subtasks ahead of time. Anthropic's coding agents use this to change an unknown number of files per GitHub issue.
This is the workflow pattern that graduates into the **supervisor multi-agent topology** (§4.1). **Verified implementation** (§8.5).

### 2.5 Evaluator–optimizer
One LLM generates; a second LLM evaluates and gives feedback; loop until the evaluation passes (or a cap is hit).
**Use when:** you have clear evaluation criteria and iterative refinement measurably helps — literary translation, complex search, code that must pass tests.
**Verified implementation** (§8.6).

### 2.6 The autonomous agent loop (when workflows aren't enough)
When you truly can't predefine the path: give the model tools, a goal, and an environment, and let it run the **think → act → observe** loop until it decides it's done or hits a stop condition. The three design decisions that matter, in order: the **environment** (what the agent can see/touch), the **tools** (the agent–computer interface — match formats the model has seen, use absolute paths, minimal escaping, examples in the tool definition), and the **guardrails/stop conditions**. Everything else (caching, parallel tool calls, progress UI) is optimization *after* behavior works. **Verified ReAct loop** in §7.3.

---

## 3. Why one agent isn't enough — and when it actually is

### 3.1 The real reasons to go multi-agent

Not "because it's cool." The legitimate drivers:

1. **Tool overload / context dilution.** An agent choosing from 40 tools is measurably worse than one choosing from 6. Grouping tools and responsibilities into focused specialists raises accuracy on each focused task. This is the single most defensible reason.
2. **Context-window pressure.** Long-horizon tasks blow the window. Splitting work across agents, each holding only its slice of context, keeps every prompt focused. (Anthropic's research system leans on this: sub-agents explore in parallel, each with its own context, and only distilled findings return to the lead.)
3. **Parallel breadth.** Independent lines of inquiry can run concurrently under a coordinator — a genuine latency and coverage win for research/analysis workloads.
4. **Failure isolation.** A misbehaving specialist can be constrained, retried, or swapped without destabilizing the whole system.
5. **Distinct, provably-different task types.** When a single agent's accuracy *plateaus despite prompt iteration* and the tasks are demonstrably different in kind, specialization helps.

### 3.2 When multi-agent is premature complexity (say this out loud to customers)

- **A 2-agent "pipeline."** If two agents run in sequence, a sequential workflow with a terminal reviewer is cheaper and more reliable than a supervisor — adding a supervisor to a 2-agent flow has been observed to *triple* cost for no gain.
- **The task decomposes cleanly ahead of time.** Then it's a *workflow*, not agents.
- **You haven't exhausted single-agent prompt/tool iteration.** Move to multi-agent only after a single agent demonstrably plateaus.
- **You can't yet measure routing accuracy.** If you can't evaluate whether the supervisor routes correctly, you can't operate the system — build the eval first (§10.4).

> **FDE scoping heuristic:** *One agent → add tools → add a workflow → only then add a second agent.* Each step is a real jump in cost and debugging surface. Make the customer earn each one with a failing eval that the added complexity fixes.

---

## 4. Multi-agent topologies

Four production topologies stabilized by 2026. Choose deliberately — the topology is a 12–24 month architectural commitment, not a tuning knob.

### 4.1 Supervisor / orchestrator–workers
A coordinator LLM controls all flow: it decides which specialist to invoke, hands off, receives results, and decides the next move or termination. Workers don't talk to each other — everything routes through the supervisor.

```
              ┌─────────────┐
              │ SUPERVISOR  │◄──── retains control every turn
              └──┬───┬───┬──┘
        ┌────────┘   │   └────────┐
   ┌────▼───┐   ┌────▼───┐   ┌────▼────┐
   │research│   │  code  │   │ writing │   specialists, isolated context
   └────────┘   └────────┘   └─────────┘
```

- **Strengths:** clean separation of concerns; single point of control makes it auditable; the supervisor can chain multiple specialists per task.
- **Weaknesses:** the supervisor is a bottleneck and a cost center (a full LLM call per routing decision).
- **Discipline that keeps it reliable:** (1) the supervisor holds *zero* specialist tools — only "route to X" or "finish"; (2) its prompt explicitly forbids doing the work itself; (3) a route-accuracy evaluator catches violations in CI.
- **Cost tactic:** frontier model for the supervisor (it needs reasoning to route well), cheaper models for the workers (they mostly need reliable tool-calling). This is the sweet spot most teams converge on.

### 4.2 Peer / network
Every agent can hand control to any other agent — no central coordinator. Communication is direct.

```
   ┌────────┐        ┌────────┐
   │agent A │◄──────►│agent B │
   └───┬────┘        └────┬───┘
       │    ┌────────┐    │
       └───►│agent C │◄───┘
            └────────┘
```

- **Strengths:** maximum flexibility; natural for open-ended collaboration.
- **Weaknesses:** at scale it becomes anarchy — hard to predict, hard to debug, easy to loop. For a handful of clearly-scoped agents it's fine; beyond that, prefer a supervisor. **Verified handoff implementation** in §8.7.

### 4.3 Hierarchical (supervisor-of-supervisors)
Supervisors managing teams of workers, coordinated by a top-level supervisor. Each subtree behaves like a supervisor topology; the sub-agents can themselves be whole graphs.
- **Use when:** the problem genuinely has nested team structure (e.g., a "research division" and a "writing division" each with their own workers).
- **Weakness:** overkill for most teams. If you have four specialists, use a flat supervisor, not a hierarchy.

### 4.4 Sequential / pipeline
Agents run in a fixed order, each consuming the previous one's output, often with a terminal reviewer/evaluator.
- **Strengths:** cheapest and most reliable multi-agent shape; deterministic; easy to reason about.
- **Use when:** the stages are known and ordered (extract → transform → validate → summarize). This is what most "multi-agent" systems *should* have been.

### 4.5 Blackboard / shared-scratchpad (collaboration)
All agents read from and write to a shared message scratchpad; everyone sees everyone's steps.
- **Strengths:** full transparency — any agent can see all intermediate work; great for debugging and for tasks where context sharing matters.
- **Weakness:** verbose and token-expensive; often only the final answer from an agent is actually needed downstream, so broadcasting every step wastes context.

### 4.6 Topology cheat-sheet

| Topology | Coordination | Best for | Main risk | Reach for it when |
|---|---|---|---|---|
| Sequential/pipeline | Fixed order | Known, ordered stages | Rigidity | Default multi-agent choice |
| Supervisor | Central orchestrator | Distinct task types, dynamic delegation | Bottleneck + 3× cost | A single agent plateaus on clearly-different tasks |
| Hierarchical | Nested supervisors | Genuinely nested teams | Over-engineering | ≥2 distinct *teams* of specialists |
| Peer/network | Direct handoffs | Small, open-ended collaboration | Anarchy at scale | ≤ a few clearly-scoped agents |
| Blackboard | Shared scratchpad | Transparency-critical work | Token bloat | You need to inspect all intermediate steps |

---

## 5. Coordination mechanics: how agents actually communicate

Topology is the *shape*; coordination is the *plumbing*. Three mechanisms, and you'll mix them.

### 5.1 Shared state (the LangGraph model)
All agents read/write a single typed state object. Concurrent writes to the same field are merged by a **reducer** — a function that says *how* to combine two updates (append to a list, take the last value, sum, etc.). This is what makes parallel fan-out safe: three workers writing to `findings` don't clobber each other; the `operator.add` reducer concatenates them. Without a reducer, concurrent writes to one channel raise an error. This is the #1 source of production incidents in graph-based agents (state conflicts), so get your reducers right.

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    findings: Annotated[list, operator.add]   # reducer: concurrent writes concatenate
    report: str                               # no reducer: single-writer field
```

### 5.2 Message passing
Agents communicate by appending messages to a shared list (the conversation). LangGraph's `add_messages` reducer handles the append-and-dedupe semantics. This is how supervisor and peer topologies pass work and results around. Use `add_messages` for the message channel, not raw `operator.add`, so message IDs and updates are handled correctly.

### 5.3 Handoffs
One agent explicitly transfers control (and context) to another. Two ways it shows up:
- **Tool-based handoff (supervisor):** the supervisor calls a `transfer_to_researcher` tool; the framework routes to that node. This is the modern recommended supervisor style (more control over context than the old `create_supervisor` helper).
- **`Command`-based handoff (peer):** a node returns a `Command(goto="agent_b", update={...})` — it both updates state *and* names the next node in one move. This is how you build network topologies. **Verified in §8.7.**

### 5.4 Private vs. shared channels
Not every agent should see everything (that's the blackboard's weakness). Design the state so that each agent reads only what it needs and writes only distilled results back to the shared channel — keep raw intermediate work in agent-private channels. Anthropic's code-execution-with-MCP approach takes this further: intermediate results stay in the *execution environment* and only what you explicitly log/return enters any model's context — which also keeps sensitive data out of the context window entirely.

---

## 6. The 2026 framework landscape

Every frontier lab now ships an agent framework tuned to its own models, and the protocol layer (MCP for tools, A2A for agent-to-agent) went to the Linux Foundation, so tools and even whole agents are increasingly portable across frameworks. As an FDE you need a working map, not loyalty to one.

### 6.1 The landscape table

| Framework | Orchestration model | State persistence | Model lock-in | Reach for it when |
|---|---|---|---|---|
| **LangGraph** (LangChain) | Directed graph, conditional edges | Built-in checkpointing + time-travel | Agnostic | Stateful, auditable, human-in-the-loop, regulated workflows; "what happens when step 7 fails" needs a first-class answer |
| **LangChain 1.x** | Composable components (LCEL) | Via LangGraph | Agnostic | Building blocks: chains, tools, retrievers, integrations |
| **Deep Agents** (on LangGraph) | Pre-built plan/subagent/filesystem agent | Inherits LangGraph | Agnostic | You want a batteries-included planning agent and can drop to raw LangGraph when it gets weird |
| **OpenAI Agents SDK** | Explicit **handoffs** + guardrails + tracing | Context vars (ephemeral); harness adds resume | 100+ models via compatible APIs, OpenAI-optimized | Triage → specialist → escalation flows; you want the user to "talk to" a specialist directly |
| **CrewAI** | Role-based crews (Researcher/Writer/Reviewer) | Task outputs passed sequentially; built-in memory | Agnostic | Fastest prototype (hours); demos; migrate out when you hit edge cases |
| **Microsoft Agent Framework 1.0** (AutoGen + Semantic Kernel, GA Apr 2026) | Graph + conversational group chat | Conversation history | Azure/.NET optimized, multi-provider | .NET / Azure-native enterprises. (AutoGen itself is now maintenance-only.) |
| **Google ADK** | Hierarchical agent tree; Sequential/Parallel/Loop workflow agents | Session state, pluggable backends | Gemini-optimized, agnostic via LiteLLM | GCP-native, multimodal, A2A cross-team agent discovery |
| **Claude Agent SDK** | One extraordinarily-capable agent + subagents; tool-use chain | Via MCP (external) + context (in-session) | Claude only | Coding agents with deep OS access; safety-first, computer use, native MCP |
| **mcp-agent** (lastmile) | Anthropic's patterns as composable workflows, MCP-native | Temporal-backed durability | Agnostic | You want Anthropic's patterns as code (if/while, not graphs) on a shared protocol |

### 6.2 The tension every framework choice comes down to
**Control vs. simplicity.** LangGraph gives the most control at the cost of boilerplate. Claude Agent SDK / Deep Agents give simplicity at the cost of fine-grained orchestration. OpenAI Agents SDK splits the difference with handoffs. CrewAI optimizes for speed-to-demo. This tension won't resolve — it reflects real, different priorities. Pick for *this* problem's needs (auditability? speed? multimodal? .NET?), not for a favorite.

### 6.3 The protocols — know these cold as an FDE

- **MCP (Model Context Protocol).** Anthropic's open standard (Nov 2024) that collapses the M×N integration problem: each model implements MCP once, each tool implements MCP once. By 2026 it's foundational — 200+ server implementations, native or adapter support in every major framework. A single host can connect to many MCP servers, each exposing one domain (files, DB, an API). **This is the standard way you'll wire enterprise tools to agents.** Watch the failure mode: loading *all* tool definitions and every intermediate result into context bloats tokens and degrades accuracy — hence Anthropic's "code execution with MCP" pattern, where the agent writes code against MCP servers as APIs and only loads the tools it needs.
- **A2A (Agent-to-Agent).** The open standard for agents *discovering and invoking each other* across frameworks (an ADK agent calling a LangGraph agent via a standardized task interface, using "Agent Cards" for discovery). This is what makes cross-vendor multi-agent systems possible.
- **Skills.** Anthropic's reusable capability packages (defined once, reused across Claude.ai, Claude Code, API), opened as a shared format in 2026. They solve prompt drift: instead of re-writing a long prompt each time, a team defines a Skill once. Relevant to multi-agent because a specialist's "method" can be a Skill rather than a brittle mega-prompt.

### 6.4 The FDE default recommendation
For an enterprise deployment you have to *operate and debug* later, **LangGraph is the safe default** — checkpointing, time-travel debugging, human-in-the-loop, and audit trails are first-class, which is exactly what regulated customers need. Prototype in CrewAI if you need a demo *this afternoon*, then migrate. Use the Claude Agent SDK when the job is really "one very capable coding/computer-use agent." And always ask first whether the answer is "no framework, just the API."

---

## 7. LangChain deep dive — the building blocks

LangChain 1.x is the component layer; LangGraph is the orchestration layer above it. You use LangChain for the *pieces* (models, tools, retrievers, prompts) and LangGraph to *wire them into stateful flows*.

### 7.1 Chains (LCEL — LangChain Expression Language)
LCEL composes components with the `|` pipe operator into a runnable. `prompt | model | parser` reads left-to-right: format the prompt, call the model, parse the output. Runnables give you streaming, batching, and async for free. Mental model: **a chain is a fixed, linear workflow** (§2.1) expressed declaratively.

### 7.2 Memory
Two kinds, and enterprises need both:
- **Static / knowledge-base memory:** policies, docs, reference material that rarely changes — this is what RAG retrieves from (§7.4).
- **Persistent / conversational memory:** what the system remembers across sessions and turns. In the LangGraph world this is handled by the **checkpointer** (thread-scoped short-term memory) and the **store** (cross-thread long-term memory) — see §9.

### 7.3 Tools and the ReAct loop
A tool is any function exposed to the model with a name, description, and typed args. `llm.bind_tools([...])` attaches the schemas so the model can emit `tool_calls`. The agent then runs the **think→act→observe** loop. Here is that loop, **verified end-to-end** (mock model emits the tool call so the loop mechanics are what's proven):

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"{city}: 22C sunny"

TOOLS = {"get_weather": get_weather}
# Real: ai = llm.bind_tools(list(TOOLS.values())).invoke(msgs)
# The loop below is identical whether the model is real or mocked.

msgs = [HumanMessage("weather in Bengaluru?")]
for _ in range(5):                       # ReAct loop with a HARD STEP CAP (loop prevention)
    ai = llm.invoke(msgs); msgs.append(ai)
    if not ai.tool_calls:                # no tool requested -> final answer
        break
    for tc in ai.tool_calls:             # execute each requested tool
        obs = TOOLS[tc["name"]].invoke(tc["args"])
        msgs.append(ToolMessage(content=obs, tool_call_id=tc["id"]))
# msgs -> [Human, AI(tool_call), Tool(result), AI(final answer)]
```

**Tool-design rules that separate seniors from juniors:** match the tool format to what the model has seen (absolute paths, minimal JSON escaping), put examples *in the tool definition*, keep each tool's scope tight and well-documented, and remember that once you cross ~20 tools you have a tool-*management* problem (credential rotation, testing tool interactions) more than a tool-*calling* problem — this is where MCP + specialization (§3.1) earns its keep.

### 7.4 Retrievers (RAG in an agentic context)
A retriever turns a query into relevant chunks pulled from a vector store, injected into the prompt so the model answers from your data instead of its parametric memory. In a multi-agent system, RAG is usually *a tool a specialist agent owns* ("search the policy corpus"), not a monolithic step. The FDE nuance: retrieval quality (chunking, embeddings, re-ranking) is where most enterprise "the agent hallucinated" complaints actually originate — the orchestration was fine; the retriever fed it garbage. Instrument retrieval hit-rate as its own metric.

---

## 8. LangGraph deep dive — verified patterns

LangGraph models an agent system as a **state machine / directed graph**: nodes are steps (an LLM call, a tool, an agent), edges are transitions, and a shared typed **state** flows through. Conditional edges are the branches. This maps 1:1 onto the "cognitive architecture as state machine" mental model. Over 70% of production agents in 2026 use a graph/state-machine structure rather than a linear chain, because real business processes branch, get interrupted, and loop.

> Every block in this section was executed on `langgraph 1.2.7`. Imports and APIs are current for the 1.x line.

### 8.1 The core primitives
```python
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages     # message-channel reducer
from langgraph.checkpoint.memory import InMemorySaver # short-term memory / durability
from langgraph.types import Command, interrupt        # handoffs + human-in-the-loop
```
State schema note: `TypedDict` is fine to start; for production, a **Pydantic model with `extra="forbid"`** is the 2026-recommended schema because it validates types and blocks stray fields from polluting state (a nasty, hard-to-locate bug class).

### 8.2 The build → compile → invoke lifecycle
1. Define a **state** schema (with reducers on any concurrently-written field).
2. Add **nodes** (functions `state -> partial_state_update`).
3. Add **edges** — fixed (`add_edge`) or **conditional** (`add_conditional_edges` with a router function).
4. `.compile()` (optionally with a `checkpointer` / `store`).
5. `.invoke()` / `.stream()`.

### 8.3 Routing workflow (§2.2) — verified
```python
class RState(TypedDict):
    query: str; category: str; answer: str

def classify(s):
    q = s["query"].lower()
    return {"category": "billing" if ("refund" in q or "charge" in q) else "tech"}

def billing(s): return {"answer": "Billing handled refund."}
def tech(s):    return {"answer": "Tech reset your device."}
def route(s) -> Literal["billing", "tech"]: return s["category"]

b = StateGraph(RState)
b.add_node("classify", classify); b.add_node("billing", billing); b.add_node("tech", tech)
b.add_edge(START, "classify")
b.add_conditional_edges("classify", route, {"billing": "billing", "tech": "tech"})
b.add_edge("billing", END); b.add_edge("tech", END)
app = b.compile()
app.invoke({"query": "I want a refund for this charge"})   # -> category "billing"
```

### 8.4 Parallel fan-out / fan-in (§2.3) — verified
The reducer is the whole trick: three nodes write `findings` concurrently and the `operator.add` reducer merges them instead of erroring.
```python
class PState(TypedDict):
    topic: str
    findings: Annotated[list, operator.add]   # concurrent-safe merge
    report: str

def research_web(s):  return {"findings": [f"web:{s['topic']}"]}
def research_docs(s): return {"findings": [f"docs:{s['topic']}"]}
def research_db(s):   return {"findings": [f"db:{s['topic']}"]}
def synthesize(s):    return {"report": " + ".join(sorted(s["findings"]))}

b = StateGraph(PState)
for n, f in [("web", research_web), ("docs", research_docs),
             ("db", research_db), ("synth", synthesize)]:
    b.add_node(n, f)
for n in ("web", "docs", "db"):
    b.add_edge(START, n); b.add_edge(n, "synth")   # fan-out then fan-in
b.add_edge("synth", END)
b.compile().invoke({"topic": "gpu", "findings": []})  # findings has all 3, no clobber
```

### 8.5 Supervisor–worker (§4.1) — verified
The supervisor routes; workers report back through shared state; the supervisor decides FINISH. In production the `supervisor` node calls an LLM with **structured output** to pick the next worker — here a deterministic mock stands in for that decision so the *routing control flow* is what's proven.
```python
class SState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    steps: Annotated[list, operator.add]

WORKERS = ["researcher", "coder"]
def supervisor(s):                          # real impl: LLM structured-output routing
    done = set(s["steps"])
    for w in WORKERS:
        if w not in done: return {"next": w}
    return {"next": "FINISH"}

def researcher(s): return {"steps": ["researcher"], "messages": [AIMessage("research done")]}
def coder(s):      return {"steps": ["coder"],      "messages": [AIMessage("code done")]}
def sroute(s):     return END if s["next"] == "FINISH" else s["next"]

b = StateGraph(SState)
b.add_node("supervisor", supervisor); b.add_node("researcher", researcher); b.add_node("coder", coder)
b.add_edge(START, "supervisor")
b.add_conditional_edges("supervisor", sroute,
                        {"researcher": "researcher", "coder": "coder", END: END})
b.add_edge("researcher", "supervisor"); b.add_edge("coder", "supervisor")  # report back
b.compile().invoke({"messages": [HumanMessage("build a scraper")], "next": "", "steps": []})
# -> steps == ["researcher", "coder"], then FINISH
```
The `langgraph-supervisor` package with `create_supervisor()` will do this for you, but LangChain now recommends the **manual tool-based supervisor** for most cases because you get more control over context engineering — and you understand what's under the hood (§1.3).

### 8.6 Evaluator–optimizer loop (§2.5) — verified
```python
class EState(TypedDict):
    draft: str; score: int; rounds: int

def generate(s):
    r = s.get("rounds", 0) + 1
    return {"draft": f"v{r}", "rounds": r}
def evaluate(s):                 # real impl: a grader LLM returns a score + feedback
    return {"score": 3 + 2 * s["rounds"]}          # r1=5, r2=7, r3=9
def gate(s) -> Literal["generate", "done"]:
    return "done" if s["score"] >= 9 or s["rounds"] >= 5 else "generate"  # cap prevents infinite loop

b = StateGraph(EState)
b.add_node("generate", generate); b.add_node("evaluate", evaluate)
b.add_edge(START, "generate"); b.add_edge("generate", "evaluate")
b.add_conditional_edges("evaluate", gate, {"generate": "generate", "done": END})
b.compile().invoke({"draft": "", "score": 0, "rounds": 0})  # -> passes at round 3, score 9
```
**Note the `rounds >= 5` cap.** Never ship a self-referential loop without a hard iteration bound.

### 8.7 Peer handoff via `Command` (§4.2) — verified
`Command` updates state *and* names the next node in one return — this is how you build network topologies without a central router.
```python
class NState(TypedDict):
    messages: Annotated[list, add_messages]
    trace: Annotated[list, operator.add]

def agent_a(s) -> Command[Literal["agent_b", "__end__"]]:
    if "agent_b" not in s["trace"]:
        return Command(goto="agent_b", update={"trace": ["agent_a"], "messages": [AIMessage("A->B")]})
    return Command(goto=END, update={"trace": ["agent_a_final"]})

def agent_b(s) -> Command[Literal["agent_a"]]:
    return Command(goto="agent_a", update={"trace": ["agent_b"], "messages": [AIMessage("B->A")]})

b = StateGraph(NState)
b.add_node("agent_a", agent_a); b.add_node("agent_b", agent_b)
b.add_edge(START, "agent_a")
b.compile().invoke({"messages": [HumanMessage("start")], "trace": []})
# -> trace == ["agent_a", "agent_b", "agent_a_final"]  (A hands to B, B back to A, A finishes)
```

### 8.8 Human-in-the-loop: interrupt / resume (§production-critical) — verified
Over 60% of production agents add human intervention points, and for anything touching customer data or money it's non-negotiable. `interrupt()` pauses the graph and persists state via the checkpointer; you resume with `Command(resume=...)`. Both branches below were executed:
```python
class HState(TypedDict):
    action: str; approved: bool; result: str

def propose(s): return {"action": "DELETE production table"}
def human_gate(s):
    decision = interrupt({"action": s["action"], "question": "Approve? (yes/no)"})  # PAUSES here
    return {"approved": decision == "yes"}
def execute(s): return {"result": "EXECUTED" if s["approved"] else "BLOCKED"}

b = StateGraph(HState)
b.add_node("propose", propose); b.add_node("human_gate", human_gate); b.add_node("execute", execute)
b.add_edge(START, "propose"); b.add_edge("propose", "human_gate")
b.add_edge("human_gate", "execute"); b.add_edge("execute", END)
app = b.compile(checkpointer=InMemorySaver())          # checkpointer REQUIRED for interrupts

cfg = {"configurable": {"thread_id": "t1"}}
first = app.invoke({"action": "", "approved": False, "result": ""}, cfg)
assert "__interrupt__" in first                        # graph paused, awaiting human
final = app.invoke(Command(resume="no"), cfg)          # human denies
assert final["result"] == "BLOCKED"                    # -> destructive action blocked
```

### 8.9 Durable execution / checkpointing (§9 preview) — verified
A `checkpointer` persists every state transition against a `thread_id`. If a run crashes at step 7, it resumes from step 6 — this is the "what happens when step 7 fails" answer that graph frameworks own. Threads are isolated:
```python
class C(TypedDict):
    log: list
def step(s): return {"log": s["log"] + ["ran"]}
b = StateGraph(C); b.add_node("step", step); b.add_edge(START, "step"); b.add_edge("step", END)
app = b.compile(checkpointer=InMemorySaver())
app.invoke({"log": []}, {"configurable": {"thread_id": "alice"}})
app.invoke({"log": []}, {"configurable": {"thread_id": "bob"}})
app.get_state({"configurable": {"thread_id": "alice"}}).values   # -> {"log": ["ran"]}, isolated per thread
```
In production swap `InMemorySaver` for a persistent checkpointer (Postgres/Redis-backed) so durability survives process restarts.

---

## 9. Memory & state architecture

Distinguish four things customers routinely conflate:

| Layer | LangGraph mechanism | Scope | Holds |
|---|---|---|---|
| **Working / short-term** | Graph **state** + **checkpointer** | One thread (conversation) | The in-flight messages and scratchpad |
| **Long-term** | **Store** (`InMemoryStore` → persistent) | Cross-thread, per-user/org | Facts, preferences, learned context across sessions |
| **Semantic knowledge** | Vector store + retriever (RAG) | Global corpus | Policies, docs, reference material |
| **Durable execution** | Checkpointer (Postgres/Redis) | Per thread | Every state transition for crash recovery + time-travel |

Key clarification for customers: **checkpointing is not semantic memory.** LangGraph ships checkpointing (state persistence) out of the box; genuine built-in *semantic* memory is a gap in most frameworks (only a few — CrewAI, Mastra, ADK — ship it), so you usually build long-term memory yourself with the store + a vector DB. Over 60% of production incidents trace to state management, so this table is where you spend your design care.

---

## 10. Production concerns — the part that makes you an FDE, not a demo-builder

### 10.1 Cost
Multi-agent is expensive. Every supervisor turn and every agent handoff is a full model call; a debate/group-chat pattern can burn 20+ calls per interaction. Controls: cheap models for workers + frontier model only for the supervisor; cap the number of turns; keep intermediate results *out* of context (code-execution-with-MCP); prune the message history you forward. Put a **per-run token budget** in state and fail closed when exceeded.

### 10.2 Latency
Each autonomous turn adds latency. Parallelize independent work (fan-out), stream tokens per node so the user sees progress, and prefer a workflow over an agent whenever the path is predictable. For high-volume, provisioned throughput keeps latency consistent.

### 10.3 Failure modes & reliability
- **Error compounding.** An early mistake propagates through every downstream agent — the source of the "17× amplification" figure. Mitigate with per-agent validation gates and a terminal reviewer.
- **Loops.** Peer meshes and evaluator loops can cycle forever. **Always** cap iterations (§8.6) and detect repeated states.
- **State pollution.** Stray fields corrupt downstream nodes — Pydantic `extra="forbid"` (§8.1).
- **Human gates on irreversible actions.** Anything touching customer data, money, or production infra goes through `interrupt()` (§8.8).
- **Durable recovery.** Checkpointing so a crash resumes rather than restarts (§8.9).

### 10.4 Observability & evals (this connects straight to your LLM service layer)
AI systems are black boxes and agents add layers, so beyond structured logging and distributed tracing you need agent-specific observability: trace every LLM call, tool invocation, and state transition so you can **replay the exact decision sequence** that produced a failure. Tooling: **LangSmith** (traces, eval, time-travel replay; free tier ~5,000 traces/month, Plus ~$39/seat with 10,000) or OpenTelemetry-based stacks; framework-native tracing (OpenAI Agents SDK, ADK via Cloud Trace).

The eval that matters most for a supervisor system is **routing accuracy** — score, per task, whether the supervisor picked the right specialist and how many tool calls it took, and fail CI on regressions. For agents generally: build a small labeled trajectory set, grade with an evaluator model + hard checks, and gate deploys on it. *Build the eval before the system* — if you can't measure whether it works, you can't operate it. This is exactly the evals + guardrails + observability that your Project-5 LLM service layer was designed to seed; multi-agent orchestration is a heavy consumer of it.

### 10.5 Security & data governance (ties to your governance playbook)
Enterprise agents face constraints consumer apps don't: sensitive data (financial, health, PII), audit requirements, and hallucination intolerance. Two concrete techniques: keep sensitive intermediate data in the execution environment so it never enters model context (code-execution-with-MCP can auto-tokenize PII before the model sees it), and use MCP's per-server isolation (each server one domain, 1:1 client connection) to contain blast radius. Map these to the EU AI Act risk tiers and your governance reference when the customer is regulated — an FDE who can say "here's how this architecture satisfies your Article-obligations" closes deals faster.

---

## 11. Real-world enterprise use cases (mapped to patterns)

Each is framed the way you'd scope it on a customer call: the pattern, why, and the trap.

1. **Customer-support triage & resolution.** *Routing* → specialist agents (billing/tech/account) each owning scoped tools (read history, update ticket, issue refund), with an `interrupt()` gate before refunds. *Trap:* teams jump to a supervisor mesh; a router + 3 specialists + a human gate is simpler and more reliable. Handoff-style frameworks (OpenAI Agents SDK) fit when the user should "talk to" the specialist directly.
2. **Deep research assistant.** *Supervisor + parallel sub-agents*: a lead planner fans out breadth-first sub-agents (each with isolated context) over web/docs/DB, then synthesizes. This is where multi-agent genuinely beats single-agent. *Trap:* token cost — prune what returns to the lead; only distilled findings, not full transcripts.
3. **Coding agent on a repo.** *Orchestrator–workers* (unknown number of files) + *evaluator–optimizer* (tests are the grader). Verifiable progress (tests pass) is what makes true agency safe here. *Trap:* no test harness = no eval = don't ship it as an agent.
4. **Enterprise "digital assembly line" (2026's highest-value pattern).** Human-guided multi-step workflows where agents run end-to-end processes across systems via **MCP** connectors (Salesforce, Drive, Jira). Usually a *sequential pipeline* with human checkpoints, not a mesh. *Trap:* over-connecting MCP servers bloats context — use code-execution-with-MCP.
5. **Data-import / CRM sync.** A single well-tooled agent writing code against MCP servers, sensitive fields tokenized before hitting context. *This is a "don't use multi-agent" showcase* — one agent, good tools, done.
6. **Content production (marketing/reports).** *Sequential* Researcher → Writer → Reviewer (CrewAI's sweet spot for a fast demo) or an *evaluator–optimizer* loop for quality. *Trap:* role-play abstractions add token overhead; measure whether the extra agents beat one good prompt.
7. **Financial/ops decisions with approvals.** Any pattern, but *mandatory* `interrupt()` human gates on irreversible actions, full trace for audit, and Pydantic-validated state. Governance-first framing wins the room.

---

## 12. Hands-on labs — build these to prove the skill

Progressive, each producing an artifact and a metric. Route every model call through your Project-5 LLM service layer so cost/traces/evals accrue automatically. **Do them in order — each reuses the last.**

**Lab 1 — Workflows before agents (½ day).** Implement all five workflow patterns (§8.3–8.6 give you three) as separate LangGraph graphs against a real model. Deliverable: a `workflows/` module + a README stating *when* to use each. *Proof:* you can explain to a customer why each is a workflow, not an agent.

**Lab 2 — Support triage system (1 day).** Routing + 3 specialist agents (billing/tech/account), each with 2–3 real tools (mock the backends), plus an `interrupt()` gate before any refund. Add a **routing-accuracy eval** over 20 labeled tickets and fail CI below 90%. *Proof:* the eval is the artifact interviewers care about — "how do you know it routes correctly?"

**Lab 3 — Supervisor research system (1–2 days).** Supervisor (frontier model, zero specialist tools, route/finish only) + parallel research workers (cheap model) with isolated context + a synthesizer. Instrument tokens-per-task and specialist selection. *Proof:* a cost table showing the 3× supervisor tax and where it's worth it — the exact analysis a customer will ask for.

**Lab 4 — Add durability & HITL (½ day).** Swap `InMemorySaver` for a Postgres checkpointer; kill the process mid-run and show it resumes from the last checkpoint; add a human approval gate on a destructive action. *Proof:* "what happens when step 7 fails" has a live demo answer.

**Lab 5 — Observability & the eval harness (1 day).** Wire LangSmith (or OTel) tracing; build a trajectory eval set; gate a mock deploy on it; produce a replayable trace of an induced failure. *Proof:* you can debug a multi-agent failure by replaying the decision sequence — the single most FDE-differentiating skill in this doc.

**Lab 6 — MCP-connected assembly line (1–2 days).** Wrap two tools as MCP servers; build a sequential pipeline agent that uses them across a mock enterprise flow (e.g., Drive→CRM sync); demonstrate keeping sensitive fields out of model context. *Proof:* the "digital assembly line" — 2026's highest-value enterprise pattern — with a governance story attached.

**Stretch — framework bake-off.** Rebuild Lab 2 in CrewAI and the OpenAI Agents SDK. Write a one-page "when I'd pick each" grounded in *your own* build friction, not a blog. *Proof:* the framework-choice conversation, from experience.

---

## 13. FDE deployment playbook & interview talking points

**The scoping conversation (in order):**
1. "What's the task, and can we *predict the path*?" → if yes, it's a **workflow**, not agents.
2. "Can a single agent with the right tools do it?" → try that first; iterate prompts/tools.
3. "Where does a single agent *plateau*, and are the failing cases *distinct in kind*?" → only now consider specialization.
4. "Can we *measure* whether the added structure helps?" → build the eval first; no eval, no multi-agent.
5. "What's irreversible here?" → human gates. "What's sensitive?" → keep it out of context; MCP isolation; map to their governance regime.

**Interview soundbites (each defensible from this doc):**
- *"Most production 'agent' systems should be a workflow or one well-tooled agent; multi-agent is a last resort you have to earn with a failing eval."*
- *"A supervisor costs ~3× the tokens because every routing decision is a full LLM call — so I put a frontier model on the supervisor and cheap models on the workers."*
- *"The topology is a 12–24 month commitment; I pick sequential by default, supervisor when a single agent plateaus on distinct task types, and peer only for a few clearly-scoped agents."*
- *"The eval I care about most for a supervisor is routing accuracy, gated in CI — if I can't measure routing, I can't operate the system."*
- *"State management is the #1 production incident source, so reducers and Pydantic-validated state come before any cleverness."*
- *"MCP collapses the M×N tool-integration problem; A2A does the same for agent-to-agent — together they're why cross-framework multi-agent is even possible in 2026."*

**Red flags to name in a design review:** a supervisor wrapping only two agents; any loop without an iteration cap; concurrent writes to a reducer-less field; irreversible actions with no human gate; a multi-agent system with no routing eval; all tool defs + intermediate results dumped into one context window.

---

## 14. Recognition index (fast lookup)

| You see / hear… | Reach for… | Section |
|---|---|---|
| Distinct input categories | Routing workflow | §2.2, §8.3 |
| Independent subtasks, need speed | Parallelization (fan-out + reducer) | §2.3, §8.4 |
| Can't predict subtasks | Orchestrator–workers → Supervisor | §2.4, §8.5 |
| Clear grader + iterative gains | Evaluator–optimizer (with cap) | §2.5, §8.6 |
| Path unknowable, progress verifiable | Autonomous agent + ReAct loop | §2.6, §7.3 |
| Single agent plateaus, tasks differ in kind | Specialize → multi-agent | §3 |
| Two agents in sequence | Sequential pipeline (NOT a supervisor) | §4.4 |
| A few agents, open-ended collab | Peer/network via `Command` | §4.2, §8.7 |
| Irreversible / sensitive action | `interrupt()` human gate | §8.8 |
| "What if it crashes at step 7?" | Checkpointer / durable execution | §8.9 |
| Concurrent writes to one field | Reducer (`operator.add`, `add_messages`) | §5.1 |
| Stray fields corrupting state | Pydantic `extra="forbid"` | §8.1 |
| Enterprise tool integration | MCP servers (isolated, per-domain) | §6.3 |
| Cross-framework agent calls | A2A protocol | §6.3 |
| "How do you know it works?" | Routing-accuracy / trajectory evals | §10.4 |
| Regulated customer | Context-isolation + MCP + governance mapping | §10.5 |
| Need a demo this afternoon | CrewAI (then migrate to LangGraph) | §6.1 |
| Auditable, stateful, HITL production | LangGraph (the default) | §6.4 |
| One very capable coding/computer-use agent | Claude Agent SDK | §6.1 |

---

### Appendix: how this doc connects to the rest of the library
- **Project 5 / LLM service layer** — the substrate every agent call routes through; this doc is a heavy consumer of its observability, cost, eval, and guardrail hooks.
- **Governance reference + EU AI Act guide** — §10.5's context-isolation and MCP patterns are your *technical* answer to their *regulatory* obligations; pair them on regulated deals.
- **System design material** — topologies (§4) are distributed-systems tradeoffs (bottlenecks, isolation, coordination) in a new coat; reuse that reasoning.
- **Decision-Driven Engineer guide** — §3 and §13's "earn each jump in complexity" is the judgment layer applied to agent architecture.

*All code verified on langgraph 1.2.7 / langchain 1.3.11 / langchain-core 1.4.8, July 2026.*
