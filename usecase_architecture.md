# Real-Time AI Systems — Design, Architecture, Trade-offs & Scaling

> **Volume 2** of the MCP / AI Workflows reference. Volume 1 covered the protocol and patterns; this volume covers *systems* — six production use cases, each with a full architecture, the trade-offs you'll actually argue about in design review, and a scaling playbook from 10 req/s to 10,000 req/s.
>
> Framing: every use case is written the way a Forward Deployed Engineer would present it to a customer — problem, latency budget, architecture, decision record, failure modes, scale path. BFSI/regulated-industry constraints (audit, data residency, human accountability) are treated as first-class requirements, not afterthoughts.

---

## Table of Contents

0. [The Universal Latency Budget](#0-the-universal-latency-budget)
1. [UC-1: Sub-Second Transaction Fraud Screening](#uc-1)
2. [UC-2: Sub-500ms RAG Support Copilot](#uc-2)
3. [UC-3: Agentic Loan Underwriting at Volume (HITL at Scale)](#uc-3)
4. [UC-4: Real-Time Document Intelligence (KYC/Onboarding)](#uc-4)
5. [UC-5: Enterprise MCP Gateway — Scaling the Tool Layer](#uc-5)
6. [UC-6: Streaming Risk Monitoring & Event-Driven Collections](#uc-6)
7. [Cross-Cutting Strategies: Cost, Observability, Failure](#7-cross-cutting)
8. [Decision Cheat Sheet](#8-decision-cheat-sheet)

---

<a name="0-the-universal-latency-budget"></a>
## 0. The Universal Latency Budget

Every real-time AI design starts by writing down the budget *before* choosing components. LLM inference is the single largest and most variable line item, so the whole discipline of real-time AI architecture is: **spend the model where it earns its latency, and engineer everything else to be nearly free.**

A reference budget for a 500ms end-to-end target:

```
  ┌────────────────────────── 500 ms end-to-end ──────────────────────────┐
  │ network+TLS │ auth │ retrieval │  rerank  │  LLM TTFT  │ stream+post  │
  │   ~30 ms    │ ~5ms │  ~40 ms   │  ~30 ms  │  ~200 ms   │   ~150 ms    │
  └─────────────┴──────┴───────────┴──────────┴────────────┴──────────────┘
                          ▲ everything left of the LLM must be < 100 ms
```

Three consequences that shape every architecture below:

1. **Tier the intelligence.** Deterministic rules (µs) → small/fast model (tens of ms) → frontier model (hundreds of ms). Route upward only on ambiguity. In practice 70–95% of traffic never reaches the expensive tier.
2. **Never serialize what can parallelize.** Retrieval, feature lookup, and policy checks fan out concurrently while the request is still being authenticated.
3. **Stream everything user-facing.** Perceived latency = time-to-first-token, not time-to-last-token. A 3s answer that starts in 250ms feels fast; a 1.5s answer that arrives whole feels slow.

---

<a name="uc-1"></a>
## UC-1: Sub-Second Transaction Fraud Screening

### Problem

A payments switch calls you synchronously on every transaction. You must return `allow | hold | step_up` within a hard SLA (commonly **150–300ms p99** — card networks time out beyond that). Miss the SLA and the transaction defaults to allow (fraud loss) or decline (revenue loss + customer rage). Volume: 2k–50k TPS with 10× festival-season spikes.

### Architecture

```
 Payment switch ──POST /score──►┌──────────────────────────────────────────────┐
        (hard 300ms timeout)    │              SCORING PLANE                   │
                                │                                              │
                                │  ┌─────────┐   ┌──────────────────────────┐  │
                                │  │ TIER 0  │   │ TIER 1: online ML model  │  │
                                │  │ rules   │──►│ (GBM/NN, <10ms, local)   │  │
                                │  │ <1 ms   │   └───────────┬──────────────┘  │
                                │  └─────────┘         gray zone only (~3-8%)  │
                                │        ▲                   ▼                 │
                                │        │        ┌──────────────────────────┐ │
                                │  ┌─────┴─────┐  │ TIER 2: LLM adjudicator  │ │
                                │  │ FEATURE   │  │ small model, 60-120ms,   │ │
                                │  │ STORE     │  │ strict token budget,     │ │
                                │  │ Redis     │  │ HARD deadline + fallback │ │
                                │  │ <5ms read │  └──────────────────────────┘ │
                                └──┴───────────┴───────────────────────────────┘
                                         ▲                    │ decision + trace
        Kafka: txn events ──► streaming  │                    ▼
        (velocity, device,    feature ───┘        ┌───────────────────────────┐
        geo aggregates)       pipeline            │ ASYNC PLANE (post-decision)│
        (Flink/Spark)                             │ frontier-LLM deep review,  │
                                                  │ case creation, SAR draft,  │
                                                  │ analyst queue              │
                                                  └───────────────────────────┘
```

### Key design decisions

**Decision 1 — The LLM is Tier 2, never Tier 0.** Rules catch the obvious (sanctioned account, impossible velocity). The ML model scores everything in <10ms. The LLM only adjudicates the gray zone where the model score is between thresholds (e.g., 0.55–0.80) — typically 3–8% of traffic. This is what makes the SLA survivable: at 10k TPS, the LLM sees 300–800 TPS, not 10k.

**Decision 2 — Hard deadline with graceful degradation.** The Tier-2 call runs under an absolute deadline (e.g., 120ms). If it doesn't return, you fall back to the ML score plus a conservative policy (`step_up` instead of `allow`). *The system's decision quality degrades under load; its availability never does.*

```python
async def adjudicate(txn, features, ml_score) -> Decision:
    try:
        verdict = await asyncio.wait_for(
            small_llm_score(txn, features), timeout=0.120)   # hard deadline
        return verdict
    except asyncio.TimeoutError:
        metrics.incr("tier2.deadline_miss")
        # Fail SAFE, not fail OPEN: ambiguous + no adjudication → step-up auth
        return Decision.STEP_UP if ml_score > 0.65 else Decision.ALLOW
```

**Decision 3 — Features are precomputed, never computed in-path.** Velocity counters, device reputation, and geo aggregates are maintained by a streaming pipeline (Flink over Kafka) and materialized into Redis. The scoring path does *reads only* — a `MGET` of ~20 keys in <5ms. Computing a 1-hour velocity window at request time would blow the entire budget.

**Decision 4 — Deep analysis is asynchronous.** The frontier-model work (narrative fraud analysis, SAR drafting, linking to prior cases) happens *after* the decision, off the critical path, feeding an analyst queue. Real-time answers the "now" question; async answers the "why" question.

### Trade-offs

| Choice | You gain | You pay |
|---|---|---|
| Tiered scoring vs LLM-on-everything | SLA feasibility, ~20× lower inference cost | Two models to train/monitor; threshold tuning is a permanent workstream |
| Hard deadline + fallback | Bounded p99, no cascading timeouts | Some gray-zone txns get the cruder decision; must monitor deadline-miss rate as a first-class SLO |
| Fail-safe (step-up) vs fail-open (allow) | Fraud loss bounded during degradation | Friction/abandonment cost during incidents — a business decision, put it in writing with the risk team |
| Precomputed features | <5ms feature access | Feature freshness lag (seconds); dual-write complexity; Redis becomes tier-0 infra needing HA |
| Small LLM in-path vs frontier | 60–120ms vs 400ms+ | Weaker reasoning; mitigate with tight prompts + distillation from the async frontier tier |

### Scaling strategy

- **10× spike handling:** Tier-1/Tier-2 are stateless — horizontal pod autoscaling on concurrency, not CPU. Pre-scale before known events (festivals, sale days); LLM capacity cannot cold-start in seconds.
- **Regional cells:** shard by issuer/region into independent cells (own Redis, own model replicas). A cell failure degrades one region, not the platform. Also satisfies data-residency (EU txns scored in-region — relevant under GDPR/EU AI Act, where fraud scoring is a regulated high-risk profile).
- **Adaptive tiering under load:** when Tier-2 queue depth rises, *narrow the gray zone* (raise the lower threshold) so fewer transactions qualify for LLM adjudication. You trade decision quality for latency automatically, and log that you did.

---

<a name="uc-2"></a>
## UC-2: Sub-500ms RAG Support Copilot

### Problem

Customer-facing chat where an answer grounded in your knowledge base must *start rendering* in ~500ms and read fluently thereafter. The naive RAG chain (embed → search → rerank → prompt → generate) serialized end-to-end lands at 2–4s. The architecture below is about deleting and overlapping stages.

### Architecture

```
 user query ──►┌───────────────────────────────────────────────────────────────┐
               │                    QUERY PLANE                                │
               │                                                               │
               │  ┌──────────────┐    launched in PARALLEL at t=0:             │
               │  │ SEMANTIC     │    ├─► embed query          (~15ms, local)  │
               │  │ CACHE        │    ├─► BM25/keyword search  (~10ms)         │
               │  │ hit? ──► ████│    ├─► vector ANN search    (~25ms, HNSW)   │
               │  │ (30-60% hits │    └─► user-context fetch   (~10ms)         │
               │  │  answer in   │                    │                        │
               │  │  <50ms)      │            hybrid merge + rerank            │
               │  └──────────────┘            (cross-encoder, top-40→8, ~30ms) │
               │         │ miss                       │                        │
               │         ▼                            ▼                        │
               │  ┌────────────────────────────────────────────────────────┐   │
               │  │ GENERATION: streaming LLM, TTFT ~200ms                 │   │
               │  │ prompt = system(cached) + docs(compact) + query        │   │
               │  │ prompt-cache the static prefix → big TTFT cut on reuse │   │
               │  └────────────────────────────────────────────────────────┘   │
               └───────────────────────────────────────────────────────────────┘
   INGESTION (offline, decoupled): chunk → embed → index (HNSW) → cache-warm FAQs
```

### Key design decisions

**Decision 1 — Semantic cache first.** Support traffic is brutally repetitive: 30–60% of queries are paraphrases of a known set. Embed the query, ANN-search a cache of (query-embedding → vetted answer), and if similarity > threshold, return the cached answer in <50ms — no LLM at all. The threshold is a precision knob: too loose and users get subtly-wrong cached answers; start conservative (≥0.95 cosine) and relax with measurement. Invalidate cache entries when their source documents change (store doc-version pointers with each entry).

**Decision 2 — Parallel fan-out, hybrid retrieval.** BM25 and vector search launch simultaneously; results merge via reciprocal-rank fusion, then a cross-encoder reranks top-40 to top-8. Hybrid matters in enterprise corpora because exact identifiers ("Form 15G", error codes, policy numbers) are where pure vector search embarrassingly fails.

**Decision 3 — Spend tokens like latency.** Prompt size drives both TTFT and cost. Compact the 8 chunks (strip boilerplate, dedupe), cap context tokens, and use provider prompt-caching on the static system prefix. On cache-hit turns this removes a large fraction of TTFT.

**Decision 4 — Stream, and pre-commit the skeleton.** First tokens render at ~250–300ms. For structured answers, stream the opening sentence while the citation block is still being assembled.

### Trade-offs

| Choice | You gain | You pay |
|---|---|---|
| Semantic cache | Huge p50 win, big cost cut | Staleness risk; cache-invalidation discipline; wrong-answer blast radius if threshold too loose |
| Hybrid + rerank vs vector-only | Precision on exact-match queries | +30–40ms and a reranker to host; worth it almost always in enterprise |
| Small context (top-8) vs stuffing top-40 | Fast TTFT, lower cost | Occasional missed context; mitigate by letting the model ask a follow-up rather than pre-loading everything |
| Self-hosted embedder/reranker | ~15ms embeds vs 50–100ms API RTT | GPU ops burden; a real cost line at low volume, a bargain at high volume |
| Streaming UX | Perceived latency ≈ TTFT | Harder guardrailing — you're moderating a stream; need output-filter-on-the-fly or a small pre-generation check |

### Scaling strategy

- **Read path is embarrassingly parallel** — replicate the retrieval stack; ANN indexes are read-mostly and replicate cheaply. Shard the vector index by tenant (also your data-isolation story).
- **Ingestion decoupled** via queue; index rebuilds happen blue/green (build new HNSW, atomically swap alias) so reads never see a half-built index.
- **Cache hierarchies at scale:** L1 exact-string LRU in-process, L2 semantic cache in Redis, L3 prompt-cache at the provider. Each layer strips load from the next.
- **Multi-region:** replicate indexes and caches per region; pin generation to in-region model endpoints for residency + RTT.

---

<a name="uc-3"></a>
## UC-3: Agentic Loan Underwriting at Volume — HITL at Scale

### Problem

Volume 1 showed one application flowing through route → assess → human gate → disburse. The real problem is **10,000 applications/day with 15 credit officers**. Human review is now your scarcest resource and slowest "component" (minutes–hours, not ms). The architecture question flips: not "how fast is the model" but "how do we spend limited human attention where it changes outcomes, without the queue melting down."

### Architecture

```
 applications ──► Kafka ──►┌──────────────────────────────────────────────────┐
 (10k/day,                 │            DURABLE WORKFLOW ENGINE               │
  bursty)                  │            (Temporal / equivalent)               │
                           │                                                  │
                           │  route ──► agent-assess ──► confidence-scored    │
                           │  (rules)   (LLM + MCP       recommendation       │
                           │   ~70%      tools: bureau,       │               │
                           │   auto)     docs, ledger)        ▼               │
                           │                        ┌──────────────────────┐  │
                           │                        │  TRIAGE & ROUTING    │  │
                           │                        │  priority = f(amount,│  │
                           │                        │  SLA-age, confidence,│  │
                           │                        │  officer skill match)│  │
                           │                        └──────────┬───────────┘  │
                           └───────────────────────────────────┼──────────────┘
                                                               ▼
                     ┌──────────────────────────────────────────────────────┐
                     │  REVIEW PLANE (the human "service")                  │
                     │  officer console: full context pre-assembled —       │
                     │  agent rationale, evidence links, ONE-CLICK          │
                     │  approve / decline / send-back-with-question         │
                     │  (send-back resumes the SAME workflow run)           │
                     └──────────────────────────────────────────────────────┘
                        every decision (auto + human) → immutable audit log
                        sampled auto-approvals → QA re-review loop
```

### Key design decisions

**Decision 1 — Durable execution, not request/response.** An underwriting case lives hours-to-days and must survive deploys, crashes, and an officer going to lunch. A durable workflow engine (Temporal-style) gives you: state that persists across restarts, a `wait_for_signal("officer_decision")` that can block for days without holding a thread, automatic retries with idempotency, and a complete event history per case — which *is* your audit trail. Hand-rolling this with queues + cron + a `status` column is the classic mistake; you end up re-implementing a worse Temporal.

```python
# Sketch: the human gate as a durable signal wait (Temporal-style pseudocode)
@workflow.defn
class Underwriting:
    @workflow.run
    async def run(self, app: Application):
        decision = await workflow.execute_activity(route, app, timeout=30s)
        if decision == NEEDS_HUMAN:
            assessment = await workflow.execute_activity(agent_assess, app,
                                                         retry_policy=exp_backoff)
            await workflow.execute_activity(enqueue_for_review, app, assessment)
            verdict = await workflow.wait_condition(          # blocks for days,
                lambda: self.officer_verdict is not None,     # costs nothing,
                timeout=timedelta(hours=48))                  # survives deploys
            if verdict.action == "send_back":
                assessment = await workflow.execute_activity( # loop, same run,
                    agent_reassess, app, verdict.question)    # full history kept
        await workflow.execute_activity(disburse, app)        # idempotency key!
```

**Decision 2 — Confidence-calibrated triage.** The agent outputs a recommendation *and* a confidence signal (self-reported + an independent evaluator model — never trust self-report alone). Triage sorts the human queue by `expected-value-of-human-attention`: high-amount × low-confidence × approaching-SLA first. Low-amount high-confidence cases get lighter review or sampled QA instead of full review. This is the difference between 15 officers drowning and 15 officers covering 10k/day.

**Decision 3 — Design the reviewer console as a product.** HITL fails in the UI more than in the backend. If an officer must open five systems to verify what the agent said, throughput dies and they start rubber-stamping. Pre-assemble everything: the rationale, *clickable evidence* (the actual bureau response, the actual statement page), diffs vs policy, and one-click actions. Measure officer decision time and overturn rate — overturn rate is your live model-quality metric.

**Decision 4 — Guard against automation complacency.** When the agent is right 95% of the time, humans stop looking. Countermeasures: inject known-bad seeded cases into the queue (measure catch rate), randomize presentation order so officers can't pattern-match "agent said approve," and require a typed reason on approvals above a threshold.

### Trade-offs

| Choice | You gain | You pay |
|---|---|---|
| Durable engine vs queues+cron | Correct long-running state, audit-grade history, painless retries | New infra to operate; workflow-versioning discipline for in-flight cases during deploys |
| Confidence triage vs FIFO | Human attention lands where risk is | Calibration is hard; a miscalibrated model quietly routes bad cases to light review — monitor overturn-rate by confidence bucket |
| Sampled QA on auto-approvals | 70% of volume needs no human | Regulatory acceptability varies — some regimes require documented human accountability for credit decisions (and EU AI Act treats creditworthiness as high-risk); the sampling rate is a compliance conversation, not just an engineering one |
| Send-back loops (agent ⇄ officer) | Fewer wrong declines; officer leverage | Unbounded loops possible — cap iterations, then force a human-only path |

### Scaling strategy

- **Scale the agent tier, not the promise.** Agent-assess is stateless activity work — autoscale workers on queue depth. The workflow engine shards by case ID.
- **Scale humans by specialization:** route by product/geography skill tags; measure per-officer throughput and route accordingly (the "skill match" term in triage).
- **Backpressure end-to-end:** if the review queue exceeds SLA capacity, *upstream tightens* — the auto-approve band narrows is wrong (risk ↑); instead, intake slows or SLAs re-tier by amount. Backpressure must be a policy decision surfaced to the business, not silent queue growth.
- **10× growth path:** the bottleneck is officers, so growth = raising the auto-decision share safely — which means investing in the QA loop and evaluator models, not more GPUs.

---

<a name="uc-4"></a>
## UC-4: Real-Time Document Intelligence (KYC / Onboarding)

### Problem

A customer uploads a document mid-onboarding (Aadhaar/passport/bank statement/salary slip). You must classify, extract, verify, and fraud-check it while they wait — target **< 8–10s perceived**, with progress feedback — because every extra onboarding second measurably drops conversion. Documents are adversarial input: wrong files, photos of screens, and deliberate forgeries.

### Architecture

```
 upload ──► object store ──► event ──►┌─────────────────────────────────────────┐
    │  (presigned PUT,                │           PIPELINE (fan-out)            │
    │   client-side                   │                                         │
    ▼   compression)                  │  stage 1: classify  (small VLM, ~300ms) │
 UI shows staged progress:            │      │ "what document is this?"         │
 "reading… extracting…                │      ▼                                  │
  verifying…"                         │  stage 2: PARALLEL                      │
 (perceived-latency design)           │   ├─► field extraction (VLM, 1-3s)      │
                                      │   ├─► forgery signals  (CV model: ELA,  │
                                      │   │    font/metadata anomalies, ~500ms) │
                                      │   └─► liveness/quality checks           │
                                      │      ▼                                  │
                                      │  stage 3: cross-verification            │
                                      │   extracted fields ⇄ application data   │
                                      │   ⇄ external registries (via MCP tools) │
                                      │      ▼                                  │
                                      │  verdict: pass / retry-with-guidance /  │
                                      │           manual-review queue (UC-3)    │
                                      └─────────────────────────────────────────┘
```

### Key design decisions

**Decision 1 — VLM-first with structured outputs, template-OCR as contrast.** Modern vision-language models with a strict JSON schema (field name → value + per-field confidence) replace brittle template-based OCR for the long tail of formats — critical in India where salary slips and bank statements have thousands of layouts. Force structured output; never parse freeform prose. Keep a rules/regex validation layer *after* extraction (checksums on PAN format, IFSC validity, date sanity) — the model extracts, deterministic code validates.

**Decision 2 — Instant-feedback tier before the heavy tier.** A cheap classifier runs in ~300ms and immediately tells the user "this looks like the wrong document / too blurry — retake" *before* the 3s extraction runs. Fast failure is a conversion feature: the user fixes the photo while still engaged.

**Decision 3 — Perceived latency is a UI contract.** Staged progress ("Reading document… Extracting details… Verifying…") with real stage transitions makes 8s feel acceptable; a spinner makes 4s feel broken. Pre-fill the form fields *as they extract* (stream per-field results) so the user reviews instead of waits.

**Decision 4 — Route by confidence, not pass/fail.** Per-field confidence drives three outcomes: high → auto-accept; medium → highlight for user confirmation ("we read your account number as X — correct?"); low or forgery-flagged → manual review queue (which is exactly UC-3's review plane, reused).

### Trade-offs

| Choice | You gain | You pay |
|---|---|---|
| VLM extraction vs template OCR | Handles unseen layouts; one model, not 500 templates | Higher per-doc cost; hallucination risk on low-quality scans — hence the deterministic validation layer and per-field confidence |
| Sync-while-waiting vs fully async | Conversion (user completes in one session) | You've put a 1–3s GPU workload on an interactive path; need reserved capacity + queue-jump for interactive jobs over batch |
| Client-side compression before upload | Big upload-time win on mobile networks | Compression artifacts can hurt forgery detection — keep original for the CV pass, compressed for extraction |
| Separate CV forgery models vs "ask the VLM" | Purpose-built detectors materially outperform on ELA/metadata forensics | Two model families to maintain; fusion logic for conflicting signals |

### Scaling strategy

- **GPU pool with priority classes:** interactive extraction preempts batch re-processing. Autoscale on queue wait time (the user-visible metric), not utilization.
- **Batch the batch:** nightly re-verification and model-upgrade re-runs go through the same pipeline at low priority with aggressive batching (vLLM continuous batching / provider batch APIs) at a fraction of interactive cost.
- **Residency:** documents are the most sensitive artifact you hold. Region-pin the object store and inference; presigned URLs are short-lived; extraction outputs (not images) flow to downstream systems. Under DPDP/GDPR-style regimes, minimize what leaves the document plane.

---

<a name="uc-5"></a>
## UC-5: Enterprise MCP Gateway — Scaling the Tool Layer Itself

### Problem

Volume 1 built individual MCP servers. At enterprise scale you have **40 teams shipping 200 MCP servers to 5,000 agent instances**, and the questions change: who may call what, with whose identity, under which policy, with what audit trail — and how does the *protocol layer* itself scale? This is the layer where agent-trust products live, and it maps directly onto the enterprise-readiness gaps the MCP roadmap itself acknowledges (audit trails, SSO-integrated auth, gateway behavior).

### Architecture

```
   agent runtimes (hosts) ──────────────┐
   Claude / IDEs / custom agents        │  single logical endpoint, SSO token
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                      MCP GATEWAY                             │
        │                                                              │
        │  authn: OAuth2/OIDC, agent identity ≠ user identity          │
        │        (on-behalf-of chain: user → agent → tool)             │
        │  authz: policy engine (OPA/Cedar-style)                      │
        │        "agent A, acting for user U, may call tool T          │
        │         with args matching P, at rate R"                     │
        │  ── virtual servers: remix/subset tools per team/agent ──    │
        │  ── schema firewall: validate args & RESULTS, strip          │
        │      suspicious instructions from tool outputs               │
        │      (prompt-injection choke point)                          │
        │  ── quotas, rate limits, circuit breakers per backend ──     │
        │  ── audit: every call → immutable log w/ trace context ──    │
        │      (W3C traceparent propagates via _meta per SEP-414)      │
        └───────┬──────────────┬──────────────┬────────────────────────┘
                ▼              ▼              ▼
        team A's servers   team B's ...   SaaS/vendor MCP servers
        (registry-discovered, health-checked, version-pinned)
```

### Key design decisions

**Decision 1 — Stateless data plane; the 2026 spec direction is your friend.** The stable (2025-11-25) protocol is session-stateful, which fights load balancers — sticky sessions pin traffic and break autoscaling. Strategy today: run Streamable HTTP servers in **stateless mode** wherever possible (no session ID, JSON responses), keep any genuinely needed state (task handles, subscriptions) in Redis behind the gateway, and treat sticky-session servers as legacy to be migrated. The 2026-07-28 revision removes the protocol-level session precisely so round-robin load balancing works; architect for that now and the migration is a config change, not a rewrite.

**Decision 2 — Identity is a chain, not a token.** The dangerous anti-pattern is the gateway holding one god-credential per backend ("confused deputy"). Correct model: user's SSO identity → delegated to agent (scoped, expiring) → exchanged at the gateway for a backend-specific token carrying *both* identities. Every audit line then answers "which human is accountable for this tool call" — the question a BaFin/EBA-style auditor actually asks.

**Decision 3 — The gateway is your prompt-injection choke point.** Tool *results* are untrusted input to the model. The gateway is the one place every result passes through: scan/strip instruction-shaped content from results, enforce that results conform to the declared output schema, and flag tools whose descriptions changed since security review (poisoned-tool / rug-pull detection — pin tool definitions by hash, alert on drift).

**Decision 4 — Virtual servers over raw exposure.** Don't expose 200 servers × 15 tools to every agent — models degrade badly when choosing among thousands of tools, and it's a needless attack surface. The gateway composes *virtual servers*: curated tool subsets per use case ("collections-agent gets these 9 tools"), with tightened descriptions. Smaller tool menus are simultaneously a quality, cost (fewer schema tokens per request), and security win.

### Trade-offs

| Choice | You gain | You pay |
|---|---|---|
| Central gateway vs direct agent→server | One place for auth/policy/audit/injection defense | The gateway is a SPOF and latency tax (+5–15ms/call) — run it as a horizontally-scaled cell per region, not a singleton |
| Stateless-first servers | Trivial horizontal scaling, LB-friendly, aligned with 2026 spec | Features needing server-push (subscriptions, progress) need a state store or degrade to polling |
| Token exchange per call | True least-privilege, human-attributable audit | IdP becomes hot-path infra; cache exchanged tokens with short TTLs |
| Result-scanning firewall | Mitigates the top real-world MCP attack class | False positives can mangle legitimate results; needs allowlists and per-tool tuning; it's mitigation, not immunity |
| Central registry + version pinning | No shadow tools; reviewed supply chain | Process friction for teams; make the paved road faster than the workaround or teams will bypass it |

### Scaling strategy

- **Cells, again:** gateway cells per region/BU, each fronting its local server fleet; a global registry, local data planes.
- **Connection economics:** thousands of agents × persistent SSE streams is a file-descriptor problem. Prefer JSON-response mode for request/response tools; reserve streams for genuinely streaming tools.
- **Hot-tool caching:** idempotent, read-only tools (annotated `readOnlyHint`) can be response-cached at the gateway keyed on (tool, args, identity-scope) — often 30–50% of agent tool traffic is repeated reads.
- **Blast-radius drills:** a slow backend server must not stall agents — per-backend circuit breakers with tool-level `unavailable` errors the model can reason about ("that tool is down, proceed without it") beat hanging calls.

---

<a name="uc-6"></a>
## UC-6: Streaming Risk Monitoring & Event-Driven Collections

### Problem

A lending book emits a continuous event stream — repayments, missed EMIs, collateral price ticks (for gold-backed lending this is the heartbeat: LTV moves with the gold price in real time), bureau alerts. You need portfolio state that is *current in seconds*, automated first-line actions, and escalation that lands in the human plane (UC-3) with full context. This is "AI on a stream," not "AI on a request."

### Architecture

```
  gold price feed ─┐
  repayment events ─┤                     ┌──► materialized views (Redis/DB):
  bureau webhooks ──┼──► Kafka topics ───►│    per-loan LTV, DPD, risk tier
  ledger CDC ───────┘    (partitioned     │    (Flink: stateful stream jobs)
                          by loan_id)     │
                                          ├──► CEP / rule layer  (ms):
                                          │    "LTV > 85%" → margin-call event
                                          │    "3rd missed EMI + no contact"
                                          │            │
                                          │            ▼
                                          └──► AGENTIC ACTION LAYER
                                               ├─ tier A: templated notification
                                               │   (deterministic, auto-send)
                                               ├─ tier B: LLM-personalized
                                               │   outreach draft — compliance-
                                               │   checked, then auto or queued
                                               └─ tier C: escalation case →
                                                   UC-3 review plane w/ agent-
                                                   assembled dossier (MCP tools)
                     exactly-once concerns: idempotency keys on every action;
                     the stream WILL redeliver — actions must tolerate replay
```

### Key design decisions

**Decision 1 — Stream processing computes state; AI decides on state.** Flink-style jobs maintain per-loan aggregates (LTV, DPD, contact history) with event-time semantics. The AI layer never scans the book; it reacts to *derived events* ("loan X crossed tier boundary") with the materialized view as context. This separation keeps the expensive layer's input rate proportional to *changes*, not to portfolio size.

**Decision 2 — Graduated autonomy by consequence.** Tier A (reminder SMS) is deterministic and auto-fires. Tier B (personalized restructuring offer) is LLM-drafted but passes a compliance-check model + rules (regulated communication: no harassment-pattern language, mandated disclosures, contact-hour windows) before sending — and above thresholds, queues for human sign-off. Tier C (legal notice, collateral liquidation) is *never* autonomous — the agent's job is assembling the dossier, the human owns the act. Same principle as UC-3's gate, applied on a stream.

**Decision 3 — Idempotency is existential.** Kafka redelivers; workflows retry. Every side-effecting action carries a deterministic idempotency key (`loan_id + event_type + window`). The difference between "sent one margin-call notice" and "sent nine" is a customer-harm incident and a regulator conversation.

**Decision 4 — Storm mode.** When gold drops 4% in an hour, thousands of loans breach LTV simultaneously. Naive design: thousands of individual LLM calls + notification storms. Correct design: the CEP layer detects the *cohort* event, switches to batch handling (one LLM call drafting per-segment communications, bulk-personalized), rate-limits outbound contact, and pages a human for the portfolio-level decision. Detect correlated events and change processing mode — per-event handling of a correlated shock is both wasteful and dangerous.

### Trade-offs

| Choice | You gain | You pay |
|---|---|---|
| Stream-derived state vs periodic batch scans | Seconds-fresh risk posture; AI input ∝ change rate | Stateful stream ops (checkpointing, rescaling, event-time skew) — a real skill investment |
| Graduated autonomy | Speed on the routine, accountability on the severe | Tier-boundary tuning is a live risk-appetite negotiation with compliance; document it |
| LLM-personalized outreach (tier B) | Measurably better repayment engagement vs templates | Every generated message is a regulated communication — you need output guardrails, archival of exactly-what-was-sent, and sampling review |
| Cohort/storm mode | Survives correlated shocks; coherent customer treatment | More complex control plane; must test with replayed historical shock data — this is your chaos-engineering scenario |

### Scaling strategy

- **Partition by loan_id** everywhere — ordering per loan is what matters; cross-loan ordering doesn't. This gives linear Kafka/Flink scaling.
- **Replay as a superpower:** new risk model? Replay 90 days of the stream through it in shadow mode and diff decisions before cutover. The event log is your backtest environment — design retention accordingly.
- **Cost note:** tier B LLM traffic follows portfolio stress, not user traffic. Budget alerts on inference spend are a *risk signal* here — a spend spike literally means the book is deteriorating.

---

<a name="7-cross-cutting"></a>
## 7. Cross-Cutting Strategies

### Cost engineering (the fourth dimension of the latency/quality/throughput triangle)

Model cascading is the strategy that appears in every use case above: cheap tier handles the bulk, expensive tier handles the residual. Complement it with: prompt-caching static prefixes (system prompts, tool schemas — tool schemas are a hidden cost, another reason UC-5's virtual servers pay off), semantic caching of whole answers (UC-2), batch APIs for anything non-interactive (often ~50% discount), and *measuring cost per business outcome* (₹ per screened transaction, per underwritten loan) rather than per token — that's the number the customer's CFO understands and the number an FDE should put on slide one.

### Observability

Trace every request end-to-end with OpenTelemetry; the MCP spec now standardizes W3C trace-context propagation in `_meta`, so a single trace can span host → gateway → MCP server → downstream API. Beyond the RED metrics, real-time AI systems need four AI-specific SLOs: **TTFT p99** (user experience), **deadline-miss / fallback rate** (how often you served the degraded path), **evaluator/overturn rate** (live quality — UC-3's officer overturns, UC-1's async-review disagreements), and **cost per request by tier** (cascade health). Log prompts+outputs with retention matching your audit obligations — in regulated deployments the *inability* to reproduce why the model decided X is itself a finding.

### Failure design

Every architecture above shares one spine: **the model is allowed to fail; the system is not.** Concretely — hard deadlines with pre-agreed fallbacks (UC-1), circuit breakers per downstream with model-legible errors (UC-5), idempotency keys on every side effect (UC-3, UC-6), durable state for anything longer than a request (UC-3), and degradation modes that are *chosen and documented* (fail-safe vs fail-open) rather than emergent. Rehearse them: replay historical shock data (UC-6), inject seeded cases (UC-3), and load-test the fallback path, not just the happy path — the fallback path is the one that runs during your worst hour.

---

<a name="8-decision-cheat-sheet"></a>
## 8. Decision Cheat Sheet

| If the requirement is… | Reach for… | Watch out for… |
|---|---|---|
| Hard sub-second SLA with AI in the loop | Tiered scoring + hard deadline + fail-safe fallback (UC-1) | Deadline-miss rate becoming your silent quality leak |
| Fast grounded answers at scale | Semantic cache → hybrid retrieval → streaming (UC-2) | Cache staleness; vector-only retrieval missing exact identifiers |
| High-stakes decisions at volume | Durable workflows + confidence triage + reviewer UX (UC-3) | Automation complacency; miscalibrated confidence |
| Interactive perception/extraction | Tiered VLM + staged-progress UX + confidence routing (UC-4) | Hallucinated fields — always validate deterministically |
| Many agents × many tools | Gateway: identity chain, virtual servers, result firewall (UC-5) | God-credentials; tool-menu bloat; sticky sessions |
| AI reacting to event streams | Stream-derived state + graduated autonomy + storm mode (UC-6) | Redelivery without idempotency; per-event handling of correlated shocks |

**The one-paragraph synthesis:** real-time AI architecture is ordinary distributed-systems discipline with two new constraints — a component (the model) whose latency is large, variable, and expensive, and whose output is probabilistic. Everything above is the same two moves applied six ways: *push the model off the critical path or shrink what runs on it* (tiering, caching, precomputation, streaming), and *wrap every probabilistic output in deterministic structure* (schemas, validators, confidence routing, human gates, idempotent actions, audit). Master those two moves and every new use case is a remix.
