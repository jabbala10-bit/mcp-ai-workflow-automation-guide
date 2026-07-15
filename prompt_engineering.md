# Prompt Engineering for Enterprise FDEs

> **Where this sits in the library:** This is the craft layer that runs *on top of* the LLM service module from Project 5. Every prompt in production should flow through that single routing point so it inherits observability, versioning, cost tracking, and eval hooks. A prompt without an eval attached is a liability, not an asset. Treat prompts as versioned artifacts (semver, git-tracked, tied to a test set), not as strings you paste into a chat window.

---

## 1. Prompt Engineering Overview — The #1 FDE Superpower

**The core insight:** In a deployment, the model is usually fixed (the customer bought Claude or GPT), the retrieval layer is fixed early, and the data is what it is. The one lever you can pull daily, cheaply, and without a redeploy is the prompt. This is why ~80% of realized AI quality traces back to prompt design and context construction — not model choice.

**The FDE reframe:** You are rarely asked "which model?" You are asked "why is the output wrong 30% of the time?" The answer is almost always in the prompt, the context assembly, or the eval loop — not the weights.

### Real deployment scenario — Contract review assistant (legal team, 400-person SaaS company)

You ship a first cut that summarizes contracts and flags risky clauses. Legal accepts 58% of outputs. Unusable. You don't touch the model. You do this:

- Add a **role** ("You are a contracts attorney reviewing on behalf of the *vendor*") — acceptance jumps because the model now flags the right side's risk.
- Add **explicit output structure** (clause → risk level → rationale → suggested redline) — reviewers can scan instead of read.
- Add **few-shot examples** of *their* three risk tiers with *their* house definitions.
- Add a **constraint**: "If the contract does not contain a limitation-of-liability clause, say so explicitly. Never infer one."

Acceptance moves to 94%. Zero infra changed. That is the superpower — and it's why the prompt layer is the first thing you instrument.

### The mental model to carry into every engagement

```
Output quality = f(task clarity, context, examples, constraints, format, decoding params)
```

When output is bad, walk that function left to right. Most teams jump straight to "we need a bigger model" — the FDE's job is to prove that wrong 90% of the time and save the customer six figures.

---

## 2. Zero-Shot / Few-Shot / Chain-of-Thought — Knowing Which Lever

### Zero-shot — no examples, just instructions

Use when the task is common in the model's training distribution and the taxonomy is intuitive.

```
Classify this support ticket's urgency as LOW, MEDIUM, or HIGH.
Ticket: "Login page returns a 500 for all users since the deploy 10 min ago."
Urgency:
```
This works out of the box. "Site down for everyone" is unambiguously HIGH. No examples needed.

### Few-shot — teach the model *your* rules

Use when the customer has an internal taxonomy the model can't guess, when edge cases matter, or when you need consistent formatting. This is the single highest-ROI technique in enterprise work because every company has private definitions.

**Scenario — insurance claim triage.** The carrier has 4 internal severity bands defined by dollar exposure *and* injury type, and the boundaries are not obvious.

```
Classify each claim into one of: FAST_TRACK, STANDARD, COMPLEX, SIU_REFERRAL.
Follow these examples exactly.

Claim: "Rear-end collision, $1,200 bumper damage, no injuries reported."
Class: FAST_TRACK

Claim: "Multi-vehicle, disputed fault, soft-tissue injury claim, attorney already involved."
Class: COMPLEX

Claim: "Single-vehicle theft, policy opened 8 days ago, prior claim last month."
Class: SIU_REFERRAL   # recency + prior pattern = fraud signal

Claim: "Windshield crack, $600, comprehensive coverage, clean history."
Class: STANDARD

Now classify:
Claim: "{{claim_text}}"
Class:
```

Notice the third example encodes a *rule* (new policy + prior claim → fraud review) that no zero-shot prompt would infer. The comment teaches the boundary. **Rule of thumb:** 3–5 examples, cover the boundaries not the obvious center, and include at least one "trap" case.

### Chain-of-Thought (CoT) — force reasoning before the answer

Use when the task requires multi-step logic, arithmetic, or comparison. The failure mode CoT fixes: the model blurts a confident wrong answer because it committed before reasoning.

**Scenario — invoice/PO three-way match with tolerances.**

```
You are validating whether an invoice matches its purchase order.
Think step by step BEFORE giving a verdict.

Rules:
- Quantity must match exactly.
- Unit price may vary within ±3%.
- Total = quantity × unit price (check the math).

PO:      50 units @ $20.00 = $1,000.00
Invoice: 50 units @ $20.50 = $1,025.00

Reasoning:
1. Quantity: 50 vs 50 → match.
2. Unit price: $20.50 vs $20.00 = +2.5% → within ±3% tolerance.
3. Math: 50 × $20.50 = $1,025.00 → matches stated total.
Verdict: APPROVE

Now evaluate:
PO:      {{po_line}}
Invoice: {{inv_line}}
Reasoning:
```

**Production caveat FDEs must know:** CoT costs tokens and latency, and it leaks the reasoning into the output. In production you often want the reasoning *computed* but not *shown*. Two moves: (a) ask for reasoning then a final line, and parse only the verdict; or (b) use the model's native extended-thinking/reasoning mode where the chain is separate from the answer. Never dump raw CoT to an end user in a customer-facing app — it's verbose and occasionally exposes uncertainty you don't want surfaced.

### The decision table (keep this in your head)

| Situation | Technique |
|---|---|
| Common task, intuitive labels | Zero-shot |
| Private taxonomy / house style / edge cases | Few-shot |
| Math, multi-step logic, tolerances, comparisons | CoT |
| Long doc, many sub-decisions, validation gates | Chaining (§4) |
| Need consistent parseable output | Few-shot + structured output (§5) |

---

## 3. System Prompts & Role Design

**Formula:** `Persona + Instructions + Constraints + (Context/Tools) = reliable behavior.`

The system prompt is where you set behavior that must hold across *every* turn. The user prompt carries the task; the system prompt carries the identity and the rules. FDEs live in the system prompt — it's the contract between the customer's requirements and the model's behavior.

### Anatomy, with a real fintech support agent

```
## Persona
You are Ava, a support agent for Northwind Bank. You are precise, calm,
and never speculative about a customer's account.

## Instructions
- Answer using ONLY the account data and policy snippets provided in context.
- If the answer isn't in the provided context, say you'll route to a human
  and do not guess.
- Keep replies under 120 words unless the customer asks for detail.

## Constraints (hard rules — never violate)
- NEVER reveal full account numbers; show only the last 4 digits.
- NEVER give tax, legal, or investment advice. Redirect to a licensed advisor.
- NEVER process a transaction; you can only explain how to.
- If the customer expresses financial distress, surface the hardship-support
  line and stop troubleshooting.

## Output
- Plain text, no markdown, first-person as Ava.
```

Why each block earns its place: the **persona** sets tone and reduces hedging; the **instructions** anchor to context and kill hallucination ("only the data provided"); the **constraints** are your compliance and safety surface — this is where AI governance (your EU AI Act / DPDP work) becomes concrete engineering. A regulated deployment's audit trail often *is* the system prompt plus its version history.

### Role design tips that move quality

- **Give the right role, not just "an assistant."** "You are a senior SRE" pulls in different priors than "you are a helpful assistant" — the former reasons about blast radius and rollbacks unprompted.
- **State what to do on uncertainty, explicitly.** The default failure mode is confident fabrication. "If you're unsure, say so and ask one clarifying question" changes the whole risk profile.
- **Put the non-negotiables in ALL CAPS or a dedicated "hard rules" block.** Models weight emphasis and structure. Compliance rules should be visually and structurally separated from soft preferences.
- **Negative constraints need a positive alternative.** "Don't give investment advice" is weaker than "Don't give investment advice — instead, direct them to a licensed advisor and offer the general education article." Tell it what to do, not just what to avoid.

---

## 4. Prompt Chaining — Decompose Instead of Cramming

**Principle:** One mega-prompt that extracts, validates, reasons, and formats will do all four jobs mediocrely and fail unobservably. Break it into steps where each step has one job, a checkable output, and its own eval.

### Why chain (the FDE argument to a skeptical customer)

- **Observability:** you can see *which* step failed instead of a black-box wrong answer.
- **Targeted fixes:** you can improve the extractor without touching the summarizer.
- **Cheaper models on easy steps:** route classification to Haiku, reasoning to Opus. Chaining is a cost-optimization lever, not just a quality one.
- **Guardrails between steps:** validate structured output before it poisons the next stage.

### Real scenario — automated RFP response pipeline (B2B sales engineering)

An inbound RFP is a 40-page PDF. One prompt can't reliably turn it into a draft response. Chain it:

```
Step 1 — EXTRACT (Haiku, temp 0)
  Input:  raw RFP text
  Output: JSON list of discrete requirements
          [{id, requirement_text, section}]

Step 2 — CLASSIFY (Haiku, temp 0)
  Input:  each requirement
  Output: {id, category: [security|integration|pricing|compliance|other],
           is_dealbreaker: bool}

Step 3 — MATCH (Opus, temp 0.2)
  Input:  requirement + retrieved product-capability docs (RAG)
  Output: {id, we_support: [full|partial|no], evidence_doc_id, gap_notes}

Step 4 — DRAFT (Opus, temp 0.4)
  Input:  matched requirements + house tone guide
  Output: prose response section per requirement

Gate between 2 and 3: if any is_dealbreaker requirement maps to
"we_support: no" in a dry-run, flag for human before drafting.
```

Each arrow is a place to log, validate, retry, and eval. That "gate" is pure FDE value — it catches the deal-killing gap *before* sales sends a confident wrong answer to a prospect.

### Chaining patterns worth naming

- **Extract → Validate → Act:** never act on unvalidated model output. Parse to a schema, check invariants, *then* proceed.
- **Map → Reduce:** summarize each chunk of a long doc (map), then summarize the summaries (reduce). Standard move for documents past the useful context window.
- **Generate → Critique → Revise:** one call drafts, a second call critiques against a rubric, a third revises. Meaningfully lifts quality on writing tasks — at 3× the cost, so reserve it for high-value output.
- **Router → Specialist:** a cheap classifier decides which specialist prompt handles the request. This is the backbone of most production agents.

---

## 5. Structured Output (JSON Mode) — Making Output Machine-Parseable

If a human reads the output, prose is fine. If *code* reads the output, you need guaranteed structure. Enterprise systems are code reading output — so this is most of the job.

### The three levels of reliability (worst → best)

**Level 1 — Ask nicely (fragile).**
```
Return the answer as JSON with keys "sentiment" and "confidence".
```
Works ~90% of the time, breaks on the 10% where the model adds "Here's your JSON:" or wraps it in prose. Never ship this alone.

**Level 2 — Prefill / constrain the opening.** With Anthropic, prefill the assistant turn with `{` so the model can only continue valid JSON. With OpenAI, set `response_format: {type: "json_object"}`. This forces well-formed JSON.

**Level 3 — Schema-enforced (best).** Use tool use / function calling with an explicit JSON Schema, or a structured-outputs mode that guarantees the response validates against your schema.

```python
# Anthropic tool-use pattern (schema-enforced extraction)
tools = [{
    "name": "record_invoice",
    "description": "Record extracted invoice fields.",
    "input_schema": {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total_amount":   {"type": "number"},
            "currency":       {"type": "string", "enum": ["USD","EUR","GBP"]},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "qty":  {"type": "integer"},
                        "unit_price": {"type": "number"}
                    },
                    "required": ["description","qty","unit_price"]
                }
            }
        },
        "required": ["invoice_number","total_amount","currency"]
    }
}]
# Force the tool so output is guaranteed to match the schema:
# tool_choice = {"type": "tool", "name": "record_invoice"}
```

### The non-negotiable backend rule

**Always validate the parsed output against a real schema (Pydantic/Zod) before you trust it.** Schema-enforced modes guarantee *shape*, not *correctness* — a model can still hallucinate a plausible invoice number. Your LLM service layer should: parse → validate → on failure, retry once with the error message fed back → on second failure, route to human. That retry-with-error loop is one of the highest-leverage 20 lines of code in a production LLM system.

```python
# Self-healing parse loop (pseudocode for the service layer)
for attempt in range(2):
    raw = call_model(prompt)
    try:
        return InvoiceModel.model_validate_json(extract_json(raw))
    except ValidationError as e:
        prompt += f"\n\nYour last output failed validation: {e}. Fix and return valid JSON only."
raise EscalateToHuman()
```

---

## 6. Temperature & Decoding Controls — Creativity vs. Precision

Temperature scales randomness in token selection. Low = deterministic and repeatable. High = diverse and creative. The FDE mistake is leaving it at the default (~1.0) for extraction tasks and then wondering why the same invoice parses differently twice.

| Task | Temp | Why |
|---|---|---|
| Extraction, classification, routing | 0.0 | You want the *same* answer every run. Reproducibility = testability. |
| Code generation, SQL, structured output | 0.0–0.2 | Correctness over variety. |
| Data transformation / reformatting | 0.0 | Deterministic mapping. |
| Q&A over documents (RAG) | 0.0–0.3 | Grounded, low invention. |
| Summarization | 0.3–0.5 | Slight fluency without drifting from source. |
| Drafting emails / customer prose | 0.4–0.7 | Natural tone, still on-brief. |
| Brainstorming, marketing variants, naming | 0.7–1.0 | You *want* spread. |

### The other knobs

- **top_p (nucleus sampling):** caps the probability mass considered. Tune temperature *or* top_p, not both — moving both makes behavior hard to reason about. Default to temperature.
- **max_tokens:** a cost and safety cap. Set it deliberately per task — an extraction task returning 4,000 tokens is a bug (probably a runaway loop or the model narrating).
- **stop sequences:** end generation at a delimiter. Useful in chaining to cut the model off cleanly (`stop: ["</result>"]`).
- **seed (where supported):** pin randomness for reproducible tests. Combine with temp 0 for eval determinism.

**FDE default posture:** start every new prompt at temp 0. Raise it only when the output is *supposed* to vary. Determinism is the friend of debugging and evals; you earn your way up to higher temperatures, you don't start there.

---

## 7. Prompt Evaluation — Measure, Don't Vibe

This is what separates an engineer from someone who "prompts." A prompt you can't measure is a prompt you can't safely change. Every prompt in the portfolio should ship with an eval set. This is also the single most FDE-signature skill — customers trust the person who can *prove* the system works.

### Step 1 — Build a golden set

50–200 real inputs with known-correct outputs, harvested from actual customer data (anonymized per your DPDP/GDPR obligations). Deliberately over-sample edge cases and past failures. This set is an asset that outlives any single prompt version.

### Step 2 — Pick a scoring method that fits the task

- **Exact/structural match** — classification, extraction. Cheap, deterministic, no LLM needed. Accuracy, precision/recall, F1, per-class confusion.
- **Assertion checks** — "output is valid JSON", "every dollar figure appears in the source", "no full account number leaked". Programmatic guardrails as tests.
- **LLM-as-judge** — for open-ended output (summaries, drafts) where exact match is meaningless. A separate model scores against a rubric.

```
# LLM-as-judge rubric prompt (run at temp 0)
Score the summary 1–5 on each dimension. Return JSON only.
- faithfulness: no claims absent from the source
- coverage: captures the key points
- concision: no filler
Source: {{source}}
Summary: {{candidate}}
```

Guard the judge: use a strong model, force structured scores, and periodically check judge agreement against human labels so you're not trusting a biased ruler.

### Step 3 — A/B test prompt versions rigorously

```
Prompt A (current) vs Prompt B (candidate), same 150-case golden set:

              Accuracy   Format-valid   Avg cost   p95 latency
Prompt A       0.86         0.97         $0.011      1.4s
Prompt B       0.93         0.99         $0.014      1.9s
```
Now it's a *decision*, not a vibe: +7pts accuracy for +$0.003/call and +0.5s. If this is contract review, ship B. If it's autocomplete, maybe not. **Log every dimension — accuracy, format validity, cost, latency — because the customer's answer depends on which one binds.**

### Step 4 — Regression-test on every prompt change

Wire the golden set into CI. A prompt edit that fixes case #7 but silently breaks cases #22 and #40 must fail the build. Prompts are code; treat edits like code changes with a test gate. This is exactly why every model call routes through the service layer — that's where the eval hook lives.

---

## 8. Enterprise Patterns — The Four Workhorses

95% of enterprise LLM work is one of these four. Learn the template for each and you can scope most deployments on sight.

### Classification — route inputs into known buckets

**Use cases:** ticket triage, intent detection, content moderation, lead scoring, claim severity.

```
Classify the customer email into exactly one: BILLING, TECHNICAL, SALES, CHURN_RISK, OTHER.
Respond with only the label.

Rules:
- CHURN_RISK: any mention of cancelling, competitor, or "not worth it".
- BILLING beats TECHNICAL if both appear (money first).

Email: "{{email}}"
Label:
```
Temp 0, few-shot for the boundaries, structured single-token output, exact-match eval. The `CHURN_RISK` rule and the tie-breaker are the kind of house logic that makes this *their* classifier, not a generic one.

### Extraction — pull structured data from unstructured text

**Use cases:** invoices, resumes, contracts, medical intake, KYC docs, lab reports.

```
Extract these fields from the resume. Use null if absent — never invent.
Return JSON only: {name, email, years_experience, top_3_skills, current_title}

Resume: "{{resume_text}}"
```
Temp 0, schema-enforced (tool use), Pydantic validation on the backend, `null`-not-guess discipline. The "never invent, use null" instruction is the difference between a usable extractor and a hallucinating one.

### Summarization — compress while preserving what matters

**Use cases:** call transcripts, research digests, exec briefings, meeting notes, long-thread recaps.

```
Summarize this support call for a manager who has 30 seconds.
- 3 bullets max.
- Include the customer's core issue, the resolution, and any follow-up owed.
- Do not include pleasantries or hold times.

Transcript: "{{transcript}}"
```
Temp 0.3–0.5, faithfulness-scored via LLM-as-judge, tight length constraint. For long inputs, use the map→reduce chain from §4. The audience framing ("manager, 30 seconds") does more work than any length number alone.

### Transformation — restructure or convert format

**Use cases:** text→SQL, JSON→email, legacy format→modern schema, tone rewrites, translation-with-glossary, code migration.

```
Convert this natural-language request into a SQL query for the schema below.
Return only the SQL. Use parameterized placeholders, never string interpolation.

Schema: orders(id, customer_id, total, created_at), customers(id, name, region)
Request: "Total revenue by region for last quarter, highest first."
SQL:
```
Temp 0, provide the schema in context, constrain to safe patterns (parameterization is a security constraint, not a nicety), validate the SQL parses before executing. This pattern is where prompt engineering and your governance/security instincts meet — an unconstrained text-to-SQL prompt is an injection vector.

---

## The FDE checklist for any prompt going to production

1. **Role + constraints** in a system prompt, with hard rules separated out.
2. **Few-shot examples** covering the boundaries and one trap case.
3. **Structured output** with schema enforcement + backend validation + retry loop.
4. **Temperature set deliberately** — default 0, earn your way up.
5. **A golden eval set** attached, wired into CI as a regression gate.
6. **Routed through the LLM service layer** so it inherits logging, versioning, cost tracking, and evals.
7. **Versioned in git** with a changelog tying each version to its eval scores.

A prompt that has all seven is an engineering artifact. A prompt missing them is a demo. Your job as an FDE is to turn the customer's demo into the artifact — that transformation *is* the deployment.
