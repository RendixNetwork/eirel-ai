# Running a miner on the EIREL subnet

This guide is for miners who want to submit an agent to the subnet. The
bulk of the SDK (how to build an agent, typed request/response schemas,
provider abstraction, resume tokens, examples) lives in the
[`eirel` SDK repo](https://github.com/rendixnetwork/eirel) — this guide
covers the subnet-side flow: register, build, submit, monitor.

## Overview

Miners submit containerized agents that the operator builds and deploys
on subnet-owned runtime. Validators dispatch evaluation tasks to your
deployment, score the responses locally, and publish weights that drive
TAO emissions.

The only launch family is **`general_chat`** — multi-turn conversational
assistant across `instant` and `thinking` modes. Owner-routed tool
services are available to your agent: web search, URL fetch, a Python
sandbox for verifiable computation, and `rag.retrieve` over per-run
document corpora (used by `rag_required` tasks). Future families are on
the roadmap.

### How scoring shapes weights

Within each family, weight allocation is **winner-take-all**. The single
highest-scoring miner of a run receives the family's full weight
allocation; every other miner in the same family gets exactly 0 emission
that run, regardless of how close their score was. To dethrone the
current winner, a challenger must beat the incumbent's re-scored value
by a fixed margin (`0.05` on the 0–1 scale) — otherwise the incumbent
is retained for the next run. Tune your agent to win, not to place.

The "score" the leaderboard ranks on is a multiplicative composite:

```
score = grounded_gate × safety_gate × tool_attestation
      × hallucination_knockout × cost_attestation
      × efficiency × (outcome_score + pairwise_bonus)
```

Each gate is in `[0, 1]` and any one of them can zero the whole score,
so robustness matters as much as raw answer quality. The Knockout gates
that most often surprise new miners:

- **`tool_attestation`** — for tasks where the task envelope marks a
  required tool (e.g. `rag.retrieve` for `rag_required`), failure to
  actually invoke that tool zeros the score. Tool calls are recorded
  on a server-side ledger; you can't fake them.
- **`hallucination_knockout`** — if your answer restates a value the
  task marked as `must_not_claim` (e.g. a phone number the user never
  shared), you zero the task. Refuse cleanly when the requested fact
  isn't in the conversation history.
- **`grounded_correctness`** — answers are cross-checked against a
  three-oracle reconciler (OpenAI + Gemini + Grok via Responses API,
  reconciled by Chutes GLM). Confidently wrong answers score very low.
- **`cost_attestation`** — exceeding the per-run USD budget on the
  provider proxy gets the over-budget request rejected and the
  attestation factor reduced.

## Prerequisites

- Python >= 3.12 (for the `eirel` SDK and CLI)
- A Bittensor wallet you control (coldkey + hotkey)
- The operator's public `OWNER_API_URL`
- TAO on your coldkey to cover **subnet registration only** (submissions
  themselves are fee-free on this subnet — see [Fees](#fees))

## 1. Create a Bittensor wallet

```bash
btcli wallet new_coldkey --wallet.name my-miner
btcli wallet new_hotkey  --wallet.name my-miner --wallet.hotkey m1
```

## 2. Register on the subnet

```bash
btcli subnet register \
  --subtensor.network finney \
  --netuid 36 \
  --wallet.name my-miner \
  --wallet.hotkey m1
```

## 3. Install the eirel SDK and build an agent

```bash
pip install eirel
```

The SDK ships a reference `general_chat` agent under
`eirel/examples/graph_general_chat/` (single-pass baseline) and a
stronger `eirel/examples/graph_dominant_chat/` (planner + tool fan-out +
memory introspection + sandbox compute) that you can fork. The canonical
authoring shape is the graph SDK (`StateGraph` + `GraphAgent`); the
legacy `BaseAgent` / `MinerApp` shape is still exported but graph-based
composition is preferred for new agents.

See the SDK README for:

- `StateGraph` builder + `GraphAgent` (canonical authoring shape)
- `MinerProviderConfig.from_env()` + `AgentProviderClient` for LLM routing
- Built-in tools: `WebSearchTool`, `UrlFetchTool`, `SandboxTool`,
  `RagTool` (the RAG client is required for `rag_required` tasks; pass
  the `metadata.rag_corpus_id` from the request envelope to
  `rag.retrieve`)
- `submission.yaml` manifest format (declares family, model, providers,
  resource requests)
- Local testing with `eirel serve`
- Compliance preflight with `eirel compliance`

Repo: <https://github.com/rendixnetwork/eirel>

## 4. Submit

The SDK CLI packages your source directory into a `.tar.gz` archive,
signs the request with your hotkey, and uploads to the operator's
`/v1/submissions` endpoint. Pass `--skip-fee` so the CLI doesn't try to
make the (no-longer-required) on-chain treasury transfer.

```bash
eirel submit \
  --source-dir ./my-agent \
  --owner-api-url https://api.eirel.ai \
  --network finney \
  --wallet-name my-miner \
  --hotkey-name m1 \
  --skip-fee
```

On success you get a `submission_id` and a `deployment_id`.

The operator's owner-api:

1. Verifies the hotkey signature.
2. Retires your previous deployment (if any) — you have one active
   deployment per family at a time.
3. Builds your image and places a pod on subnet-owned runtime.
4. Marks the deployment `deployed_for_eval` once healthy.

Subsequent runs pick it up automatically; validators start scoring
against the new deployment at the next run boundary.

## 5. Check status and scores

```bash
eirel status \
  --owner-api-url https://api.eirel.ai \
  --wallet-name my-miner \
  --hotkey-name m1
```

Shows the current submission state, progress of the in-flight build, and
the last few scorecards once evaluation runs complete.

## Fees

**Submissions are free on this subnet.** The on-chain treasury transfer
that used to gate `/v1/submissions` has been disabled — pass `--skip-fee`
to the SDK CLI and your submission goes through with no TAO transfer.
You still need TAO to register your hotkey on the subnet (`btcli subnet
register`), but nothing beyond that for the submit flow itself.

## Submission limits

To keep the eval queue clean and discourage spray-and-pray spam, every
hotkey can submit at most **2 submissions, lifetime**. Both attempts
count even if the first one failed to build — there's no "free retry"
on a broken archive. The 3rd POST to `/v1/submissions` from the same
hotkey returns HTTP `429`.

Plan accordingly:

- Run your agent locally with `eirel serve` and exercise it against
  representative prompts before submitting.
- Use `eirel compliance` to catch manifest errors and missing capability
  declarations before they burn one of your two slots.
- Save your first slot for a polished v1, not a smoke test.

## Limitations to design around

A short list of constraints worth knowing before you architect your
agent:

- **One family at launch (`general_chat`).** Other families
  (`deep_research`, `code_agent`, ...) are on the roadmap but not yet
  scoring on chain.
- **Multi-turn history cap.** The SDK's request schema enforces
  `history.max_length = 100`. Multi-turn agentic-memory fixtures can
  exceed this; the validator's flattened history may end up over 100,
  in which case the request 400s at SDK validation. Defensive miners
  either truncate at their app boundary (override the `/v1/agent/infer*`
  routes — see `graph_dominant_chat` for an example) or accept that
  >100-turn tasks may fail.
- **One active deployment per family.** A new submission retires your
  prior deployment for that family. Don't expect to A/B two builds
  simultaneously under one hotkey.
- **Per-run USD budget.** Your provider-proxy spend is capped per run
  (`EIREL_RUN_BUDGET_USD` on the operator side). Over-budget LLM calls
  are rejected at the proxy layer; persistent rejections drag
  `cost_attestation` down.
- **Source becomes public after the run that scored it closes.** During
  the run, only you can download your own archive. Once the run is
  marked `completed`, every submission archive scored in that run is
  publicly downloadable + viewable from the leaderboard. Your
  competitors can study your code after a run ends — write for the
  expected post-run scrutiny.
- **Tasks are hidden until run time.** Eval bundles are generated at
  run open and never published. You can't pre-train against them. You
  *can* analyze your per-task `EvalFeedback` rows after a run to
  understand which capabilities you missed (see below).

## Fair-play notes

- **Your agent is invoked via an HTTP proxy.** Validators never talk to
  your container directly — they hit the operator's
  `/runtime/{deployment_id}/v1/agent/infer` path, which routes to your
  pod. This means you don't need a public IP or axon.
- **Budget enforcement is at the proxy.** Your LLM spend is capped per
  run at the provider-proxy layer. Requests that would push you
  over-budget are rejected.
- **Latency is measured at the proxy.** Your deployment's p50 latency
  feeds a penalty curve on the operator side. Large p50 deficits drag
  your official score down even if raw quality is high.
- **Tool calls are server-attested.** When you invoke a subnet tool
  (web search, URL fetch, sandbox, `rag.retrieve`), the orchestrator
  records the call on a server-side ledger. Validators score
  `tool_attestation` from this ledger — claiming "I retrieved the
  document" in your response without actually calling `rag.retrieve`
  zeros that factor for `rag_required` tasks.

## Per-task feedback

After each run, you can fetch your per-task `EvalFeedback` rows from
owner-api directly with a hotkey-signed request:

```bash
curl -H "X-Eirel-Hotkey: <your-hotkey>" \
     -H "Authorization: <signature>" \
     "$OWNER_API_URL/v1/eval/feedback?run_id=run-N"
```

The `eirel` CLI's `status` subcommand surfaces these alongside the
scorecard. Each row includes `outcome` (correct / partial / wrong /
hallucinated / refused / disputed), `failure_mode`, `guidance`,
`composite_score`, and any `knockout_reasons` that zeroed your score.
Rows are filtered to your hotkey by the signature — you can't read
another miner's feedback.

## Troubleshooting

**`hotkey submission cap reached (lifetime limit: 2 submissions per
hotkey)`** — you've used both of your lifetime submission slots from
this hotkey. There is no override. Register a new hotkey if you need
another two attempts, or wait for the cap policy to evolve.

**Deployment stuck at `build_failed`.** Your archive failed to build
(missing dep, bad `submission.yaml`, oversized image). Check the
deployment's `health_details_json` via `eirel status` for the build
log reason. Note: this still counts as one of your two lifetime
submissions.

**Deployment reaches `deployed_for_eval` but validators score you 0.**
Either your agent is returning empty responses, your citations are
being trace-gated, your `provider_proxy` credentials aren't propagating,
or you're failing one of the composite knockout gates (most commonly
`tool_attestation` for `rag_required` tasks — verify your agent
actually calls `rag.retrieve` for those). Exercise your agent locally
with `eirel serve` and hit it with a sample invocation before
re-submitting.

**Multi-turn task fails with `pod returned 400`.** The fixture flattened
to more than 100 history entries and the SDK rejected it. See
[Limitations](#limitations-to-design-around) — the fix is to override
the `/v1/agent/infer*` route in your app and truncate history before
the SDK validates it.
