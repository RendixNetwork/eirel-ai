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
assistant with optional web search, across `instant` and `thinking`
modes. Future families are on the roadmap.

## Prerequisites

- Python >= 3.12 (for the `eirel` SDK and CLI)
- A Bittensor wallet you control (coldkey + hotkey)
- The operator's public `OWNER_API_URL`
- TAO on your coldkey to cover registration and the per-submission fee
  (see [Fees](#fees))

## 1. Create a Bittensor wallet

```bash
btcli wallet new_coldkey --wallet.name my-miner
btcli wallet new_hotkey  --wallet.name my-miner --wallet.hotkey m1
```

## 2. Register on the subnet

```bash
# finney mainnet example (netuid 36 at launch)
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

The SDK ships a minimal `general_chat` agent under
`eirel/examples/general_chat_agent/` that you can fork as a starting
point. See the SDK README for:

- `BaseAgent` / `MinerApp` patterns
- `MinerProviderConfig.from_env()` + `AgentProviderClient` for LLM routing
- `submission.yaml` manifest format (declares family, model, providers,
  resource requests)
- Local testing with `eirel serve`
- Compliance preflight with `eirel compliance`

Repo: <https://github.com/rendixnetwork/eirel>

## 4. Submit

The SDK CLI packages your source directory into a `.tar.gz` archive,
signs the request with your hotkey, pays the submission fee from your
coldkey, and uploads to the operator's `/v1/submissions` endpoint.

```bash
eirel submit \
  --source-dir ./my-agent \
  --owner-api-url https://api.eirel.ai \
  --network finney \
  --wallet-name my-miner \
  --hotkey-name m1
```

The CLI prompts to confirm the fee transfer, pays on-chain, then
uploads. On success you get a `submission_id` and a `deployment_id`.

The operator's owner-api:

1. Verifies the hotkey signature + fee extrinsic on chain.
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

Every submission requires a **0.1 TAO** transfer to the subnet treasury.
The fee is non-refundable and exists to make subnet spam economically
uninteresting. The SDK handles payment + hash inclusion automatically;
if you already have an extrinsic hash (e.g. you paid manually), pass it
with `--extrinsic-hash <hash> --block-hash <hash>` instead of re-paying.

## Fair-play notes

- **Your agent is invoked via an HTTP proxy.** Validators never talk to
  your container directly — they hit the operator's
  `/runtime/{deployment_id}/v1/agent/infer` path, which routes to your
  pod. This means you don't need a public IP or axon.
- **Budget enforcement is at the proxy.** Your LLM spend is capped per
  run at the provider-proxy layer (`EIREL_RUN_BUDGET_USD`). Requests
  that would push you over-budget are rejected.
- **Latency is measured at the proxy.** Your deployment's p50 latency
  feeds a penalty curve on the operator side. Large p50 deficits drag
  your official score down even if raw quality is high.

## Troubleshooting

**`submission fee payment required: provide extrinsic_hash of 0.1 TAO
transfer to treasury`** — fee-verification is enabled and your
submission didn't carry an extrinsic hash. Either let the SDK pay
(`eirel submit` without `--skip-fee`) or pre-pay and supply the hash.

**`extrinsic_hash already used for a prior submission`** — you can't
re-use a single fee payment for multiple submissions. Pay again, or
supply a fresh pre-paid hash.

**Deployment stuck at `build_failed`.** Your archive failed to build
(missing dep, bad `submission.yaml`, oversized image). Check the
deployment's `health_details_json` via `eirel status` for the build
log reason.

**Deployment reaches `deployed_for_eval` but validators score you 0.**
Either your agent is returning empty responses, your citations are
being trace-gated, or your `provider_proxy` credentials aren't
propagating. Exercise your agent locally with `eirel serve` and hit it
with a sample invocation before re-submitting.
