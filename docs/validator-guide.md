# Running a validator on the EIREL subnet

This guide is for third parties who want to run a validator. If you're the
subnet operator, you want the root [`README.md`](../README.md) and the
`docker-compose.yml` at the repo root instead.

## Overview

A validator on the EIREL subnet does three things:

1. **Claim tasks** from the operator's shared task pool (one batch at a
   time, with a per-task timeout so unattended tasks go back to `pending`).
2. **Evaluate** each task by calling the miner agent (via the operator's
   HTTP proxy) and scoring the response with an **LLM judge you run
   locally on your own machine**.
3. **Publish weights** on-chain based on the scores you've computed for
   each miner.

The judge runs on your side (not the operator's) so subnet consensus is
driven by independent validator opinions, not a single operator oracle.
You pay for your own judge LLM calls — budget roughly **$0.15–$0.30 per
20-task run × 3 miners** with Kimi-K2.5 via Chutes at current rates.

## Prerequisites

- Docker + Docker Compose
- A Bittensor wallet you control (coldkey + hotkey)
- The operator's public `OWNER_API_URL`
- Your validator hotkey added to the subnet's allow-list (the operator
  does this after you share your SS58)
- A **Chutes API key** for the judge LLM. Sign up at <https://chutes.ai>
  and create a key. You'll fund this account yourself.

## Machine requirements

The three validator services are I/O-bound (HTTP to the operator's
owner-api, HTTPS to Chutes, and to the Bittensor chain). None of them
train or run models locally — the LLM call goes out to Chutes.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| vCPU | 2 | 4 |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB (SSD) | 40 GB (SSD) |
| Network (out) | 25 Mbit/s, stable | 100 Mbit/s+ |
| OS | Linux x86_64 with kernel ≥ 5.10 | Ubuntu 22.04 LTS / Debian 12 |
| Docker | Engine 24+ with Compose v2 | Latest stable |
| Uptime | 24/7 | 24/7 with automatic restart |

**No public IP, no inbound ports.** The validator only makes outbound
connections. You do not need to expose anything to the internet.

**Outbound reachability.** The host must be able to reach:

- The operator's `OWNER_API_URL` (HTTPS)
- `https://llm.chutes.ai` (or your configured `EIREL_JUDGE_BASE_URL`)
- A Bittensor chain endpoint (finney: `wss://entrypoint-finney.opentensor.ai`; testnet: `wss://test.finney.opentensor.ai`)
- `ghcr.io` / Docker Hub for image pulls

**Clock sync.** Keep the host clock within a few seconds of UTC. Signed
requests include a timestamp and replay protection rejects drift beyond
the signature window.

**Uptime matters.** Every run you miss reduces the task share your
hotkey contributes to consensus, which directly reduces emissions.
Anything you'd trust for a long-running process (a small VPS, a home
server with good power + internet) is fine.

## 1. Create a Bittensor wallet

```bash
btcli wallet new_coldkey --wallet.name my-validator
btcli wallet new_hotkey  --wallet.name my-validator --wallet.hotkey v1
```

Keys land at `~/.bittensor/wallets/my-validator/`.

## 2. Register on the subnet

```bash
# finney mainnet example (netuid 36 at launch)
btcli subnet register \
  --subtensor.network finney \
  --netuid 36 \
  --wallet.name my-validator \
  --wallet.hotkey v1
```

Registration costs TAO; see `btcli help subnet register` for the live
rate.

## 3. Share your hotkey with the operator

Send your validator hotkey SS58 to the subnet operator. They add it to
their validator allow-list; without this your `/v1/tasks/claim` calls
return 403. Get your SS58 with:

```bash
btcli wallet list --wallet.name my-validator
```

## 4. Configure your validator

From the repo root:

```bash
cp .env.validator.example .env.validator
```

Fill in `.env.validator`:

| Variable | What it is |
|----------|------------|
| `OWNER_API_URL` | Public URL provided by the operator |
| `EIREL_VALIDATOR_WALLET_NAME` | `my-validator` (from step 1) |
| `EIREL_VALIDATOR_HOTKEY_NAME` | `v1` (from step 1) |
| `BITTENSOR_NETWORK` | `finney` for mainnet |
| `BITTENSOR_NETUID` | `36` (mainnet) |
| `BITTENSOR_WALLETS_PATH` | Host path to wallets dir (default `/root/.bittensor/wallets`) |
| `EIREL_PROVIDER_CHUTES_API_KEY` | Your Chutes key — **funds the judge LLM bill** |
| `EIREL_JUDGE_API_KEY` | Same as `EIREL_PROVIDER_CHUTES_API_KEY` (for the judge service) |
| `EIREL_JUDGE_MODEL` | Leave at the default (`moonshotai/Kimi-K2.5-TEE`) unless you have a reason |
| `EIREL_JUDGE_TIMEOUT_SECONDS` | Validator client timeout when calling the judge (default `90`). Must exceed the judge's own LLM timeout so the deterministic fallback can return. |

## 5. Start the stack

```bash
docker compose -f docker-compose.validator.yml up -d
```

Two containers come up:

- `eiretes-judge` — LLM judge, listens on port **18095** (host) / 8095
  (container). First request triggers a health check against Chutes.
- `validator-engine` — claims + invokes miners + judges + submits, and
  also runs the in-process weight-setting loop that publishes consensus
  weights on-chain every ~180 blocks. Listens on port **18010**.

Watch logs:

```bash
docker compose -f docker-compose.validator.yml logs -f validator-engine
```

You should see claim → invoke → judge → submit cycles once your hotkey
is whitelisted AND the operator has an open run with queued tasks.

## Trust model & scoring

**Every validator scores independently.** Two validators evaluating the
same miner response can arrive at different scores if they use different
models or run into different judge-LLM hiccups. Bittensor yuma consensus
reconciles the differences — validators whose weights align with the
majority earn more emissions; outliers earn less.

**Run the recommended model.** `moonshotai/Kimi-K2.5-TEE` is what the
operator used to tune the rubric and what miners expect. Using a
materially different model (e.g. a tiny OSS model, a model with
different reasoning style) produces weight divergence from consensus
and directly reduces your TAO earnings. The field is unchanged from
`EIREL_JUDGE_MODEL` in `.env.validator`, so it's just a matter of not
changing it.

**Anti-gaming stays server-side.** The operator's owner-api applies
trace integrity checks, honeytoken detection, and latency axis on top
of your LLM quality score when you submit. You don't see the honeytoken
URL list, the trace-gate heuristics, or the latency penalty curve —
those live inside the operator process.

## Costs

At current Chutes rates (Kimi-K2.5-TEE, ~1k tokens per judge call):

| Scenario | Est. judge cost |
|----------|-----------------|
| One 20-task run, 3 active miners | ~$0.15–$0.30 |
| Steady-state: 10 runs/month × 3 miners | ~$1.50–$3.00/month |

Your judge LLM spend is separate from miner LLM spend (miners pay their
own provider). The operator does not see or subsidize your judge bill.

## Health & monitoring

Each container exposes a `/healthz` on its HTTP port:

| Service | Port |
|---------|------|
| `validator-engine` | `18010` |
| `eiretes-judge` | `18095` |

```bash
curl http://localhost:18095/healthz   # judge
curl http://localhost:18010/healthz   # engine
```

`validator-engine` and `eiretes-judge` both emit Prometheus metrics on
`/metrics` — scrape into your own Prometheus if you want dashboards.

## Upgrading

```bash
docker compose -f docker-compose.validator.yml pull
docker compose -f docker-compose.validator.yml up -d
```

Subscribe to the operator's release channel (see subnet docs) for
protocol-change announcements.

## Troubleshooting

**`/v1/tasks/claim` returns 403.** Your hotkey isn't on the operator's
allow-list. Share your SS58 with the operator (step 3).

**`validator-engine` logs show "local judge call failed".** Either
`eiretes-judge` isn't running, your Chutes key is invalid, your key is
out of credits, or Chutes upstream is down. Check `eiretes-judge` logs.
When the judge call fails the validator submits `task_score=0` +
`judge_output=None` for that task — the owner-api accepts the
submission but your weight for that miner suffers.

**`weight-setting: run run-N already published`.** Normal — the
validator-engine's weight-setting loop tracks the last published run
and no-ops until a new run closes.

**`weight-setting: chain verification failed`.** Transient; the
extrinsic was accepted on-chain, the post-commit metagraph read just
raced with a recent update. Verify with `btcli wallet overview`.

**Miner invocations return 502 and tasks fail.** The operator's
owner-api couldn't reach the miner pod. Not a validator-side issue —
report to the operator.

**Judge takes ~30 seconds per task.** Normal for Kimi-K2.5-TEE thinking
mode. Bump `EIREL_JUDGE_TIMEOUT_SECONDS` if your network to Chutes is
slow.
