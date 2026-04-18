# Running a validator on the EIREL subnet

This guide is for third parties who want to run a validator. If you're the
subnet operator, you want the root `README.md` + `docker-compose.yml`
instead — that stack hosts the owner-api and scoring services the whole
subnet depends on.

A validator runs a tiny two-service stack that polls the operator's
owner-api over HTTP, scores miner responses, and publishes weights
on-chain. Validators do **not** host the owner-api, judge, tool services,
or miner runtime. No local database is needed — both services query
Subtensor directly via the bittensor SDK.

## Prerequisites

- Docker + Docker Compose
- A Bittensor wallet you control (coldkey + hotkey) on the target network
- The operator's public `OWNER_API_URL` (ask on Discord / subnet docs)
- Your validator hotkey added to the subnet's `ValidatorRecord` allow-list
  (the operator adds it after you provide the SS58)

## 1. Create a Bittensor wallet

```bash
btcli wallet new_coldkey --wallet.name my-validator
btcli wallet new_hotkey  --wallet.name my-validator --wallet.hotkey v1
```

This writes keys to `~/.bittensor/wallets/my-validator/`.

## 2. Register on the subnet

```bash
# testnet example
btcli subnet register \
  --subtensor.network test \
  --netuid 144 \
  --wallet.name my-validator \
  --wallet.hotkey v1
```

Registration costs TAO — amount depends on the subnet. See `btcli help
subnet register` for the current rate.

## 3. Get whitelisted by the operator

Send your validator hotkey SS58 to the subnet operator. They add it to the
`ValidatorRecord` table; without this your `/v1/tasks/claim` calls will
return 403. You can retrieve your SS58 with:

```bash
btcli wallet list --wallet.name my-validator
```

## 4. Configure your validator

```bash
cp .env.validator.example .env.validator
```

Edit `.env.validator`:

| Variable | Value |
|----------|-------|
| `OPERATOR_OWNER_API_URL` | Public URL provided by the operator |
| `VALIDATOR_WALLET_NAME` | `my-validator` (from step 1) |
| `VALIDATOR_HOTKEY_NAME` | `v1` (from step 1) |
| `BITTENSOR_NETWORK` | `test` or `finney` |
| `BITTENSOR_NETUID` | subnet ID (testnet `144`, mainnet TBD) |
| `BITTENSOR_WALLETS_PATH` | Host path to your wallets dir (default `/root/.bittensor/wallets`) |

## 5. Start the stack

```bash
docker compose -f docker-compose.validator.yml up -d
```

Two containers come up:

- `validator-engine` — claims tasks from the operator, invokes miners,
  submits judged results back to the operator.
- `weight-setter` — publishes your computed weights on-chain on the
  subnet's scoring cadence.

Check logs:

```bash
docker compose -f docker-compose.validator.yml logs -f validator-engine
```

You should see tasks being claimed and evaluated once:

1. Your hotkey is whitelisted (step 3).
2. The operator has an open evaluation run with tasks queued.

## Health + monitoring

Each container exposes a `/healthz` on its HTTP port:

| Service | Port |
|---------|------|
| `validator-engine` | `18010` |
| `weight-setter` | `18012` |

```bash
curl http://localhost:18010/healthz
```

The validator also emits Prometheus metrics on each `/metrics` endpoint —
scrape those into your own Prometheus if you want dashboards.

## Upgrading

```bash
docker compose -f docker-compose.validator.yml pull
docker compose -f docker-compose.validator.yml up -d
```

Subscribing to the operator's release channel (see subnet docs) lets you
know when a new image ships with scoring / protocol changes that require
an upgrade.

## Troubleshooting

**`/v1/tasks/claim` returns 403.** Your hotkey isn't in `ValidatorRecord`.
Ask the operator to add it.

**`validator-engine` logs show `weight-setting: run run-N already published`.**
Normal — `weight-setter` tracks the last run it published and no-ops
until a new run closes.

**`weight-setting: chain verification failed`.** Transient — happens when
the local metagraph read races with a recent `set_weights`. The extrinsic
was accepted on-chain; verify with `btcli wallet overview`.

**Miner invocations return 502 and tasks fail.** The operator's owner-api
can't reach the miner pod. Not a validator-side issue — report to the
operator.
