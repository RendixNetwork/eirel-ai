# Eirel AI

Owner control-plane for the EIREL Bittensor subnet. Handles miner
submissions, runtime deployments, evaluation orchestration, score
aggregation, and chain publication.

> **Not the operator?**
> - Running a miner → [Miner Guide](docs/miner-guide.md)
> - Running a validator → [Validator Guide](docs/validator-guide.md)

## Architecture

```
                             Miners                Validators
                               |                       |
                               |  POST /v1/submissions |  POST /v1/tasks/claim
                               v                       v
                      +----------------------+---------------------+
                      |           owner-api (control plane)        |
                      +----------------------+---------------------+
                                 |                       |
                   ┌─────────────┼─────────────┐         |
                   v             v             v         |
              postgres        redis      artifact store  |
                                                         |
                            +---------------------+      |
                            | miner runtime pods  |<-----+  per-task invoke
                            | (docker / k3s)      |      |  through owner proxy
                            +---------------------+      |
                                                         |
             +------------+        +----------------+    |
             |provider-   |        | tool services  |    |
             |proxy (LLM) |        | web / x / ...  |    |
             +------------+        +----------------+    |
                                                         v
                                      validator-side: eiretes-judge
                                      + validator-engine
```

Each validator runs its own `eiretes-judge` sidecar and calls the
operator's owner-api to claim evaluation tasks, invoke miners through
the operator's HTTP proxy, score locally, and submit results. The
operator does **not** run a judge. See
[docs/validator-guide.md](docs/validator-guide.md) for the validator
stack.

## Operator services

Started via `docker-compose.yml` at the repo root.

**Core subnet** (required for operating the subnet):

| Service | Purpose |
|---------|---------|
| `owner-api` | Submission lifecycle, deployment management, run orchestration, score aggregation, chain-publication readiness checks |
| `metagraph-listener` | Syncs registered neurons from Bittensor chain state; gates miner submission acceptance |
| `provider-proxy` | LLM provider fan-out with per-run USD budget enforcement |
| `web-search-tool-service` | Web search tool exposed to miner agents |
| `url-fetch-tool-service` | URL fetch / read tool exposed to miner agents |
| `sandbox-tool-service` | Server-side Python sandbox for verifiable computation |
| `rag-tool-service` | Indexes per-run document corpora; serves `rag.retrieve` for `rag_required` tasks |
| `postgres`, `redis` | Storage + coordination |

**Consumer-facing product** (optional — only if you run the subnet's end-user chat product on top of the subnet):

| Service | Purpose |
|---------|---------|
| `orchestrator` | DAG-backed request coordination for the consumer API |
| `execution-worker` | Async task consumer (Redis stream) |
| `consumer-chat-api` | User-facing chat/session API |

**Monitoring** (optional compose profile):

| Service | Purpose |
|---------|---------|
| `prometheus`, `alertmanager`, `grafana` | Dashboards + alerts |

Services that used to live here but now run validator-side:
`eiretes-judge`, `validator-engine`. See `docker-compose.validator.yml`.

## Launch family

The subnet launches with a single family:

| Family | Description |
|--------|-------------|
| **general_chat** | Multi-turn conversational assistant across `instant` and `thinking` modes. Backed by owner-routed tool services: web search, URL fetch, a Python sandbox for verifiable computation, and RAG retrieval over per-run document corpora. |

Additional families are defined on the roadmap and will activate in
future releases. Enforcement is gated by `EIREL_LAUNCH_MODE=true`.

## Submission / evaluation lifecycle

1. Miners submit a Docker-packaged agent. Fee (0.1 TAO) is verified on
   chain; signature is verified via Bittensor.
2. owner-api builds the image and places a pod on subnet-owned runtime
   (docker or k3s, selectable via `OWNER_RUNTIME_BACKEND`).
3. At each run boundary, the current deployment set is frozen into an
   `EpochTargetSnapshot`; task rows are seeded into
   `miner_evaluation_tasks` — one row per `(miner, task)`. Bundles are
   sampled across a capability × domain matrix so coverage stays
   broad. RAG corpora referenced by `rag_required` tasks are indexed
   into `rag-tool-service` once per run.
4. Validators claim tasks in batches (with lease TTL), invoke the miner
   through the owner proxy, enrich each task at task-claim time via a
   3-oracle layer (OpenAI + Gemini + Grok + Chutes reconciler) for
   `expected_claims`, and judge locally with a multiplicative composite
   that gates on `grounded_correctness` + `instruction_safety` and
   incorporates server-attested tool-call attestation from the
   orchestrator's ledger.
5. Per-(run, miner, task) `EvalFeedback` rows are persisted server-side
   when validators submit results; miners read their own rows via a
   hotkey-signed endpoint.
6. On run close, per-miner summaries roll up into an
   `AggregateFamilyScoreSnapshot` and `DeploymentScoreRecord` rows drive
   the carryover / weight-publication path. After close, every
   submission archive scored in that run becomes publicly downloadable
   from the leaderboard.

## Local development

### Prerequisites

- Python >= 3.12
- PostgreSQL 16 (local or via compose)
- Redis 7

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../eirel
pip install -e ../eiretes
pip install -e .[dev]
eirel-ai migrate
pytest tests/ -q
```

### Docker Compose

```bash
cp .env.compose.example .env.compose

docker compose --env-file .env.compose up -d
docker compose --env-file .env.compose run --rm api-gateway eirel-ai migrate
```

Optional monitoring:

```bash
docker compose --env-file .env.compose --profile monitoring up -d prometheus grafana
```

- Prometheus: `http://127.0.0.1:19090`
- Grafana: `http://127.0.0.1:13000`

Tear down:

```bash
docker compose --env-file .env.compose down -v
```

## Kubernetes deployment

```bash
cp deploy/k8s/overlays/staging/secret.env.example deploy/k8s/overlays/staging/secret.env
kubectl apply -k deploy/k8s/overlays/staging
kubectl apply -f deploy/k8s/overlays/staging/jobs/db-migrate.yaml
```

Staging runs in-cluster Postgres/Redis with an HPA and Kubernetes-native
miner runtime reconciliation. Production expects external managed
Postgres/Redis/S3 — see
[`deploy/k8s/overlays/production/README.md`](deploy/k8s/overlays/production/README.md).

## Configuration

| Category | Variables |
|----------|-----------|
| **Database** | `DATABASE_URL`, `REDIS_URL` |
| **Bittensor** | `BITTENSOR_NETWORK`, `BITTENSOR_NETUID`, `EIREL_OWNER_WALLET_NAME`, `EIREL_OWNER_HOTKEY_NAME` |
| **Submission fee** | `EIREL_SUBMISSION_TREASURY_ADDRESS` (empty disables), `EIREL_SUBMISSION_FEE_TAO` |
| **Runtime** | `OWNER_RUNTIME_BACKEND` (`docker` / `kubernetes`) |
| **Launch mode** | `EIREL_LAUNCH_MODE` (`true` → restrict to `general_chat`) |
| **Internal auth** | `EIREL_INTERNAL_SERVICE_TOKEN` (orchestrator ↔ owner-api ↔ tool platforms; **not** distributed to validators) |
| **Provider keys** | `EIREL_PROVIDER_OPENAI_API_KEY`, `EIREL_PROVIDER_ANTHROPIC_API_KEY`, `EIREL_PROVIDER_OPENROUTER_API_KEY`, `EIREL_PROVIDER_CHUTES_API_KEY` |
| **RAG tool** | `EIREL_RAG_TOOL_URL`, `EIREL_RAG_TOOL_API_TOKEN`, `EIREL_RAG_EMBEDDING_API_KEY` (defaults to `OPENAI_API_KEY`) |
| **Storage** | `OBJECT_STORAGE_BACKEND` (`filesystem` / `s3`), `ARTIFACT_STORAGE_ROOT` |
| **Datasets** | `EIREL_OWNER_DATASET_ROOT_PATH` |
| **Tracing** | `OTEL_*` |

## Entrypoints

**Long-running services** (one container / one k8s Deployment per process):

```bash
owner-api                # Control plane
metagraph-listener       # Chain state sync
eirel-provider-proxy     # LLM provider fan-out
web-search-tool-service  # Tool: web search
url-fetch-tool-service   # Tool: URL fetch / read
sandbox-tool-service     # Tool: Python sandbox
rag-tool-service         # Tool: RAG retrieval over per-run corpora
# consumer product (optional):
orchestrator
execution-worker
consumer-chat-api
# validator-side (shipped in same image, started by docker-compose.validator.yml):
validator-engine
```

**Operator CLI** — one entrypoint with subcommands:

```bash
eirel-ai migrate         # Run pending DB migrations
eirel-ai admin whoami    # Show signed identity
eirel-ai admin runs list
eirel-ai admin runs current
eirel-ai admin submissions list [--limit N]
eirel-ai admin deployments list [--limit N]
eirel-ai admin validators {add,remove,list,enable,disable} ...
eirel-ai admin neurons list
```

## Security

- Bittensor hotkey signatures + replay protection on all authenticated
  endpoints. Validators authenticate every owner-api call with their
  wallet hotkey (claim, result, ledger read, feedback read); miners do
  the same for submission upload + their own per-task feedback. The
  operator's `EIREL_INTERNAL_SERVICE_TOKEN` is reserved for inter-service
  calls within the operator stack (orchestrator ↔ owner-api ↔ tool
  platforms) — it is **not** distributed to 3rd-party validators.
- Submission archives are private during the run that scored them;
  after the run closes, archives become publicly downloadable from
  the leaderboard (operators can audit; competitors can study).
- Server-attested tool-call ledger: validators score `tool_attestation`
  from owner-side `OrchestratorToolCallLog` rows rather than trusting
  miner-emitted trace frames.
- Circuit breakers for miner-pod invocation and chain extrinsic
  submission.
- Pluggable artifact storage (filesystem for dev, S3 for production).
- Per-run USD budget enforcement at the provider-proxy layer.

## License

MIT. See [LICENSE](LICENSE).
