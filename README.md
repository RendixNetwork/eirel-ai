# Eirel AI

The owner control-plane and validator infrastructure for the EIREL Bittensor subnet. Manages the full lifecycle of miner competition: submission, deployment, benchmarking, scoring, and on-chain weight publication.

## Architecture

EIREL AI operates as a set of cooperating microservices that together form the subnet's validation and orchestration layer.

```
                              User Requests
                                   |
                            +------v------+
                            | API Gateway |  auth, rate limiting
                            +------+------+
                                   |
                  +----------------+----------------+
                  |                |                 |
           +------v------+  +-----v------+  +-------v--------+
           |Task Service  |  |Classifier  |  |Consumer API    |
           |normalize     |  |family route|  |chat sessions   |
           +------+------+  +-----+------+  +-------+--------+
                  |                |                 |
                  +-------+-------+                 |
                          |                         |
                   +------v------+                  |
                   |DAG Executor |  build, dispatch, fallback
                   +------+------+
                          |
               +----------+----------+
               |                     |
        +------v------+      +------v--------+
        |Context Svc  |      |Execution      |
        |output pack  |      |Worker (queue) |
        +-------------+      +---------------+
                                     |
                              +------v------+
                              |Miner Agents |  (managed deployments)
                              +-------------+

        Validation Pipeline (parallel)

        +----------------+     +------------------+     +---------------+
        |Validator Engine|---->|Weight Setter     |---->|Chain (TAO)    |
        |benchmark score |     |normalize, submit |     |weight commits |
        +-------+--------+     +------------------+     +---------------+
                |
        +-------v----------+
        |Metagraph Listener|  sync validators/miners from chain
        +------------------+

        Owner Control Plane

        +-------------+
        |Owner API    |  submissions, deployments, epochs, scoring
        +-------------+
```

### Services

| Service | Purpose |
|---------|---------|
| **owner_api** | Miner submission, deployment lifecycle, epoch coordination, score aggregation, active-fleet registry |
| **api_gateway** | Public HTTP ingress with API-key auth, rate limiting, and task orchestration |
| **task_service** | Raw user input normalization, modality detection |
| **classifier_service** | Family routing via keyword matching or optional LLM-based planning |
| **dag_executor** | Builds execution DAGs from routing plans, dispatches to miners, handles fallback |
| **execution_worker** | Async Redis stream consumer for queued task execution |
| **context_service** | Output packing and compression for downstream consumption |
| **attribution_service** | Dependency-depth attribution and query-volume accounting |
| **consumer_api** | User-facing chat/session API with traffic logging for evaluation sampling |
| **validator_engine** | Per-family benchmark scoring against owner-frozen evaluation bundles |
| **weight_setter** | Score normalization and on-chain weight payload submission with Bittensor signing |
| **metagraph_listener** | Periodic sync of validator/miner registry from Bittensor chain state |

### Shared Modules

| Module | Purpose |
|--------|---------|
| `shared/models.py` | TaskObject, RoutingPlan, ExecutionDAG, ExecutionResult, AttributionRecord |
| `shared/specialist_contracts.py` | Per-family output contracts: required fields, types, tools, latency budgets |
| `common/models.py` | SQLAlchemy ORM: validators, submissions, deployments, scores, epochs |
| `common/config.py` | Environment-based Settings dataclass |
| `common/security.py` | API-key auth, request signature verification, replay protection |
| `common/execution_store.py` | Task result persistence (PostgreSQL) |
| `common/artifacts.py` | Pluggable artifact storage (filesystem dev, S3 production) |
| `common/circuit_breaker.py` | Failure isolation for miner and chain calls |
| `common/tracing.py` | OpenTelemetry integration |

## Launch Families

At subnet launch, three families are active. Set `EIREL_LAUNCH_MODE=true` to enforce this:

| Family | Weight | Description |
|--------|--------|-------------|
| **analyst** | 0.45 | Research, reasoning, evidence-grounded synthesis |
| **builder** | 0.30 | Autonomous multi-day project implementation from specs |
| **verifier** | 0.25 | Grounded inspection, auditing, defect detection |

Five additional families (browser, data, media, memory, planner) are defined but inactive during launch. The `ensure_active_family_id()` function in `eirel` rejects non-launch families when launch mode is enabled.

## Family Benchmark Model

Competition is family-native:

1. Miners submit code for **one family** per submission
2. Owner deploys selected revisions on subnet-owned infrastructure (miners submit Docker archives)
3. Owner freezes an immutable `FamilyEvaluationBundle` per `(run_id, family_id)` — tasks cannot change after publication
4. Validators score miners against frozen bundles via `family_protocol`
5. Hard-block **calibration gates** must pass before promotion (consistency + recent stability)
6. **One winner per family** is promoted at epoch boundaries
7. Weights are submitted to chain based on normalized scores

### Builder Long-Running Execution

Builder agents work autonomously for up to 48 hours. The execution flow uses a webhook callback protocol:

```
Owner                           Miner Agent
  |                                  |
  |--- POST /v1/agent/infer -------->|  (with callback_url, execution_mode="autonomous_project")
  |<-- 200 {status: "deferred"} -----|
  |                                  |
  |<-- POST /v1/callbacks/{task_id} -|  (event: "checkpoint", progress: {...})
  |<-- POST /v1/callbacks/{task_id} -|  (event: "checkpoint", progress: {...})
  |<-- POST /v1/callbacks/{task_id} -|  (event: "completed", output: {...})
  |                                  |
  |--- Score result independently ---|
```

Owner independently verifies results by running hidden test suites and static analysis against the agent's delivered code.

## Local Development

### Prerequisites

- Python >= 3.12
- PostgreSQL 16
- Redis 7

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../eirel
pip install -e ../eiretes
pip install -e .[dev]
db-migrate
pytest tests/test_control_plane_reset.py -q
```

### Judge service contract

`eirel-ai` talks to the `eiretes-judge` sidecar over HTTP. The contract is
owned by `eiretes` and consumed here via `core.judge_client.JudgeServiceClient`.

| Endpoint | Purpose |
|----------|---------|
| `GET /healthz` | liveness; returns `judge_model`, `rubric_version`. Use `JudgeServiceClient.healthcheck(expected_rubric_version=...)` at lifespan startup so the first judge call isn't also the first network probe. |
| `GET /v1/catalog` | live rubric catalog (`family_id → {rubric_name, judge_mode, judge_weight, dimensions, profiles?}`). Consumed lazily by `benchmark/_rubric_support.py::get_supported_families()`, which caches on first use and falls back to `FALLBACK_SUPPORTED_FAMILIES` when the service is unreachable. |
| `POST /v1/judge` | score a single response. Request shape: `{family_id, prompt, response_excerpt, rubric_variant?, judge_profile?}`. Invalid `family_id` returns HTTP 400. |
| `POST /v1/extract-claims` | batched research-claim extraction from a markdown report. |

**Unified excerpt builder.** Both `benchmark/_orchestration.py` and
`validator/engine.py` construct judge excerpts via
`benchmark._judge.build_judge_excerpt(family_id=..., run=...)` — the same
miner output must not score differently depending on which pipeline invoked
the judge.

**Rubric versioning.** When the eiretes rubric version changes, bump
`EIREL_JUDGE_RUBRIC_VERSION` in both services. `healthcheck()` raises
`RuntimeError` on drift, and `get_supported_families()` re-fetches the
catalog on each miss. Set `EIREL_DISABLE_DYNAMIC_CATALOG=1` to pin the
bundled snapshot (used by tests that don't want to mock the HTTP layer).

### Docker Compose

```bash
# Copy environment files
cp .env.compose.example .env.compose
cp .env.host.example .env.host

# Start the stack
docker compose --env-file .env.compose up -d \
  postgres redis provider-proxy owner-api api-gateway execution-worker \
  consumer-chat-api validator-engine metagraph-listener weight-setter

# Run migrations
set -a && . ./.env.host && set +a
db-migrate

# Run smoke tests
eirel-compose-smoke

# Tear down
docker compose --env-file .env.compose down -v
```

### Monitoring (Optional)

```bash
docker compose --env-file .env.compose --profile monitoring up -d prometheus grafana
```

- Prometheus: `http://127.0.0.1:19090`
- Grafana: `http://127.0.0.1:13000`
- Dashboard: `EIREL Staging Text GA`

## Kubernetes Deployment

### Staging

```bash
cp deploy/k8s/overlays/staging/secret.env.example deploy/k8s/overlays/staging/secret.env
kubectl apply -k deploy/k8s/overlays/staging
kubectl apply -f deploy/k8s/overlays/staging/jobs/db-migrate.yaml
kubectl apply -f deploy/k8s/overlays/staging/jobs/cluster-smoke.yaml
```

Staging includes in-cluster PostgreSQL/Redis, public ingress, HPA, PDB, and Kubernetes-native miner runtime reconciliation.

### Production

```bash
kubectl apply -k deploy/k8s/overlays/production
```

Production assumes external managed PostgreSQL, Redis, and S3-compatible storage. See [`deploy/k8s/overlays/production/README.md`](./deploy/k8s/overlays/production/README.md).

## Configuration

### Key Environment Variables

| Category | Variables |
|----------|-----------|
| **Database** | `DATABASE_URL`, `REDIS_URL` |
| **Endpoints** | `OWNER_API_URL`, `API_GATEWAY_URL`, `EXECUTION_WORKER_URL`, etc. |
| **Auth** | `GATEWAY_API_KEYS`, `EIREL_INTERNAL_SERVICE_TOKEN` |
| **Storage** | `OBJECT_STORAGE_BACKEND` (filesystem/s3), `ARTIFACT_STORAGE_ROOT` |
| **Runtime** | `OWNER_RUNTIME_BACKEND` (docker/kubernetes) |
| **Datasets** | `EIREL_OWNER_DATASET_ROOT_PATH` |
| **Bittensor** | `EIREL_VALIDATOR_MNEMONIC`, hotkey/wallet config |
| **Launch** | `EIREL_LAUNCH_MODE` (true/false) |
| **Tracing** | `OTEL_*` variables for OpenTelemetry |

## CLI Entrypoints

```bash
owner-api                  # Owner control plane
api-gateway                # Public HTTP ingress
task-service               # Task normalization
classifier-service         # Family routing
dag-executor               # DAG orchestration
context-service            # Output packing
attribution-service        # Attribution calculation
consumer-chat-api          # User chat API
validator-engine           # Benchmark scoring
metagraph-listener         # Chain state sync
weight-setter              # Weight submission
execution-worker           # Async task consumer
db-migrate                 # Database migrations
eirel-staging-validate     # Staging validation
eirel-compose-smoke        # Docker Compose smoke test
eirel-ops-snapshot         # Operational snapshot
```

## Owner Datasets

Production evaluation tasks live in `owner_datasets/families/`:

- `analyst.json` — Research and reasoning tasks
- `builder.json` — Autonomous project implementation tasks
- `verifier.json` — Grounded inspection tasks

These are owner-frozen and cannot be influenced by miners. Public example fixtures are in `eiretes/examples/`.

## Testing

```bash
# Full test suite (230 tests)
pytest tests/ -q \
  --ignore=tests/test_compose_smoke.py \
  --ignore=tests/test_docker_runtime.py \
  --ignore=tests/test_e2e_smoke.py \
  --ignore=tests/test_live_research_smoke.py \
  --ignore=tests/test_ops_snapshot.py \
  --ignore=tests/test_sandbox_executor.py

# Specific modules
pytest tests/test_owner_api.py -q           # Owner API coverage
pytest tests/test_validator_engine_epoch.py  # Epoch scoring
pytest tests/test_control_plane_reset.py     # State management
```

## Security

- **API-key auth** on all public gateway endpoints
- **Internal service tokens** for inter-service communication
- **Request signatures** with replay protection for validator/miner calls
- **Rate limiting** with sliding window per principal
- **Circuit breakers** for miner call isolation and chain submission
- **Resume token signing** (HMAC-SHA256) for multi-turn state integrity

## License

Proprietary. See LICENSE for details.
