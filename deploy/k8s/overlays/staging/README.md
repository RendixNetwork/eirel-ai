# Kubernetes Staging

This overlay is the production-like staging topology for the managed EIREL control plane:

- `owner-api`
- `api-gateway`
- `execution-worker`
- `consumer-chat-api`
- `validator-engine`
- `metagraph-listener`
- `weight-setter`
- `provider-proxy`
- in-cluster `postgres` and `redis`

The staging request path is:

- `consumer-chat-api -> api-gateway -> execution-worker -> owner-api -> managed miner revision`

The staging validator path is:

- `validator-engine -> owner-api frozen targets -> owner-api epoch-pinned miner revision`

Managed miner revisions are materialized as Kubernetes `Deployment + Service` resources by the owner runtime controller path. The staging overlay no longer depends on Docker socket control.

## Apply staging

1. Create the secret env file:

```bash
cp deploy/k8s/overlays/staging/secret.env.example deploy/k8s/overlays/staging/secret.env
```

2. Edit `deploy/k8s/overlays/staging/secret.env` with real values.

Required fixture roots for family benchmarking in staging:

- `EIREL_OWNER_DATASET_ROOT_PATH=/app/eirel-ai/data/owner_datasets/families`
- `EIREL_CALIBRATION_FIXTURES_ROOT_PATH=/app/eirel-ai/data/calibration`
- `EIREL_WORKFLOW_CORPUS_ROOT_PATH=/app/eirel-ai/data/workflow_corpus`

3. Apply the staging stack:

```bash
kubectl apply -k deploy/k8s/overlays/staging
```

4. Run migrations:

```bash
kubectl apply -f deploy/k8s/overlays/staging/jobs/db-migrate.yaml
kubectl wait --for=condition=complete --timeout=5m job/eirel-db-migrate -n eirel
kubectl logs job/eirel-db-migrate -n eirel
```

5. Run the cluster smoke:

```bash
kubectl apply -f deploy/k8s/overlays/staging/jobs/cluster-smoke.yaml
kubectl wait --for=condition=complete --timeout=10m job/eirel-cluster-smoke -n eirel
kubectl logs job/eirel-cluster-smoke -n eirel
```

## Notes

- The smoke job uses the in-cluster service DNS names and submits the bundled sample miner from `/app/eirel/examples/sample_miner`.
- The ingress host is `chat.staging.eirel.local`; change it to your real staging host before use.
- The secret file is intentionally excluded from version control.
- The owner runtime backend is Kubernetes-native in this overlay; make sure the `owner-runtime-controller` service account has the RBAC resources from the base manifests.
- The provider proxy is part of the staging stack and should be configured with staging provider credentials or explicit empty-provider behavior if you only want local smoke coverage.

## Observability

The staging observability assets live in:

- `monitoring/prometheus/prometheus.yml`
- `monitoring/prometheus/alerts/eirel-staging-rules.yml`
- `monitoring/grafana/dashboards/eirel-staging-text-ga.json`
- `docs/runbooks/`

The key staging alerts cover:

- service availability for `owner-api`, `api-gateway`, `execution-worker`, `consumer-chat-api`, `metagraph-listener`, and `weight-setter`
- provider proxy availability and quota rejection pressure
- queued task backlog and worker failure growth
- unhealthy managed deployments
- stuck epoch snapshots or aggregate snapshots
- metagraph sync failure and validator quorum risk

Use the matching runbook path embedded in each alert annotation when paging on staging incidents.
