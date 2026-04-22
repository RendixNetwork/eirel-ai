# Kubernetes deployments

The EIREL control plane ships with **two supported deployment modes**. Pick
one; they're not meant to run simultaneously.

## Mode A — Hybrid (`owner-api-hybrid/`)

**Topology**: owner-api runs in k3s; every other service (postgres, redis,
tool services, orchestrator, validator-engine, weight-setter, provider-proxy,
consumer-chat-api) runs in docker-compose on the same host.

**Use when**:
- Single-host deployment (testnet, dev, small production)
- You already run compose for the non-k8s stack
- You want a narrow k8s footprint

**Deploy**:
```bash
cd deploy/k8s/owner-api-hybrid
cp configmap.env.example configmap.env
cp secret.env.example    secret.env
$EDITOR configmap.env secret.env
kubectl apply -k .
```

See `owner-api-hybrid/README.md` for details.

---

## Mode B — Full k8s (`base/` + `overlays/{staging,production}/`)

**Topology**: every control-plane service runs in k8s. No compose.

**Use when**:
- Multi-node cluster (real production)
- External managed Postgres / Redis / S3 are available (production)
- You want horizontal autoscaling, PodDisruptionBudgets, etc.

**Deploy (staging)**:
```bash
cd deploy/k8s/overlays/staging
cp configmap.env.example configmap.env
cp secret.env.example    secret.env
$EDITOR configmap.env secret.env
kubectl apply -k .
```

**Deploy (production)**:
```bash
cd deploy/k8s/overlays/production
cp configmap.env.example configmap.env
cp secret.env.example    secret.env
$EDITOR configmap.env secret.env   # includes DATABASE_URL, REDIS_URL, S3 creds
kubectl apply -k .
```

See the overlay READMEs for specifics.

---

## Shared assets

| Path | Purpose |
|---|---|
| `base/` | Mode B base manifests (9 Deployments + Services + PVC + RBAC + ConfigMap) |
| `overlays/staging/` | Mode B staging overlay (in-cluster postgres+redis, HPAs, ingress) |
| `overlays/production/` | Mode B production overlay (external managed deps, ingress) |
| `owner-api-hybrid/` | Mode A manifests |
| `jobs/` | Shared Jobs (`db-migrate`) referenced by both overlays |
| `kube-state-metrics/` | Optional monitoring addon (namespace-agnostic) |

---

## Image management

All Deployments and Jobs use the **monolith image**
`rendixnetwork/eirel-ai:latest`. Every service selects its entrypoint via the
`command:` field, matching the pattern in `docker-compose.yml`.

To retag across everything in an overlay:
```bash
cd deploy/k8s/overlays/staging   # or production, or owner-api-hybrid
kustomize edit set image rendixnetwork/eirel-ai=rendixnetwork/eirel-ai:v0.2.0
```

To point at a different registry (e.g. GHCR private, internal harbor):
```bash
kustomize edit set image rendixnetwork/eirel-ai=ghcr.io/myorg/eirel-ai:v0.2.0
```

---

## Validation before apply

```bash
# Always build first
kubectl kustomize deploy/k8s/overlays/staging | less

# Client-side schema validation
kubectl apply --dry-run=client -k deploy/k8s/overlays/staging

# Server-side dry-run (requires target namespace to exist)
kubectl create namespace eirel --dry-run=client -o yaml | kubectl apply -f -
kubectl apply --dry-run=server -k deploy/k8s/overlays/staging
```
