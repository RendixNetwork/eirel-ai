# owner-api-hybrid — k3s manifests

Runs **owner-api** inside k3s while all other services (postgres, redis, tool
services, orchestrator, validator-engine, etc.) remain in docker-compose on the
same host. The two stacks talk over the node's host IP on the compose-published
ports.

This overlay is deployment-agnostic — environment-specific values (chain,
wallets, images, endpoints) are injected via a gitignored `configmap.env`, so
the committed manifests are safe to share and reproducible from a fresh clone.

---

## First-time deploy

```bash
cd deploy/k8s/owner-api-hybrid

# 1. Copy the two env-file templates and fill them in.
cp configmap.env.example configmap.env
cp secret.env.example    secret.env
$EDITOR configmap.env secret.env

# 2. Validate the rendered manifests (server-side dry-run).
kubectl apply --dry-run=server -k .

# 3. Apply.
kubectl apply -k .

# 4. Wait for rollout + verify health.
kubectl -n eirel-control-plane rollout status deploy/owner-api --timeout=180s
curl -s http://<NODE_IP>:30020/healthz
```

### What each file feeds

| File | Purpose | Committed? |
|---|---|---|
| `namespace.yaml`, `serviceaccount.yaml`, `clusterrole.yaml`, `service.yaml`, `networkpolicy.yaml` | Static cluster resources | ✅ |
| `deployment.yaml` | Pod spec, probes, mounts, inline env wiring | ✅ |
| `configmap.yaml` | Deployment-agnostic defaults (runtime tuning, tool backend toggles, label keys) | ✅ |
| `configmap.env.example` | Template listing every host-specific key | ✅ |
| `configmap.env` | Your filled values — merged onto `configmap.yaml` at apply time | ❌ gitignored |
| `secret.env.example` | Template for sensitive values | ✅ |
| `secret.env` | Your filled sensitive values | ❌ gitignored |
| `kustomization.yaml` | Wires it all together | ✅ |

---

## Updating a running deployment

```bash
# Code change: pull the newer image
kubectl -n eirel-control-plane rollout restart deployment/owner-api
kubectl -n eirel-control-plane rollout status deployment/owner-api

# Config change: edit configmap.env, then re-apply
$EDITOR configmap.env
kubectl apply -k .
# ConfigMap change doesn't auto-restart pods — force it:
kubectl -n eirel-control-plane rollout restart deployment/owner-api

# Image pin: change the tag
kustomize edit set image rendixnetwork/eirel-ai=rendixnetwork/eirel-ai:v0.2.0
kubectl apply -k .
# (kustomize edit modifies kustomization.yaml in place)

# Rollback
kubectl -n eirel-control-plane rollout undo deployment/owner-api
```

---

## Architecture

- **Namespace**: `eirel-control-plane` (dedicated — not `eirel-system` or `eirel`).
- **NodePort**: `30020` → container port `8000`.
- **Node pinning**: `nodeSelector: eirel.dev/role=control-plane`. Label the
  intended node once: `kubectl label node <name> eirel.dev/role=control-plane`.
- **Compose-side deps**: reached via `status.hostIP` downward API on the
  compose-published ports (25432 postgres, 16379 redis, 18085–18092 tool
  services, 18092 provider-proxy).
- **Bittensor wallet**: `hostPath` mount from the node's
  `~/.bittensor/wallets/`. The owner-api loads the hotkey at startup to derive
  `owner_hotkey_ss58` for admin-signature verification.
- **In-cluster config**: leaving `EIREL_OWNER_KUBECONFIG_PATH=""` triggers
  `load_incluster_config()` so owner-api can manage miner pods via k8s RBAC.

---

## Tear-down / rollback

```bash
# Remove everything this overlay creates (preserves namespace unless pinned)
kubectl delete -k .

# If callers were already cut over to k3s, revert .env + docker-compose.yml
# and restart compose services:
docker compose restart orchestrator execution-worker validator-engine
```

---

## Known limitations (single point of failure for current deploy)

- **Wallet**: mounted via `hostPath` from the labeled node. If that node is
  down, owner-api is down.
- **Artifact storage**: `emptyDir` — lost on pod restart. Archives are
  re-fetched from DB metadata on demand. Follow-up: migrate to a PVC.

Both are acceptable for a testnet / single-operator deployment. Revisit when
moving to multi-node (Mode B — see `../base/` + `../overlays/`).
