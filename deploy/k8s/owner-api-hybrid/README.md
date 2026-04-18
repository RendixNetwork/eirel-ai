# owner-api-hybrid — k3s manifests

Runs **owner-api** inside k3s while all other services (postgres, redis, tool
services, orchestrator, validator-engine, etc.) remain in docker-compose.

## Quick start

```bash
# 1. Copy and fill in secrets
cp secret.env.example secret.env
vi secret.env

# 2. Dry-run to validate
kubectl apply --dry-run=server -k .

# 3. Apply
kubectl apply -k .

# 4. Wait for rollout
kubectl -n eirel-control-plane rollout status deploy/owner-api --timeout=180s

# 5. Verify health
curl -s http://<SERVER_A_IP>:30020/healthz
```

## Architecture

- **Namespace:** `eirel-control-plane` (dedicated; not `eirel-system` or `eirel`)
- **NodePort:** `30020` -> container port `8000`
- **Node pinning:** `nodeSelector: eirel.dev/role=control-plane` (Server A)
- **Compose deps:** reached via `status.hostIP` downward API on published ports
  (15432, 16379, 18085-18092)
- **In-cluster config:** `EIREL_OWNER_KUBECONFIG_PATH=""` triggers
  `load_incluster_config()`, giving owner-api CoreDNS access to resolve
  `miner-<id>.eirel-miners.svc.cluster.local`

## SPOF warning

- **Wallet:** mounted via `hostPath` from Server A. Pod can only schedule on the
  labeled node. If that node is down, owner-api is down.
- **Artifact storage:** `emptyDir` — lost on pod restart. Archives are re-fetched
  from DB metadata on demand. Follow-up: PVC.

## Rollback

```bash
# Delete all resources created by this overlay
kubectl delete -k deploy/k8s/owner-api-hybrid/

# If callers were already cut over (PR 5), revert .env + docker-compose.yml
# and restart compose services
docker compose restart orchestrator execution-worker validator-engine \
  validator-engine-2 weight-setter
```
