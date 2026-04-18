# kube-state-metrics for Eirel

Cluster-wide pod / deployment / namespace lifecycle metrics scraped by the
Prometheus instance running in `docker-compose` on Server A.

## Scope

Limited to the Eirel namespaces:

- `eirel-miners` — miner runtime pods
- `eirel-control-plane` — owner-api
- `eirel-system` — shared infra (secrets, etc.)

Other namespaces (`kube-system`, `calico-*`, `tigera-operator`) are
intentionally excluded via the `--namespaces` arg on the deployment to keep
series count low.

## Install

```bash
kubectl apply -k deploy/k8s/kube-state-metrics
kubectl -n eirel-system rollout status deploy/kube-state-metrics
```

## Verify

```bash
# From a node that can reach the k3s NodePort:
curl -s host:30080/metrics | head

# From the prometheus container:
docker compose --profile monitoring exec prometheus \
  wget -qO- http://host.docker.internal:30080/metrics | head
```

Prometheus scrape target is in `monitoring/prometheus/prometheus.yml`
under `job_name: kube_state_metrics`.

## Alerts

See `monitoring/prometheus/alerts/eirel-kube-rules.yml` for
`EirelMinerPodCrashLooping`, `EirelMinerPodStuckPending`,
`EirelPodImagePullBackOff`, and `EirelControlPlanePodNotReady`.
