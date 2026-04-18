# Kubernetes Production

This overlay is the first real production target for the managed EIREL subnet.

Production assumptions:

- single-region HA Kubernetes
- external managed Postgres
- external managed Redis
- external S3-compatible artifact storage
- Kubernetes-native managed miner runtime
- provider proxy required for miner/provider access
- one miner submission maps to exactly one of the 4 families
- validators score miners through control-plane benchmark workflows
- Top-K eligible deployments per group remain active; non-Top-K eligible revisions drain to cold

## Apply production

1. Copy and fill the secret file:

```bash
cp deploy/k8s/overlays/production/secret.env.example deploy/k8s/overlays/production/secret.env
```

2. Review the ingress host and TLS secret in `ingress.yaml`.

3. Apply:

```bash
kubectl apply -k deploy/k8s/overlays/production
```

4. Run migrations with the production secret values available to the job or an equivalent one-off admin task.

5. Run the production smoke job after migrations and ingress are ready.

## Notes

- This overlay expects the base manifests to run with the Kubernetes-native runtime backend.
- `owner-api` requires RBAC to create and delete managed miner Deployments and Services in-cluster.
- Provider proxy credentials are production secrets and should not be committed.
- Object-storage credentials in `secret.env` back the S3-compatible artifact store used for media outputs and retained scoring evidence.
- Signed write paths rely on the internal service token plus replay-protected validator/miner signatures.
