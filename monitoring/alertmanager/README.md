# Alertmanager

Routes the ~30 Prometheus alert rules to webhook receivers, with severity-based
routing and inhibit rules to suppress derived alerts when a root-cause alert is
already firing.

## Webhook configuration

Set these environment variables in your compose `.env` file:

```bash
ALERTMANAGER_DEFAULT_WEBHOOK_URL=https://hooks.slack.com/services/T.../B.../xxx
ALERTMANAGER_CRITICAL_WEBHOOK_URL=https://events.pagerduty.com/generic/2010-04-15/create_event.json
```

Any webhook-compatible endpoint works (Slack, Discord, PagerDuty, Opsgenie,
custom HTTP). When unset, the URLs fall back to localhost noop endpoints, so
Alertmanager starts cleanly without external integrations.

The config file (`alertmanager.yml`) contains placeholder URLs that are
substituted at container startup via `sed`. The docker-compose `environment`
block passes the env vars into the container, where they replace the
placeholders before Alertmanager loads the config.

## Routing

| Severity   | Receiver           | group_wait | repeat_interval |
|------------|--------------------|------------|-----------------|
| `critical` | `critical-webhook` | 10s        | 1h              |
| `warning`  | `default-webhook`  | 2m         | 12h             |
| (default)  | `default-webhook`  | 30s        | 4h              |

Alerts are grouped by `[alertname, service, cluster]`.

## Inhibit rules

Three inhibit rules prevent alert storms when a root-cause service is down:

1. **OwnerApiDown suppresses derived alerts** — when `EirelOwnerApiDown` fires,
   the following alerts are inhibited (same `cluster`):
   `EirelUnhealthyDeploymentsPresent`, `EirelEpochSnapshotsStuckOpen`,
   `EirelAggregateSnapshotsPendingTooLong`, `EirelWorkflowDeadLetterGrowing`,
   `EirelNoRuntimeNodesAvailable`.

2. **RuntimeNodeDown suppresses NodeExporterDown** — when
   `EirelRuntimeNodeDown` fires, `EirelNodeExporterDown` is inhibited for the
   same `node`.

3. **MetagraphListenerDown suppresses chain-derived alerts** — when
   `EirelMetagraphListenerDown` fires, `EirelValidatorSetTooSmall` and
   `EirelMetagraphSyncFailing` are inhibited (same `cluster`).
