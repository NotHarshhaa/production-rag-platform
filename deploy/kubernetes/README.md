# Kubernetes deployment

Example manifests for running the platform in-cluster. They are a starting
point — adjust replicas, storage classes, and image references for your
cluster.

## What's included

| File | Purpose |
|---|---|
| `namespace.yaml` | Dedicated `rag-platform` namespace |
| `configmap.yaml` | Non-secret API configuration (Qdrant/Postgres URLs, auth on, background ingestion) |
| `secret.yaml` | Template for `API_KEYS` and `LLM_API_KEY` — create real secrets out-of-band |
| `postgres.yaml` | StatefulSet + Service for document metadata |
| `qdrant.yaml` | StatefulSet + Service for vectors |
| `api.yaml` | API Deployment (2 replicas), Service, and CPU-based HPA |
| `ingress.yaml` | Optional nginx ingress with SSE-friendly settings |

## Deploy

```bash
# 1. Build and push the image
docker build -t ghcr.io/your-org/production-rag-platform:1.0.0 .
docker push ghcr.io/your-org/production-rag-platform:1.0.0

# 2. Update the image reference in api.yaml, then create the real secret
kubectl -n rag-platform create secret generic rag-api-secrets \
  --from-literal=API_KEYS="sk-team-a:acme,sk-team-b:globex" \
  --from-literal=LLM_API_KEY="sk-..."

# 3. Apply everything
kubectl apply -f deploy/kubernetes/
kubectl -n rag-platform rollout status deployment/rag-api
```

## Notes

- The config sets `INGESTION_MODE=background`; indexing runs on per-pod
  worker threads, which is appropriate at this scale. See
  `docs/architecture.md` for the single-process caveats.
- `AUTH_ENABLED=true` requires `API_KEYS` in the secret; every request must
  present `X-API-Key` (or `Authorization: Bearer`). `/health` and `/metrics`
  stay open for probes and scraping — restrict them with a NetworkPolicy if
  your cluster exposes them.
- Point a Prometheus scrape at `rag-api.rag-platform.svc.cluster.local:8000/metrics`
  via a `ServiceMonitor` or scrape config; the Grafana dashboard in
  `docker/grafana/dashboards/` works unchanged.
