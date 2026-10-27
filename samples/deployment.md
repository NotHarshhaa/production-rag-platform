# CI/CD and Deployment

## Deployment pipeline

The deployment pipeline validates every change before it reaches production.

Changes enter through a pull request, where the CI pipeline runs unit tests,
integration tests, and lint checks. Only branches that pass all checks can be
merged into the main branch.

After a merge, the build stage compiles the application and produces a Docker
image. The image is tagged with the commit hash so every deployment is
traceable to an exact revision.

The deploy stage promotes the image to a staging environment first. Staging
runs the same containers as production, including the database and the
vector store, so configuration problems surface before release.

Production deployments use a blue-green strategy. The new version receives a
small percentage of traffic while health checks and error rates are watched.
If the error rate rises above the threshold, traffic is shifted back to the
previous version automatically.

Rollback is a matter of re-pointing the load balancer at the previous image
tag, which keeps recovery time short.

## Release checklist

Before a release, the team confirms that the evaluation suite passes, the
Grafana dashboards show healthy trends, and the changelog is up to date.
