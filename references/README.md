# Reference Baselines

This directory stores small, reviewable metadata for external projects that shape
the demo pipeline. It must not vendor third-party code, model checkpoints,
datasets, simulator assets, or generated benchmark results.

- `upstreams.yaml` pins the external repositories and documentation used for
  current engineering decisions.
- `xpolicylab_baseline.yaml` defines the first replication target at the
  interface level.

Large upstream checkouts should live in the repo-local ignored directory
`upstreams/` when the whole project is placed on shared storage.
