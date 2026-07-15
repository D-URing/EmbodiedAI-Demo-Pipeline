# ADR-0003: LeRobot-First Pipeline With FastWAM As First-Class Policy Path

Date: 2026-07-14

Status: Accepted

## Context

The project goal has been clarified: the first demo pipeline should use LeRobot
as the primary reference, and should run end-to-end from data loading to policy
inference. FastWAM is not merely a separate external backend anymore: current
LeRobot documentation exposes FastWAM as a policy path through
`policy.type=fastwam`, and the policy can be loaded through the LeRobot policy
API for action prediction.

At the same time, the project already has an internal
`D-URing/fastwam-realrobot-pipeline` overlay. That work should not be discarded,
because future custom models, private real-robot datasets, cluster recipes, and
non-upstream model variants will still need a self-owned path.

## Decision

Use **LeRobot-first** as the main demo pipeline:

```text
LeRobot dataset
  -> LeRobot policy config
  -> LeRobot training or checkpoint loading
  -> LeRobot policy inference
  -> normalized evidence/report
```

Treat FastWAM as two related but distinct integration modes:

1. **LeRobot-native FastWAM path**: preferred path for the official demo chain,
   using `policy.type=fastwam` and LeRobot policy APIs.
2. **Custom FastWAM overlay path**: retained for internal real-robot data,
   cluster recipes, private extensions, and future self-built models.

Household mock demos remain useful, but they move to an application/evaluation
layer. They should not be the primary proof that the LeRobot pipeline is
working.

## Consequences

- The first demo-chain target becomes data-read -> train/load -> inference ->
  report, rather than adding more household mock tasks.
- Existing LeRobot train smoke remains valid and should be extended with dataset
  inspection and offline inference smoke.
- Existing FastWAM overlay integration remains valid, but should be described as
  an internal/custom extension path rather than the main official path.
- Future custom models should plug into the same evidence contract, either
  through LeRobot-compatible policies or through a custom backend adapter.
- The core Python environment still stays lightweight; LeRobot/FastWAM CUDA
  dependencies remain in cluster-specific environments.

## Non-Goals

- Do not vendor LeRobot, FastWAM, checkpoints, datasets, or private overlay code.
- Do not claim real-robot or simulator success from offline inference alone.
- Do not delete the internal FastWAM overlay path.
- Do not force every future model to be upstreamed into LeRobot before it can be
  evaluated.

## Follow-Up

- Add `docs/LEROBOT_FIRST_PIPELINE.md` as the new implementation target.
- Add a LeRobot-first demo-chain spec for dataset inspection, training/loading,
  inference, and reporting.
- Implement LeRobot dataset inspection and inference experiment before adding more
  household mock tasks.
- Reframe FastWAM docs around two paths: LeRobot-native and custom overlay.
