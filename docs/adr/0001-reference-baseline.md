# ADR-0001: Reference Baseline For The Demo Pipeline

Date: 2026-07-13

Status: Accepted

## Context

The project is still in the contract-first stage. We need one external project
to replicate as an engineering baseline, but we should not pull heavyweight
simulator, CUDA, model, or real-robot dependencies into the core package.

The future target environment is an NVIDIA cluster, so the baseline should also
prepare us for split-process and remote policy execution.

## Decision

Use XPolicyLab `demo_policy` and debug-mode evaluation as the primary
interface-level replication baseline.

Use RoboDojo as the future external simulation and evaluation target after the
NVIDIA/Isaac environment exists.

Use LeRobot as the future data and lightweight training format reference, not as
the current core framework.

## Consequences

The local M2/M3 implementation keeps `inproc` policy calls for deterministic
mock development, but the policy lifecycle mirrors XPolicyLab:

- `reset`
- `update_obs` / `update_obs_batch`
- `get_action` / `get_action_batch`

WebSocket transport remains a compatibility target for M5/M6, not a requirement
for the first mock demos.

RoboDojo task names such as `fold_clothes`, `organize_table`, and
`classify_objects` become external mapping targets for the two MVP tasks.

## Non-Goals

- Do not vendor XPolicyLab or RoboDojo code.
- Do not make RoboDojo, Isaac Sim, CUDA, or LeRobot dependencies of the core
  environment.
- Do not claim benchmark capability from mock results.
- Do not block M2/M3 on simulator installation.

## Follow-Up

- Add a local `PolicyAdapter` contract and tests in M2.
- Keep artifact manifests able to record `reference_baseline` and upstream pins.
- Add WebSocket loopback only after deterministic mock runner and evaluator are
  stable.
- Run RoboDojo smoke on the NVIDIA cluster once the cluster environment is known.
