# ADR-0002: FastWAM Evidence Chain And Demo Coverage Roadmap

Date: 2026-07-14

Status: Accepted

Superseded-in-part-by: [ADR-0003](0003-lerobot-first-fastwam-pipeline.md)

Note: This ADR remains valid for the custom FastWAM overlay evidence path. ADR-0003
reframes the first demo pipeline as LeRobot-first, with FastWAM preferred through
the LeRobot-native policy path when possible.

## Context

The project now has two different kinds of runnable evidence:

- deterministic household mock demos, which validate task contracts, rollout,
  logging, evaluation, and reports;
- a FastWAM real-robot training backend, which validates that a real CUDA
  training pipeline can produce loss logs, checkpoints, and handoff artifacts.

These are both valuable, but they answer different questions. The team needs a
stable planning frame so future demos can expand without conflating mock success,
training loss, simulation success, and real-robot success.

## Decision

Introduce a three-layer evidence model:

1. task and engineering-chain evidence;
2. real training evidence;
3. closed-loop capability evidence.

Use the FastWAM realrobot overlay as the first real training evidence backend
for the project. Keep it as an external backend with pinned upstream references,
scripts, and artifact importers; do not vendor FastWAM code, weights, datasets,
or run outputs into this repository.

Track demo expansion with an explicit readiness ladder:

- R0 Task Spec
- R1 Mock Rollout
- R2 Training Evidence
- R3 Offline Action / Replay
- R4 Simulation
- R5 Real Shadow
- R6 Real Closed-loop

Use [`DEMO_COVERAGE_ROADMAP.md`](../DEMO_COVERAGE_ROADMAP.md) as the stable
coverage planning document. The master plan should link to that document rather
than becoming the only place where every future demo task is enumerated.

## Consequences

- The project can answer “does loss go down?” through FastWAM/LeRobot training
  evidence without claiming household task success.
- The first-stage demo scope expands beyond two tasks, but only a small number
  of additional tasks should enter R0/R1 at a time.
- Viewer remains lower priority than task specs, runner contracts, evidence
  artifacts, and evaluator correctness.
- NVIDIA cluster work is prepared through scripts, Slurm templates, and
  backend-specific runbooks, not by adding CUDA/Isaac dependencies to the core
  Python environment.
- Every report must display backend/readiness level clearly.

## Non-Goals

- Do not claim FastWAM training evidence is equivalent to kitchen, folding, or
  household service success.
- Do not create CPU toy fallbacks for training demos.
- Do not make FastWAM, LeRobot, RoboDojo, RoboCasa, RoboTwin, Isaac, or real
  robot SDKs dependencies of the core package.
- Do not prioritize a complex web viewer before artifacts and reports are
  stable.

## Follow-Up

- Kitchen counter sorting and drawer pick/place have entered R1 mock rollout.
  Add one additional cleaning or laundry task next.
- Factor common mock primitives for object-in-region, category routing,
  articulated state, and stage predicates.
- Keep `embodied-demo run --config configs/runs/tabletop_sorting_mock.yaml` as the first combined demo-pack entry and add a
  unified summary report after the next R1 task lands.
- Run FastWAM `pilot` on the NVIDIA cluster to generate real loss descent
  evidence for the current PR handoff.
