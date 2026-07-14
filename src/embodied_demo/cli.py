from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from embodied_demo import __version__
from embodied_demo.config import dump_yaml, load_registry, load_resolved_run, load_task
from embodied_demo.demo_runner import run_mock_demo
from embodied_demo.errors import PipelineError, SchemaValidationError
from embodied_demo.fastwam_report import generate_fastwam_report
from embodied_demo.registry import iter_registered_tasks
from embodied_demo.schemas import (
    ActionChunk,
    DatasetEvidence,
    EpisodeResult,
    EvaluationManifest,
    InferenceEvidence,
    Observation,
    RunSpec,
    TaskRegistry,
    TaskSpec,
    TrainingEvidence,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _command_validate(args: argparse.Namespace) -> int:
    if args.kind == "run":
        resolved = load_resolved_run(args.config)
        print(
            "VALID "
            f"run={resolved.run.name} task={resolved.task_spec.id}@{resolved.task_spec.version} "
            f"mode={resolved.run.runtime.mode.value} "
            f"backend={resolved.run.environment.backend.value}"
        )
    elif args.kind == "task":
        task = load_task(args.config)
        print(f"VALID task={task.id}@{task.version}")
    else:
        registry = load_registry(args.config)
        print(f"VALID registry tasks={len(registry.tasks)}")
    return 0


def _command_list_tasks(args: argparse.Namespace) -> int:
    tasks = iter_registered_tasks(args.registry)
    headings = ("ID", "VERSION", "STATUS", "DIFFICULTY", "BACKENDS")
    rows = [
        (
            entry.task.id,
            entry.task.version,
            entry.status.value,
            entry.task.difficulty.value,
            ",".join(item.value for item in entry.task.backends.supported),
        )
        for entry in tasks
    ]
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rows))
        for index in range(len(headings))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headings)))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return 0


def _command_dry_run(args: argparse.Namespace) -> int:
    resolved = load_resolved_run(args.config)
    content = dump_yaml(resolved)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        _write_text(output, content)
        print(f"RESOLVED {output}")
    else:
        print(content, end="")
    return 0


def _command_export_schema(args: argparse.Namespace) -> int:
    destination = Path(args.output_dir).expanduser().resolve()
    schemas = {
        "task.schema.json": TaskSpec,
        "run.schema.json": RunSpec,
        "observation.schema.json": Observation,
        "action_chunk.schema.json": ActionChunk,
        "episode_result.schema.json": EpisodeResult,
        "evaluation_manifest.schema.json": EvaluationManifest,
        "task_registry.schema.json": TaskRegistry,
        "training_evidence.schema.json": TrainingEvidence,
        "dataset_evidence.schema.json": DatasetEvidence,
        "inference_evidence.schema.json": InferenceEvidence,
    }
    for filename, model in schemas.items():
        content = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
        _write_text(destination / filename, content)
    print(f"EXPORTED {len(schemas)} schemas to {destination}")
    return 0


def _command_run(args: argparse.Namespace) -> int:
    artifact_dir = run_mock_demo(args.config, args.output_dir)
    result = json.loads((artifact_dir / "result.json").read_text(encoding="utf-8"))
    print(f"RUN_COMPLETE {artifact_dir}")
    print(
        "SUMMARY "
        f"success={str(result['episode_success']).lower()} "
        f"progress={result['progress_score']:.1f} "
        f"steps={result['episode_steps']}"
    )
    print(f"REPORT {artifact_dir / 'report.md'}")
    print(f"RESULT {artifact_dir / 'result.json'}")
    return 0


def _command_report_fastwam(args: argparse.Namespace) -> int:
    artifact_dir = generate_fastwam_report(
        args.run_dir,
        output_dir=args.output_dir,
        mock_run_dirs=args.mock_run_dir,
        chain_config=args.chain_config,
    )
    evidence = json.loads((artifact_dir / "training_evidence.json").read_text(encoding="utf-8"))
    loss_decreased = evidence["loss_decreased"]
    loss_decreased_text = "unknown" if loss_decreased is None else str(loss_decreased).lower()
    print(f"REPORT_FASTWAM_COMPLETE {artifact_dir}")
    print(
        "SUMMARY "
        f"status={evidence['validation_status']} "
        f"loss_decreased={loss_decreased_text} "
        f"initial_loss={evidence['initial_loss']} "
        f"final_loss={evidence['final_loss']} "
        f"steps={evidence['final_step']}/{evidence['max_steps']}"
    )
    print(f"REPORT {artifact_dir / 'report.md'}")
    print(f"EVIDENCE {artifact_dir / 'training_evidence.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodied-demo",
        description="Contract and configuration tools for the EmbodiedAI demo pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a run, task, or registry")
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--kind", choices=("run", "task", "registry"), default="run")
    validate.set_defaults(handler=_command_validate)

    list_tasks = subparsers.add_parser("list-tasks", help="list and validate registered tasks")
    list_tasks.add_argument("--registry", type=Path, default=Path("tasks/registry.yaml"))
    list_tasks.set_defaults(handler=_command_list_tasks)

    dry_run = subparsers.add_parser("dry-run", help="resolve config without executing a rollout")
    dry_run.add_argument("--config", required=True, type=Path)
    dry_run.add_argument("--output", type=Path)
    dry_run.set_defaults(handler=_command_dry_run)

    run = subparsers.add_parser("run", help="execute a deterministic mock demo rollout")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output-dir", type=Path)
    run.set_defaults(handler=_command_run)

    report_fastwam = subparsers.add_parser(
        "report-fastwam", help="normalize a FastWAM run into demo-chain evidence"
    )
    report_fastwam.add_argument("--run-dir", required=True, type=Path)
    report_fastwam.add_argument("--output-dir", type=Path)
    report_fastwam.add_argument(
        "--mock-run-dir",
        action="append",
        type=Path,
        default=[],
        help="optional mock demo artifact directory containing result.json",
    )
    report_fastwam.add_argument(
        "--chain-config",
        type=Path,
        default=Path("demo_chains/fastwam_realrobot_v0.yaml"),
    )
    report_fastwam.set_defaults(handler=_command_report_fastwam)

    export_schema = subparsers.add_parser(
        "export-schema", help="export public contracts as JSON Schema"
    )
    export_schema.add_argument("--output-dir", type=Path, default=Path("schemas"))
    export_schema.set_defaults(handler=_command_export_schema)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (PipelineError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            exc = SchemaValidationError(str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
