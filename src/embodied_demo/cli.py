from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from embodied_demo import __version__
from embodied_demo.errors import PipelineError, SchemaValidationError
from embodied_demo.fastwam_report import generate_fastwam_report
from embodied_demo.schemas import (
    ActionChunk,
    DatasetEvidence,
    EpisodeResult,
    EvaluationManifest,
    InferenceEvidence,
    Observation,
    TrainingEvidence,
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _command_export_schema(args: argparse.Namespace) -> int:
    destination = Path(args.output_dir).expanduser().resolve()
    schemas = {
        "observation.schema.json": Observation,
        "action_chunk.schema.json": ActionChunk,
        "episode_result.schema.json": EpisodeResult,
        "evaluation_manifest.schema.json": EvaluationManifest,
        "training_evidence.schema.json": TrainingEvidence,
        "dataset_evidence.schema.json": DatasetEvidence,
        "inference_evidence.schema.json": InferenceEvidence,
    }
    for filename, model in schemas.items():
        content = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
        _write_text(destination / filename, content)
    print(f"EXPORTED {len(schemas)} schemas to {destination}")
    return 0


def _command_report_fastwam(args: argparse.Namespace) -> int:
    artifact_dir = generate_fastwam_report(
        args.run_dir,
        output_dir=args.output_dir,
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

    report_fastwam = subparsers.add_parser(
        "report-fastwam", help="normalize a FastWAM run into demo-chain evidence"
    )
    report_fastwam.add_argument("--run-dir", required=True, type=Path)
    report_fastwam.add_argument("--output-dir", type=Path)
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
