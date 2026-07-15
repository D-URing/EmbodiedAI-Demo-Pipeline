from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

JsonObject = dict[str, Any]


def _read_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: JSON file must contain an object: {path}")
    return payload


def _copy_json(source: Path, destination: Path) -> JsonObject:
    payload = _read_json(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return payload


def _maybe_copy_json(source: Path | None, destination: Path) -> JsonObject | None:
    if source is None:
        return None
    return _copy_json(source, destination)


def _render_report(
    chain_id: str,
    dataset: JsonObject,
    inference: JsonObject,
    training: JsonObject | None,
) -> str:
    lines = [
        f"# Demo Chain Report: {chain_id}",
        "",
        "- status: PLANNED/SMOKE",
        "- backend: lerobot",
        f"- dataset: {dataset.get('repo_id')}",
        f"- dataset_length: {dataset.get('length')}",
        f"- sample_index: {dataset.get('sample_index')}",
        f"- policy_type: {inference.get('policy_type')}",
        f"- policy_class: {inference.get('policy_class')}",
        f"- policy_path: {inference.get('policy_path')}",
        f"- device: {inference.get('device')}",
        f"- latency_ms: {inference.get('latency_ms')}",
        "",
        "## Action Output",
        "",
    ]
    action = inference.get("action")
    if isinstance(action, dict):
        for key, value in sorted(action.items()):
            lines.append(f"- {key}: `{json.dumps(value, ensure_ascii=False)}`")
    else:
        lines.append("- action: unknown")

    lines.extend(["", "## Training / Checkpoint", ""])
    if training:
        lines.extend(
            [
                f"- initial_loss: {training.get('initial_loss')}",
                f"- final_loss: {training.get('final_loss')}",
                f"- loss_decreased: {training.get('loss_decreased')}",
                f"- loss_drop_ratio: {training.get('loss_drop_ratio')}",
            ]
        )
    else:
        lines.append("- not attached; this report used an existing local policy path or checkpoint.")

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report proves a LeRobot data-to-inference smoke path: dataset read, policy load,",
            "one offline action prediction, and normalized evidence artifacts. It does not claim",
            "simulator benchmark completion or real-robot closed-loop success.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_handoff(output_dir: Path) -> str:
    return "\n".join(
        [
            "# Handoff: LeRobot Data-to-Inference",
            "",
            "Minimal sequence on a prepared LeRobot environment:",
            "",
            "```bash",
            "make lerobot-data-smoke",
            "LEROBOT_POLICY_PATH=/path/to/local/checkpoint bash experiments/lerobot/<matching_inference_experiment>/launch.sh",
            "LEROBOT_DATASET_PROFILE=/path/to/dataset_profile.json \\",
            "LEROBOT_INFERENCE_EVIDENCE=/path/to/inference_evidence.json \\",
            "python scripts/lerobot/generate_data_to_inference_report.py \\",
            "  --dataset-profile \"$LEROBOT_DATASET_PROFILE\" \\",
            "  --inference-evidence \"$LEROBOT_INFERENCE_EVIDENCE\" \\",
            "  --output-dir build/lerobot-chain-report",
            "```",
            "",
            f"Artifacts are in `{output_dir}`.",
            "",
        ]
    )


def generate_report(
    dataset_profile: Path,
    inference_evidence: Path,
    output_dir: Path,
    training_summary: Path | None = None,
    chain_config: Path = Path("demo_chains/lerobot_fastwam_data_to_inference_v0.yaml"),
) -> Path:
    dataset = _copy_json(dataset_profile, output_dir / "dataset_profile.json")
    inference = _copy_json(inference_evidence, output_dir / "inference_evidence.json")
    training = _maybe_copy_json(training_summary, output_dir / "training_summary.json")

    chain_id = "lerobot_fastwam_data_to_inference_v0"
    if chain_config.exists():
        payload = yaml.safe_load(chain_config.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict) and isinstance(payload.get("chain_id"), str):
            chain_id = payload["chain_id"]

    manifest = {
        "schema_version": "1.0",
        "chain_id": chain_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "chain_config": str(chain_config),
        "artifacts": {
            "dataset_profile": "dataset_profile.json",
            "inference_evidence": "inference_evidence.json",
            "training_summary": "training_summary.json" if training else None,
            "report": "report.md",
            "handoff": "handoff.md",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chain_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_render_report(chain_id, dataset, inference, training), encoding="utf-8")
    (output_dir / "handoff.md").write_text(_render_handoff(output_dir), encoding="utf-8")
    print(f"LEROBOT_CHAIN_REPORT {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a LeRobot data-to-inference evidence report.")
    parser.add_argument("--dataset-profile", required=True, type=Path)
    parser.add_argument("--inference-evidence", required=True, type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chain-config", type=Path, default=Path("demo_chains/lerobot_fastwam_data_to_inference_v0.yaml"))
    args = parser.parse_args()
    generate_report(
        dataset_profile=args.dataset_profile,
        inference_evidence=args.inference_evidence,
        training_summary=args.training_summary,
        output_dir=args.output_dir,
        chain_config=args.chain_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
