from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from embodied_demo.config import dump_yaml
from embodied_demo.errors import ConfigurationError
from embodied_demo.schemas.training import TrainingCheckpointSummary, TrainingEvidence

JsonObject = dict[str, Any]


def _read_json(path: Path) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"JSON file must contain an object: {path}")
    return payload


def _read_yaml_if_exists(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError(f"YAML file must contain a mapping: {path}")
    return payload


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _native_output_dir(run_dir: Path, backend_manifest: JsonObject) -> str | None:
    value = backend_manifest.get("fastwam_native_output_dir")
    if isinstance(value, str) and value:
        return value
    native_file = run_dir / "fastwam_native_output_dir.txt"
    if native_file.exists():
        return native_file.read_text(encoding="utf-8").strip() or None
    return None


def _checkpoint_summary(loss_summary: JsonObject) -> TrainingCheckpointSummary | None:
    latest = loss_summary.get("latest_checkpoint")
    if not isinstance(latest, dict):
        return None
    return TrainingCheckpointSummary(
        step=latest.get("step") if isinstance(latest.get("step"), int) else None,
        weights=latest.get("weights") if isinstance(latest.get("weights"), str) else None,
        state=latest.get("state") if isinstance(latest.get("state"), str) else None,
    )


def _derive_status(loss_summary: JsonObject) -> tuple[str, list[str]]:
    notes: list[str] = []
    parsed_count = int(loss_summary.get("parsed_train_count") or 0)
    loss_decreased = loss_summary.get("loss_decreased")
    training_completed = bool(loss_summary.get("training_completed"))
    latest_checkpoint = loss_summary.get("latest_checkpoint")

    if parsed_count == 0:
        return "failed", ["未解析到任何训练 loss 记录。"]
    if loss_decreased is False:
        return "failed", ["loss 首末值没有下降，需要检查训练配置、数据或学习率。"]

    if parsed_count < 2:
        notes.append("只有一个 loss 点；这通常是 smoke，只能证明链路可跑，不能证明下降趋势。")
    if loss_decreased is None:
        notes.append("loss_decreased 为 unknown；建议跑 pilot/full 获取多个日志点。")
    if not training_completed:
        notes.append("训练日志没有出现 done 记录；可能是中断 run 或仍在运行。")
    if not isinstance(latest_checkpoint, dict):
        notes.append("没有解析到 latest checkpoint；请检查 save_every 或训练日志。")

    return ("warning" if notes else "passed"), notes


def _build_training_evidence(
    run_dir: Path, backend_manifest: JsonObject, loss_summary: JsonObject
) -> TrainingEvidence:
    validation_status, notes = _derive_status(loss_summary)
    run_id = str(backend_manifest.get("run_id") or run_dir.name)
    return TrainingEvidence(
        backend=str(backend_manifest.get("backend") or "fastwam-realrobot"),
        run_id=run_id,
        source_run_dir=str(run_dir),
        native_output_dir=_native_output_dir(run_dir, backend_manifest),
        mode=backend_manifest.get("mode") if isinstance(backend_manifest.get("mode"), str) else None,
        recipe=backend_manifest.get("recipe") if isinstance(backend_manifest.get("recipe"), str) else None,
        task_name=backend_manifest.get("task_name")
        if isinstance(backend_manifest.get("task_name"), str)
        else None,
        official_ref=backend_manifest.get("official_ref")
        if isinstance(backend_manifest.get("official_ref"), str)
        else None,
        overlay_ref=backend_manifest.get("overlay_ref")
        if isinstance(backend_manifest.get("overlay_ref"), str)
        else None,
        parsed_train_count=int(loss_summary.get("parsed_train_count") or 0),
        initial_loss=loss_summary.get("initial_loss"),
        final_loss=loss_summary.get("final_loss"),
        loss_drop_ratio=loss_summary.get("loss_drop_ratio"),
        loss_decreased=loss_summary.get("loss_decreased"),
        final_step=loss_summary.get("final_step"),
        max_steps=loss_summary.get("max_steps"),
        training_completed=bool(loss_summary.get("training_completed")),
        latest_checkpoint=_checkpoint_summary(loss_summary),
        validation_status=validation_status,  # type: ignore[arg-type]
        notes=notes,
    )


def _default_output_dir(chain_id: str, evidence: TrainingEvidence) -> Path:
    return Path("runs") / "demo_chains" / chain_id / evidence.run_id


def _format_percent(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2%}"


def _render_report(
    chain_id: str,
    evidence: TrainingEvidence,
    loss_summary: JsonObject,
) -> str:
    status = evidence.validation_status.upper()
    checkpoint = evidence.latest_checkpoint
    lines = [
        f"# Demo Chain Report: {chain_id}",
        "",
        f"- status: {status}",
        f"- backend: {evidence.backend}",
        f"- mode / recipe: {evidence.mode or 'unknown'} / {evidence.recipe or 'unknown'}",
        f"- task: {evidence.task_name or 'unknown'}",
        f"- run_id: {evidence.run_id}",
        f"- loss_decreased: {str(evidence.loss_decreased).lower() if evidence.loss_decreased is not None else 'unknown'}",
        f"- initial_loss: {evidence.initial_loss}",
        f"- final_loss: {evidence.final_loss}",
        f"- loss_drop_ratio: {_format_percent(evidence.loss_drop_ratio)}",
        f"- steps: {evidence.final_step} / {evidence.max_steps}",
        f"- parsed_train_count: {evidence.parsed_train_count}",
        f"- training_completed: {str(evidence.training_completed).lower()}",
        "",
        "## Checkpoint",
        "",
        f"- weights: {checkpoint.weights if checkpoint else None}",
        f"- state: {checkpoint.state if checkpoint else None}",
        f"- native_output_dir: {evidence.native_output_dir}",
    ]

    lines.extend(["", "## Loss Metrics", ""])
    metric_summary = loss_summary.get("metric_summary")
    if isinstance(metric_summary, dict):
        for key, item in sorted(metric_summary.items()):
            if isinstance(item, dict):
                lines.append(
                    f"- {key}: initial={item.get('initial')} final={item.get('final')} delta={item.get('delta')}"
                )
    if evidence.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in evidence.notes)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "这份报告证明 demo 工程链路已经包含真实 CUDA 训练后端、loss 证据和 checkpoint 归档。",
            "它不声称已经完成 RoboDojo 仿真 benchmark 或真机闭环。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_handoff(chain_id: str, evidence: TrainingEvidence, output_dir: Path) -> str:
    return "\n".join(
        [
            f"# Handoff: {chain_id}",
            "",
            "给团队的最小复现顺序：",
            "",
            "```bash",
            "python experiments/custom/fastwam_realrobot_single8_random/run.py",
            f"embodied-demo report-fastwam --run-dir {evidence.source_run_dir} --output-dir {output_dir}",
            "```",
            "",
            "验收重点：",
            "",
            "- `training_evidence.json` 中 `validation_status` 应为 `passed` 或可解释的 `warning`。",
            "- `loss_decreased=true` 才能回答 loss 正常下降。",
            "- `latest_checkpoint.weights/state` 应指向 FastWAM 原生输出。",
            "- 真机闭环、安全 gating、RoboDojo/RoboTwin benchmark 是后续阶段。",
            "",
        ]
    )


def generate_fastwam_report(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    chain_config: str | Path = "demo_chains/fastwam_realrobot_v0.yaml",
) -> Path:
    source_run_dir = Path(run_dir).expanduser().resolve()
    backend_manifest = _read_json(source_run_dir / "backend_manifest.json")
    loss_summary = _read_json(source_run_dir / "loss_summary.json")
    chain_config_path = Path(chain_config).expanduser().resolve()
    chain_payload = _read_yaml_if_exists(chain_config_path)
    chain_id = str(chain_payload.get("chain_id") or "fastwam_realrobot_v0")

    evidence = _build_training_evidence(source_run_dir, backend_manifest, loss_summary)

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else _default_output_dir(chain_id, evidence).resolve()
    )
    generated_at = datetime.now(UTC).isoformat()
    chain_manifest = {
        "schema_version": "1.0",
        "chain_id": chain_id,
        "generated_at": generated_at,
        "chain_config": str(chain_config_path) if chain_config_path.exists() else None,
        "status": evidence.validation_status,
        "artifacts": {
            "training_evidence": "training_evidence.json",
            "checkpoint_summary": "checkpoint_summary.json",
            "report": "report.md",
            "handoff": "handoff.md",
        },
        "source": {
            "fastwam_run_dir": str(source_run_dir),
        },
    }

    checkpoint = evidence.latest_checkpoint
    checkpoint_summary = {
        "native_output_dir": evidence.native_output_dir,
        "latest_checkpoint": checkpoint.model_dump(mode="json") if checkpoint else None,
        "training_completed": evidence.training_completed,
        "final_step": evidence.final_step,
        "max_steps": evidence.max_steps,
    }

    _write_text(destination / "chain_manifest.yaml", dump_yaml(chain_manifest))
    _write_json(destination / "training_evidence.json", evidence.model_dump(mode="json"))
    _write_json(destination / "checkpoint_summary.json", checkpoint_summary)
    _write_text(destination / "report.md", _render_report(chain_id, evidence, loss_summary))
    _write_text(destination / "handoff.md", _render_handoff(chain_id, evidence, destination))
    return destination
