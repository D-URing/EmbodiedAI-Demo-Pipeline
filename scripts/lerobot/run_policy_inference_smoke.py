from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.lerobot.inspect_dataset import _describe_sample, _describe_value, _load_dataset, _set_offline_if_needed


DEFAULT_POLICY_CLASSES = {
    "act": [
        "lerobot.policies.act.modeling_act.ACTPolicy",
        "lerobot.common.policies.act.modeling_act.ACTPolicy",
    ],
    "diffusion": [
        "lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy",
        "lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy",
    ],
    "smolvla": [
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
        "lerobot.common.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    ],
    "pi0": [
        "lerobot.policies.pi0.modeling_pi0.PI0Policy",
        "lerobot.common.policies.pi0.modeling_pi0.PI0Policy",
    ],
    "pi0_fast": [
        "lerobot.policies.pi0fast.modeling_pi0fast.PI0FASTPolicy",
        "lerobot.common.policies.pi0fast.modeling_pi0fast.PI0FASTPolicy",
    ],
    "pi0fast": [
        "lerobot.policies.pi0fast.modeling_pi0fast.PI0FASTPolicy",
        "lerobot.common.policies.pi0fast.modeling_pi0fast.PI0FASTPolicy",
    ],
    "fastwam": [
        "lerobot.policies.fastwam.modeling_fastwam.FastWAMPolicy",
        "lerobot.common.policies.fastwam.modeling_fastwam.FastWAMPolicy",
    ],
}


def _import_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _candidate_policy_classes(policy_type: str, explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    return DEFAULT_POLICY_CLASSES.get(policy_type, [])


def _load_policy(policy_type: str, policy_path: Path, policy_class: str | None) -> tuple[Any, str]:
    errors: list[str] = []
    for candidate in _candidate_policy_classes(policy_type, policy_class):
        try:
            cls = _import_symbol(candidate)
            if hasattr(cls, "from_pretrained"):
                return cls.from_pretrained(str(policy_path)), candidate
            return cls(str(policy_path)), candidate
        except Exception as exc:  # pragma: no cover - external LeRobot API
            errors.append(f"{candidate}: {exc}")
    if not errors:
        errors.append("no default policy class known; set LEROBOT_POLICY_CLASS")
    raise SystemExit("ERROR: failed to load LeRobot policy. " + " | ".join(errors))


def _move_value_to_device(value: Any, device: str) -> Any:
    try:
        import torch
    except Exception:  # pragma: no cover - external env
        return value
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim > 0:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    return value


def _prepare_batch(sample: Any, device: str) -> Any:
    if isinstance(sample, dict):
        return {key: _move_value_to_device(value, device) for key, value in sample.items()}
    return _move_value_to_device(sample, device)


def _run_policy(policy: Any, batch: Any) -> Any:
    if hasattr(policy, "eval"):
        policy.eval()
    if hasattr(policy, "select_action"):
        return policy.select_action(batch)
    if hasattr(policy, "predict_action_chunk"):
        return policy.predict_action_chunk(batch)
    if callable(policy):
        return policy(batch)
    raise SystemExit("ERROR: loaded policy has no select_action, predict_action_chunk, or __call__")


def _describe_action(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return {str(key): _describe_value(value) for key, value in sorted(action.items())}
    return {"action": _describe_value(action)}


def run_inference(
    dataset_repo_id: str,
    policy_type: str,
    policy_path: Path,
    output_dir: Path,
    policy_class: str | None = None,
    dataset_root: str | None = None,
    sample_index: int = 0,
    device: str = "cuda",
    allow_download: bool = False,
) -> Path:
    _set_offline_if_needed(allow_download)
    if not policy_path.exists():
        raise SystemExit(
            f"ERROR: policy path does not exist: {policy_path}. "
            "Set LEROBOT_POLICY_PATH to a local checkpoint/pretrained directory. "
            "This smoke does not download checkpoints by default."
        )

    if device == "cuda":
        try:
            import torch
        except Exception as exc:  # pragma: no cover - external env
            raise SystemExit(f"ERROR: torch is required for cuda inference: {exc}") from exc
        if not torch.cuda.is_available():
            raise SystemExit("ERROR: cuda inference requested but torch.cuda.is_available() is false.")

    try:
        dataset = _load_dataset(dataset_repo_id, dataset_root, split=None)
        sample = dataset[sample_index]
    except Exception as exc:  # pragma: no cover - external LeRobot data/cache
        raise SystemExit(
            "ERROR: failed to load LeRobot dataset sample for inference. "
            "Downloads are disabled unless LEROBOT_ALLOW_DOWNLOAD=1. "
            f"Details: {exc}"
        ) from exc

    policy, resolved_policy_class = _load_policy(policy_type, policy_path, policy_class)
    if hasattr(policy, "to"):
        policy = policy.to(device)
    batch = _prepare_batch(sample, device)

    start = time.perf_counter()
    action = _run_policy(policy, batch)
    latency_ms = (time.perf_counter() - start) * 1000.0

    payload = {
        "schema_version": "1.0",
        "backend": "lerobot",
        "policy_type": policy_type,
        "policy_class": resolved_policy_class,
        "policy_path": str(policy_path),
        "dataset_repo_id": dataset_repo_id,
        "dataset_root": dataset_root,
        "sample_index": sample_index,
        "device": device,
        "action": _describe_action(action),
        "input_sample": _describe_sample(sample),
        "latency_ms": latency_ms,
        "validation_status": "passed",
        "notes": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "inference_evidence.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LEROBOT_INFERENCE_EVIDENCE {output_path}")
    print(f"SUMMARY policy_type={policy_type} device={device} latency_ms={latency_ms:.2f}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline LeRobot policy inference on one dataset sample.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--policy-type", required=True)
    parser.add_argument("--policy-class")
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    run_inference(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        sample_index=args.sample_index,
        policy_type=args.policy_type,
        policy_class=args.policy_class,
        policy_path=args.policy_path.expanduser().resolve(),
        device=args.device,
        output_dir=args.output_dir,
        allow_download=args.allow_download,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
