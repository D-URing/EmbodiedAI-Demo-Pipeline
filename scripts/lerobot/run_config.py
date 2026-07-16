#!/usr/bin/env python3
from __future__ import annotations

# 将 experiments/lerobot/*/config.yaml 转换为底层 LeRobot shell config，并启动训练。
#
# 普通使用：
#   python experiments/lerobot/pi05_so100_8gpu_probe/run.py --dry-run
#   python experiments/lerobot/pi05_so100_8gpu_probe/run.py
#
# run.py 会调用本脚本。本脚本负责：
#   1. 读取 YAML；
#   2. 把中文友好的实验配置转换成 LEROBOT_* 环境变量；
#   3. 生成 runs/generated_configs/lerobot/.../*.sh 便于复盘；
#   4. 调用 scripts/lerobot/run_train_accelerate.sh。

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - cluster-side error path.
    raise SystemExit(
        "ERROR: PyYAML is required to read LeRobot experiment configs. "
        "Install it in the active environment with: python -m pip install PyYAML"
    ) from exc


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "scripts/lerobot/run_train_accelerate.sh").exists():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def project_path(project_root: Path, value: Any, default: str) -> str:
    raw = optional(value) or default
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def env_override(name: str, value: Any) -> str:
    override = os.environ.get(name)
    if override is not None and override != "":
        return override
    return str(value)


def export_line(name: str, value: Any) -> str:
    return f"export {name}={shlex.quote(bool_text(value))}"


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"ERROR: config root must be a mapping: {path}")
    backend = payload.get("backend")
    if backend != "lerobot":
        raise SystemExit(f"ERROR: unsupported backend={backend!r}; expected 'lerobot'")
    return payload


def build_env(config: dict[str, Any], project_root: Path, config_path: Path) -> dict[str, str]:
    experiment = config.get("experiment") or {}
    paths = config.get("paths") or {}
    dataset = config.get("dataset") or {}
    policy = config.get("policy") or {}
    training = config.get("training") or {}
    distributed = config.get("distributed") or {}
    runtime = config.get("runtime") or {}

    for section_name, section in [
        ("experiment", experiment),
        ("paths", paths),
        ("dataset", dataset),
        ("policy", policy),
        ("training", training),
        ("distributed", distributed),
        ("runtime", runtime),
    ]:
        if not isinstance(section, dict):
            raise SystemExit(f"ERROR: {section_name} section must be a mapping")

    name = experiment.get("name") or config_path.parent.name
    route = experiment.get("route") or "lerobot"
    run_name = experiment.get("run_name") or name
    run_id = experiment.get("run_id") or f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    data_root = project_path(project_root, paths.get("data_root"), "data")
    model_root = project_path(project_root, paths.get("model_root"), "models")
    run_root = project_path(project_root, paths.get("run_root"), f"runs/experiments/{route}")
    hf_home = project_path(project_root, paths.get("hf_home"), "hf_cache")
    torch_home = project_path(project_root, paths.get("torch_home"), "hf_cache/torch")

    dataset_repo_id = str(dataset.get("repo_id", "lerobot/svla_so100_pickplace"))
    dataset_root = project_path(project_root, dataset.get("root"), "data/lerobot/svla_so100_pickplace")
    policy_pretrained_path = project_path(
        project_root,
        policy.get("pretrained_path"),
        "models/lerobot/pi05/pi05_base",
    )

    env: dict[str, str] = {
        "PROJECT_ROOT": str(project_root),
        "EMBODIED_REPO_ROOT": str(project_root),
        "EMBODIED_DATA_ROOT": data_root,
        "EMBODIED_MODEL_ROOT": model_root,
        "EMBODIED_RUN_ROOT": project_path(project_root, paths.get("embodied_run_root"), "runs"),
        "EXPERIMENT_ROUTE": str(route),
        "EXPERIMENT_NAME": str(name),
        "LEROBOT_RUN_NAME": str(run_name),
        "LEROBOT_RUN_ID": str(run_id),
        "LEROBOT_RUN_ROOT": str(run_root),
        "LEROBOT_DATASET_REPO_ID": dataset_repo_id,
        "LEROBOT_DATASET_ROOT": dataset_root,
        "LEROBOT_DATASET_VIDEO_BACKEND": str(dataset.get("video_backend", "pyav")),
        "LEROBOT_DATASET_EVAL_SPLIT": str(dataset.get("eval_split", "")),
        "LEROBOT_POLICY_TYPE": str(policy.get("type", "pi05")),
        "LEROBOT_POLICY_REPO_ID": str(policy.get("repo_id", f"local/{name}")),
        "LEROBOT_POLICY_PUSH_TO_HUB": bool_text(policy.get("push_to_hub", False)),
        "LEROBOT_POLICY_DEVICE": str(policy.get("device", "cuda")),
        "LEROBOT_POLICY_PRETRAINED_PATH": policy_pretrained_path,
        "LEROBOT_POLICY_DTYPE": str(policy.get("dtype", "bfloat16")),
        "LEROBOT_POLICY_COMPILE_MODEL": bool_text(policy.get("compile_model", False)),
        "LEROBOT_POLICY_GRADIENT_CHECKPOINTING": bool_text(policy.get("gradient_checkpointing", True)),
        "LEROBOT_STEPS": str(training.get("steps", 200)),
        "LEROBOT_BATCH_SIZE": str(training.get("batch_size", 1)),
        "LEROBOT_NUM_WORKERS": str(training.get("num_workers", 4)),
        "LEROBOT_PREFETCH_FACTOR": str(training.get("prefetch_factor", 4)),
        "LEROBOT_PERSISTENT_WORKERS": bool_text(training.get("persistent_workers", True)),
        "LEROBOT_LOG_FREQ": str(training.get("log_freq", 10)),
        "LEROBOT_SAVE_CHECKPOINT": bool_text(training.get("save_checkpoint", True)),
        "LEROBOT_SAVE_FREQ": str(training.get("save_freq", 100)),
        "LEROBOT_SEED": str(training.get("seed", 1005)),
        "LEROBOT_ENV_EVAL_FREQ": str(training.get("env_eval_freq", 0)),
        "LEROBOT_EVAL_STEPS": str(training.get("eval_steps", 0)),
        "LEROBOT_MAX_EVAL_SAMPLES": str(training.get("max_eval_samples", "")),
        "LEROBOT_WANDB_ENABLE": bool_text(training.get("wandb", False)),
        "LEROBOT_NUM_PROCESSES": env_override("LEROBOT_NUM_PROCESSES", distributed.get("num_processes", 8)),
        "LEROBOT_NUM_MACHINES": env_override("LEROBOT_NUM_MACHINES", distributed.get("num_machines", 1)),
        "LEROBOT_MACHINE_RANK": env_override("LEROBOT_MACHINE_RANK", distributed.get("machine_rank", 0)),
        "LEROBOT_MAIN_PROCESS_IP": env_override(
            "LEROBOT_MAIN_PROCESS_IP",
            distributed.get("main_process_ip", "127.0.0.1"),
        ),
        "LEROBOT_MAIN_PROCESS_PORT": env_override(
            "LEROBOT_MAIN_PROCESS_PORT",
            distributed.get("main_process_port", 29505),
        ),
        "LEROBOT_ACCELERATE_MIXED_PRECISION": str(distributed.get("mixed_precision", "bf16")),
        "LEROBOT_NCCL_DEBUG": str(runtime.get("nccl_debug", "WARN")),
        "NCCL_DEBUG": str(runtime.get("nccl_debug", "WARN")),
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": project_path(project_root, paths.get("hf_hub_cache"), "hf_cache/hub"),
        "HF_DATASETS_CACHE": project_path(project_root, paths.get("hf_datasets_cache"), "hf_cache/datasets"),
        "TORCH_HOME": torch_home,
    }

    offline = runtime.get("offline")
    if offline is not None:
        offline_text = "1" if bool(offline) else "0"
        env["HF_HUB_OFFLINE"] = offline_text
        env["HF_DATASETS_OFFLINE"] = offline_text
        env["TRANSFORMERS_OFFLINE"] = offline_text
    if "hf_hub_disable_xet" in runtime:
        env["HF_HUB_DISABLE_XET"] = "1" if bool(runtime["hf_hub_disable_xet"]) else "0"
    if "allow_download" in runtime:
        env["LEROBOT_ALLOW_DOWNLOAD"] = "1" if bool(runtime["allow_download"]) else "0"

    extra_args = training.get("extra_args")
    if extra_args:
        if isinstance(extra_args, list):
            env["LEROBOT_TRAIN_EXTRA_ARGS"] = " ".join(str(item) for item in extra_args)
        elif isinstance(extra_args, str):
            env["LEROBOT_TRAIN_EXTRA_ARGS"] = extra_args
        else:
            raise SystemExit("ERROR: training.extra_args must be a string or list of strings")

    return env


def write_shell_config(output_path: Path, base_config: Path, source_yaml: Path, env: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by scripts/lerobot/run_config.py. Do not edit in place.",
        f"# Source YAML: {source_yaml}",
        "# shellcheck shell=bash",
        "",
        f"source {shlex.quote(str(base_config))}",
        "",
    ]
    for key in sorted(env):
        lines.append(export_line(key, env[key]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_path.chmod(0o755)


def print_preflight(config: dict[str, Any], env: dict[str, str]) -> None:
    expected = ((config.get("environment") or {}).get("conda_env") or "").strip()
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    if expected and active and active != expected:
        print(
            f"WARNING: active conda env is {active!r}, expected {expected!r}. "
            "If imports fail, activate the expected env first.",
            file=sys.stderr,
        )
    elif expected and not active:
        print(
            f"WARNING: expected conda env {expected!r}, but CONDA_DEFAULT_ENV is empty. "
            "If you are not using conda, make sure this Python has LeRobot dependencies.",
            file=sys.stderr,
        )

    public_env = {key: value for key, value in env.items() if "TOKEN" not in key and "PASSWORD" not in key}
    print("LEROBOT_CONFIG_RESOLVED")
    print(json.dumps(public_env, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a LeRobot experiment from YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Render config and print command without training")
    parser.add_argument(
        "--output-shell",
        default="",
        help="Optional generated shell config path. Defaults to runs/generated_configs/lerobot/<experiment>/<run_id>.sh",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    project_root = find_project_root(config_path.parent)
    config = load_config(config_path)
    env = build_env(config, project_root, config_path)

    base_config = project_root / str(config.get("base_config", "configs/lerobot/train/svla_so100_pi05_8gpu_probe.sh"))
    if not base_config.exists():
        raise SystemExit(f"ERROR: base_config not found: {base_config}")

    if args.output_shell:
        generated_config = Path(args.output_shell).resolve()
    else:
        generated_config = (
            project_root
            / "runs/generated_configs/lerobot"
            / env["EXPERIMENT_NAME"]
            / f"{env['LEROBOT_RUN_ID']}.sh"
        )
    write_shell_config(generated_config, base_config, config_path, env)

    command = ["bash", "scripts/lerobot/run_train_accelerate.sh", str(generated_config)]
    print_preflight(config, env)
    print("LEROBOT_GENERATED_CONFIG", generated_config)
    print("LEROBOT_RUN_COMMAND", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    child_env = os.environ.copy()
    child_env.update(env)
    return subprocess.call(command, cwd=project_root, env=child_env)


if __name__ == "__main__":
    raise SystemExit(main())
