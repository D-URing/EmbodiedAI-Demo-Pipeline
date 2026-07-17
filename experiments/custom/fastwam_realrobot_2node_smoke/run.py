#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "scripts/distributed/ssh_launch.py").exists():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def read_launcher_python_hint(path: Path) -> str:
    """用标准库从 YAML 文本里读 launch.launcher_python。

    这里故意不依赖 PyYAML：如果用户用系统 Python 启动，而系统 Python 没有 yaml，
    wrapper 仍然能先切到实验配置指定的 conda Python，再继续解析完整 YAML。
    """
    in_launch = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("launch:"):
            in_launch = True
            continue
        if in_launch and raw_line[:1].strip():
            break
        if in_launch and line.lstrip().startswith("launcher_python:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def reexec_with_config_python(config_path: Path) -> None:
    launcher_python = read_launcher_python_hint(config_path)
    if not launcher_python:
        return
    launcher_path = Path(launcher_python)
    if not launcher_path.exists():
        return
    try:
        if launcher_path.resolve() == Path(sys.executable).resolve():
            return
    except OSError:
        if str(launcher_path) == sys.executable:
            return
    os.execv(str(launcher_path), [str(launcher_path), str(Path(__file__).resolve()), *sys.argv[1:]])


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "ERROR: PyYAML is required, and automatic env bootstrap did not find a usable launcher_python."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: config root must be a mapping: {path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run this FastWAM distributed smoke experiment.")
    parser.add_argument("--dry-run", action="store_true", help="只打印每个节点将执行的命令，不启动训练")
    parser.add_argument("--run-id", default="", help="手动指定 run_id；默认用 config 里的前缀 + 时间戳")
    parser.add_argument("--profile", default="", help="临时覆盖 launch.profile")
    parser.add_argument("--print-command", action="store_true", help="只打印底层命令，不执行")
    args, passthrough = parser.parse_known_args()

    here = Path(__file__).resolve().parent
    project_root = find_project_root(here)
    config_path = here / "config.yaml"
    reexec_with_config_python(config_path)
    config = load_config(config_path)
    launch = config.get("launch") or {}
    if not isinstance(launch, dict):
        raise SystemExit("ERROR: launch section must be a mapping")

    profile = Path(args.profile or str(launch.get("profile") or ""))
    if not profile.is_absolute():
        profile = project_root / profile
    if not profile.exists():
        raise SystemExit(f"ERROR: distributed profile not found: {profile}")

    launcher_python = str(launch.get("launcher_python") or sys.executable)
    if not Path(launcher_python).exists():
        print(f"WARNING: launcher_python not found, fallback to current Python: {launcher_python}", file=sys.stderr)
        launcher_python = sys.executable

    run_id = args.run_id or f"{launch.get('run_id_prefix') or config_path.parent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [
        launcher_python,
        "scripts/distributed/ssh_launch.py",
        "--config",
        str(config_path.relative_to(project_root)),
        "--profile",
        str(profile.relative_to(project_root)),
        "--run-id",
        run_id,
    ]
    if args.dry_run:
        command.append("--dry-run")
    command.extend(passthrough)

    print("RUN_EXPERIMENT", config_path.parent.name)
    print("RUN_CONFIG", config_path.relative_to(project_root))
    print("RUN_PROFILE", profile.relative_to(project_root))
    print("RUN_ID", run_id)
    print("RUN_COMMAND", " ".join(shlex.quote(part) for part in command))
    if args.print_command:
        return 0
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
