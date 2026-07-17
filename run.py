#!/usr/bin/env python3
from __future__ import annotations

"""Small project-level launcher.

This is the human-facing entrypoint.  It hides the long distributed launch
command behind stable aliases configured in configs/launch/aliases.yaml.
"""

import argparse
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
    raise SystemExit("ERROR: PyYAML is required for run.py. Activate the project env first.") from exc


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "configs").is_dir() and (path / "scripts").is_dir():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def load_aliases(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "configs/launch/aliases.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases = data.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise SystemExit(f"ERROR: aliases must be a mapping in {path}")
    return aliases


def resolve_path(project_root: Path, value: str, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise SystemExit(f"ERROR: {field} not found: {path}")
    return path


def choose_python(alias: dict[str, Any]) -> str:
    configured = str(alias.get("launcher_python") or "").strip()
    if configured and Path(configured).exists():
        return configured
    if configured:
        print(
            f"WARNING: configured launcher_python does not exist, fallback to current Python: {configured}",
            file=sys.stderr,
        )
    return sys.executable


def print_aliases(aliases: dict[str, dict[str, Any]]) -> None:
    print("可用启动别名：")
    width = max([len(name) for name in aliases] or [1])
    for name in sorted(aliases):
        desc = str(aliases[name].get("description") or "")
        print(f"  {name:<{width}}  {desc}")
    print()
    print("示例：")
    print("  ./run.py pi05-2node --dry-run")
    print("  ./run.py fastwam-2node-smoke")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def main(argv: list[str] | None = None) -> int:
    project_root = find_project_root(Path(__file__).resolve())
    aliases = load_aliases(project_root)

    parser = argparse.ArgumentParser(
        description="EmbodiedAI-Demo-Pipeline 统一启动入口；复杂命令收敛到 configs/launch/aliases.yaml。"
    )
    parser.add_argument("alias", nargs="?", help="启动别名；用 `list` 查看全部")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不启动训练")
    parser.add_argument("--run-id", default="", help="手动指定 run_id；默认用 alias 前缀 + 时间戳")
    parser.add_argument("--profile", default="", help="临时覆盖 distributed profile 路径")
    parser.add_argument("--config", default="", help="临时覆盖 experiment config 路径")
    parser.add_argument("--print-command", action="store_true", help="打印底层命令，方便复制排查")
    args, passthrough = parser.parse_known_args(argv)

    if not args.alias or args.alias in {"list", "ls"}:
        print_aliases(aliases)
        return 0

    if args.alias not in aliases:
        print(f"ERROR: unknown launch alias: {args.alias}", file=sys.stderr)
        print_aliases(aliases)
        return 2

    alias = aliases[args.alias]
    backend = str(alias.get("backend") or "").strip()
    if backend not in {"lerobot", "fastwam"}:
        raise SystemExit(f"ERROR: alias {args.alias} has invalid backend={backend!r}")

    config = resolve_path(project_root, args.config or str(alias.get("config") or ""), "config")
    profile_value = args.profile or str(alias.get("profile") or "")
    profile = resolve_path(project_root, profile_value, "profile") if profile_value else None
    run_id = args.run_id or f"{alias.get('run_id_prefix') or args.alias}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    launcher_python = choose_python(alias)
    if profile is not None:
        command = [
            launcher_python,
            "scripts/distributed/ssh_launch.py",
            "--config",
            str(config.relative_to(project_root)),
            "--profile",
            str(profile.relative_to(project_root)),
            "--run-id",
            run_id,
        ]
        if args.dry_run:
            command.append("--dry-run")
    else:
        runner = "scripts/lerobot/run_config.py" if backend == "lerobot" else "scripts/fastwam/run_config.py"
        command = [
            launcher_python,
            runner,
            "--config",
            str(config.relative_to(project_root)),
        ]
        if args.dry_run:
            command.append("--dry-run")

    command.extend(passthrough)

    print(f"RUN_ALIAS {args.alias}")
    print(f"RUN_BACKEND {backend}")
    print(f"RUN_CONFIG {config.relative_to(project_root)}")
    if profile is not None:
        print(f"RUN_PROFILE {profile.relative_to(project_root)}")
    print(f"RUN_ID {run_id}")
    print(f"RUN_COMMAND {shell_join(command)}")

    if args.print_command:
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.call(command, cwd=project_root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
