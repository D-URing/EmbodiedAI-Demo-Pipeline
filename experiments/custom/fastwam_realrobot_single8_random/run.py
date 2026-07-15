#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "scripts/fastwam/run_config.py").exists():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = find_project_root(here)
    config = here / "config.yaml"
    runner = project_root / "scripts/fastwam/run_config.py"
    command = [sys.executable, str(runner), "--config", str(config), *sys.argv[1:]]
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
