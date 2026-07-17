#!/usr/bin/env python3
from __future__ import annotations

"""Launch LeRobot/FastWAM distributed experiments across SSH-reachable nodes.

This script intentionally does not replace the backend launchers.  It is a thin
orchestration layer that:

1. reads an experiment YAML and a cluster/profile YAML;
2. allocates node ranks and shared rendezvous settings;
3. starts the existing backend runner on every node via SSH;
4. stores one launcher log per rank under runs/distributed/.

Run from rank0/trainer0:

  python scripts/distributed/ssh_launch.py \
    --config experiments/lerobot/pi05_so100_8gpu_probe/config.yaml \
    --profile configs/distributed/scut_gpu11_single.yaml --dry-run
"""

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - cluster-side error path.
    raise SystemExit("ERROR: PyYAML is required. Install with: python -m pip install PyYAML") from exc


SUPPORTED_BACKENDS = {"lerobot", "fastwam"}


@dataclass(frozen=True)
class Node:
    host: str
    gpus: int
    local: bool = False
    user: str | None = None
    port: int | None = None
    workdir: str | None = None
    conda_env: str | None = None
    backend_conda_envs: dict[str, str] | None = None
    conda_init: str | None = None
    cuda_visible_devices: str | None = None

    @property
    def label(self) -> str:
        safe = self.host.replace("/", "_").replace(":", "_")
        return safe


def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "scripts").is_dir():
            return path
    raise SystemExit(f"ERROR: cannot locate project root from {start}")


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: YAML root must be a mapping: {path}")
    return data


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def quote_export(name: str, value: str | int | bool) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def get_backend(experiment: dict[str, Any]) -> str:
    backend = str(experiment.get("backend") or "").strip()
    if backend not in SUPPORTED_BACKENDS:
        raise SystemExit(f"ERROR: unsupported backend={backend!r}; expected one of {sorted(SUPPORTED_BACKENDS)}")
    return backend


def get_experiment_name(config_path: Path, experiment: dict[str, Any]) -> str:
    section = experiment.get("experiment") or {}
    if not isinstance(section, dict):
        raise SystemExit("ERROR: experiment section must be a mapping")
    return str(section.get("name") or config_path.parent.name)


def get_conda_env(experiment: dict[str, Any], profile: dict[str, Any], backend: str, node: Node) -> str:
    if node.backend_conda_envs and node.backend_conda_envs.get(backend):
        return node.backend_conda_envs[backend]
    if node.conda_env:
        return node.conda_env
    profile_env = profile.get("environment") or {}
    if not isinstance(profile_env, dict):
        raise SystemExit("ERROR: profile.environment must be a mapping")
    backend_envs = profile_env.get("backend_conda_envs") or {}
    if backend_envs:
        if not isinstance(backend_envs, dict):
            raise SystemExit("ERROR: profile.environment.backend_conda_envs must be a mapping")
        env = str(backend_envs.get(backend) or "").strip()
        if env:
            return env
    section = experiment.get("environment") or {}
    if not isinstance(section, dict):
        raise SystemExit("ERROR: environment section must be a mapping")
    env = str(section.get("conda_env") or "").strip()
    if not env:
        raise SystemExit("ERROR: environment.conda_env is required for SSH distributed launch")
    return env


def parse_nodes(profile: dict[str, Any]) -> list[Node]:
    raw_nodes = profile.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SystemExit("ERROR: profile.nodes must be a non-empty list")

    default_gpus = int((profile.get("distributed") or {}).get("gpus_per_node", 8))
    nodes: list[Node] = []
    for idx, item in enumerate(raw_nodes):
        if isinstance(item, str):
            nodes.append(Node(host=item, gpus=default_gpus))
            continue
        if not isinstance(item, dict):
            raise SystemExit(f"ERROR: profile.nodes[{idx}] must be a string or mapping")
        host = str(item.get("host") or "").strip()
        if not host:
            raise SystemExit(f"ERROR: profile.nodes[{idx}].host is required")
        node_backend_envs = item.get("backend_conda_envs") or {}
        if node_backend_envs and not isinstance(node_backend_envs, dict):
            raise SystemExit(f"ERROR: profile.nodes[{idx}].backend_conda_envs must be a mapping")
        nodes.append(
            Node(
                host=host,
                gpus=int(item.get("gpus", default_gpus)),
                local=bool(item.get("local", False)),
                user=str(item["user"]) if item.get("user") else None,
                port=int(item["port"]) if item.get("port") else None,
                workdir=str(item["workdir"]) if item.get("workdir") else None,
                conda_env=str(item["conda_env"]) if item.get("conda_env") else None,
                backend_conda_envs={str(k): str(v) for k, v in node_backend_envs.items()}
                if node_backend_envs
                else None,
                conda_init=str(item["conda_init"]) if item.get("conda_init") else None,
                cuda_visible_devices=str(item["cuda_visible_devices"])
                if item.get("cuda_visible_devices") is not None
                else None,
            )
        )
    return nodes


def ssh_target(node: Node) -> str:
    return f"{node.user}@{node.host}" if node.user else node.host


def ssh_base_command(profile: dict[str, Any], node: Node) -> list[str]:
    ssh = profile.get("ssh") or {}
    if not isinstance(ssh, dict):
        raise SystemExit("ERROR: profile.ssh must be a mapping")

    command = ["ssh"]
    for option in ssh.get("options", []) or []:
        command.extend(["-o", str(option)])
    if node.port:
        command.extend(["-p", str(node.port)])
    command.append(ssh_target(node))
    return command


def resolve_master_addr(profile: dict[str, Any], nodes: list[Node]) -> str:
    distributed = profile.get("distributed") or {}
    if not isinstance(distributed, dict):
        raise SystemExit("ERROR: profile.distributed must be a mapping")
    master_addr = str(distributed.get("master_addr") or "").strip()
    if master_addr:
        return master_addr
    # Default to the first host.  On many clusters this is exactly trainer0's
    # routable hostname.  If not, put the private IP/hostname in profile YAML.
    return nodes[0].host


def resolve_master_port(profile: dict[str, Any], backend: str) -> int:
    distributed = profile.get("distributed") or {}
    backend_ports = distributed.get("backend_master_ports") or {}
    if backend_ports:
        if not isinstance(backend_ports, dict):
            raise SystemExit("ERROR: profile.distributed.backend_master_ports must be a mapping")
        value = backend_ports.get(backend)
        if value:
            return int(value)
    default = 29505 if backend == "lerobot" else 29500
    return int(distributed.get("master_port", default))


def resolve_repo_root(profile: dict[str, Any], local_project_root: Path, node: Node) -> str:
    if node.workdir:
        return node.workdir
    paths = profile.get("paths") or {}
    if not isinstance(paths, dict):
        raise SystemExit("ERROR: profile.paths must be a mapping")
    return str(paths.get("repo_root") or local_project_root)


def resolve_conda_init(profile: dict[str, Any], node: Node | None = None) -> str:
    if node is not None and node.conda_init:
        return node.conda_init
    env = profile.get("environment") or {}
    if not isinstance(env, dict):
        raise SystemExit("ERROR: profile.environment must be a mapping")
    return str(env.get("conda_init") or "")


def backend_runner(backend: str) -> tuple[str, str]:
    if backend == "lerobot":
        return "scripts/lerobot/run_config.py", "scripts/lerobot/run_train_accelerate.sh"
    if backend == "fastwam":
        return "scripts/fastwam/run_config.py", "scripts/fastwam/run_realrobot_train_eval.sh"
    raise AssertionError(backend)


def rank_env(
    *,
    backend: str,
    run_id: str,
    rank: int,
    nodes: list[Node],
    node: Node,
    master_addr: str,
    master_port: int,
) -> dict[str, str]:
    if backend == "lerobot":
        return {
            "LEROBOT_RUN_ID": run_id,
            "LEROBOT_NUM_MACHINES": str(len(nodes)),
            "LEROBOT_MACHINE_RANK": str(rank),
            "LEROBOT_NUM_PROCESSES": str(node.gpus),
            "LEROBOT_MAIN_PROCESS_IP": master_addr,
            "LEROBOT_MAIN_PROCESS_PORT": str(master_port),
        }
    if backend == "fastwam":
        return {
            "FASTWAM_RUN_ID": run_id,
            "FASTWAM_NNODES": str(len(nodes)),
            "FASTWAM_NODE_RANK": str(rank),
            "FASTWAM_GPUS_PER_NODE": str(node.gpus),
            "FASTWAM_MASTER_ADDR": master_addr,
            "FASTWAM_MASTER_PORT": str(master_port),
        }
    raise AssertionError(backend)


def remote_script(
    *,
    backend: str,
    repo_root: str,
    conda_init: str,
    conda_env: str,
    config_rel: str,
    generated_rel: str,
    env: dict[str, str],
    node: Node,
) -> str:
    runner, launcher = backend_runner(backend)
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(repo_root)}",
    ]
    if conda_init:
        lines.append(f"source {shlex.quote(conda_init)}")
    lines.append(f"conda activate {shlex.quote(conda_env)}")
    if node.cuda_visible_devices is not None:
        lines.append(quote_export("CUDA_VISIBLE_DEVICES", node.cuda_visible_devices))
    for key in sorted(env):
        lines.append(quote_export(key, env[key]))
    lines.extend(
        [
            f"python {shlex.quote(runner)} --config {shlex.quote(config_rel)} "
            f"--output-shell {shlex.quote(generated_rel)} --dry-run",
            f"bash {shlex.quote(launcher)} {shlex.quote(generated_rel)}",
        ]
    )
    return "\n".join(lines)


def stream_process(prefix: str, process: subprocess.Popen[str], log_path: Path) -> None:
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{prefix}] {line}", end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch LeRobot/FastWAM experiment on multiple SSH nodes.")
    parser.add_argument("--config", required=True, help="Experiment config.yaml")
    parser.add_argument("--profile", required=True, help="Distributed cluster profile YAML")
    parser.add_argument("--run-id", default="", help="Shared run id. Defaults to <experiment>_<timestamp>.")
    parser.add_argument("--dry-run", action="store_true", help="Print remote commands without executing SSH.")
    parser.add_argument("--no-stream", action="store_true", help="Do not stream remote stdout to this terminal.")
    args = parser.parse_args(argv)

    project_root = find_project_root(Path.cwd())
    config_path = Path(args.config).resolve()
    profile_path = Path(args.profile).resolve()
    experiment = load_yaml(config_path)
    profile = load_yaml(profile_path)
    backend = get_backend(experiment)
    experiment_name = get_experiment_name(config_path, experiment)
    run_id = args.run_id or f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    nodes = parse_nodes(profile)
    master_addr = resolve_master_addr(profile, nodes)
    master_port = resolve_master_port(profile, backend)

    config_rel = relpath(config_path, project_root)
    launch_root = project_root / "runs" / "distributed" / backend / experiment_name / run_id
    logs_dir = launch_root / "launcher_logs"
    generated_dir = Path("runs") / "generated_configs" / backend / experiment_name / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "launcher_host": socket.gethostname(),
        "backend": backend,
        "experiment_name": experiment_name,
        "config": config_rel,
        "profile": relpath(profile_path, project_root),
        "run_id": run_id,
        "master_addr": master_addr,
        "master_port": master_port,
        "nodes": [],
    }

    processes: list[tuple[int, Node, subprocess.Popen[str], Path]] = []
    print(f"DISTRIBUTED_LAUNCH backend={backend} experiment={experiment_name} run_id={run_id}")
    print(f"DISTRIBUTED_TOPOLOGY nnodes={len(nodes)} master={master_addr}:{master_port}")

    for rank, node in enumerate(nodes):
        repo_root = resolve_repo_root(profile, project_root, node)
        conda_env = get_conda_env(experiment, profile, backend, node)
        conda_init = resolve_conda_init(profile, node)
        env = rank_env(
            backend=backend,
            run_id=run_id,
            rank=rank,
            nodes=nodes,
            node=node,
            master_addr=master_addr,
            master_port=master_port,
        )
        generated_rel = str(generated_dir / f"rank{rank:02d}_{node.label}.sh")
        script = remote_script(
            backend=backend,
            repo_root=repo_root,
            conda_init=conda_init,
            conda_env=conda_env,
            config_rel=config_rel,
            generated_rel=generated_rel,
            env=env,
            node=node,
        )
        remote = ["bash", "-lc", script]
        command = ssh_base_command(profile, node) + [shell_join(remote)]
        if node.local:
            command = remote
        log_path = logs_dir / f"rank{rank:02d}_{node.label}.log"
        manifest["nodes"].append(
            {
                "rank": rank,
                "host": node.host,
                "target": ssh_target(node),
                "local": node.local,
                "gpus": node.gpus,
                "conda_env": conda_env,
                "repo_root": repo_root,
                "cuda_visible_devices": node.cuda_visible_devices,
                "generated_config": generated_rel,
                "launcher_log": str(log_path.relative_to(project_root)),
                "env": env,
                "ssh_command": shell_join(command),
            }
        )
        launch_kind = "local" if node.local else "ssh"
        print(
            f"RANK {rank}: host={node.host} launch={launch_kind} "
            f"gpus={node.gpus} env={conda_env} log={log_path}"
        )
        if args.dry_run:
            print(f"--- rank {rank} remote script ---")
            print(script)
            print(f"--- rank {rank} launch command ---")
            print(shell_join(command))
            continue
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((rank, node, process, log_path))
        # Give rank0 a small head start to create run metadata and rendezvous.
        if rank == 0:
            time.sleep(float((profile.get("launch") or {}).get("rank0_warmup_seconds", 3)))

    manifest_path = launch_root / "launcher_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DISTRIBUTED_LAUNCH_MANIFEST {manifest_path}")

    if args.dry_run:
        return 0

    threads: list[threading.Thread] = []
    if args.no_stream:
        for rank, node, process, log_path in processes:
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"# stdout not streamed by launcher; process pid={process.pid}\n")
    else:
        for rank, node, process, log_path in processes:
            thread = threading.Thread(
                target=stream_process,
                args=(f"rank{rank}:{node.host}", process, log_path),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    exit_codes: dict[int, int] = {}
    try:
        for rank, node, process, _log_path in processes:
            exit_codes[rank] = process.wait()
    except KeyboardInterrupt:
        print("WARNING: interrupted; terminating SSH launcher children", file=sys.stderr)
        for _rank, _node, process, _log_path in processes:
            process.terminate()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=5)

    manifest["exit_codes"] = exit_codes
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failed = {rank: code for rank, code in exit_codes.items() if code != 0}
    if failed:
        print(f"DISTRIBUTED_LAUNCH_FAILED {failed}", file=sys.stderr)
        return max(failed.values())
    print("DISTRIBUTED_LAUNCH_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
