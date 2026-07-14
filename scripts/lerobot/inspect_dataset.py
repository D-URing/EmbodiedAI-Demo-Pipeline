from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


def _set_offline_if_needed(allow_download: bool) -> None:
    if allow_download:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _import_lerobot_dataset() -> type:
    candidates = [
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ]
    errors: list[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "LeRobotDataset")
        except Exception as exc:  # pragma: no cover - depends on external LeRobot install
            errors.append(f"{module_name}: {exc}")
    raise SystemExit(
        "ERROR: cannot import LeRobotDataset. Install LeRobot in the cluster env first. "
        + " | ".join(errors)
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def _describe_value(value: Any) -> dict[str, Any]:
    description: dict[str, Any] = {"python_type": type(value).__name__}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        description["shape"] = [int(item) for item in tuple(shape)]
    if dtype is not None:
        description["dtype"] = str(dtype)
    if isinstance(value, str | int | float | bool):
        description["value"] = value
    elif isinstance(value, dict):
        description["keys"] = sorted(str(key) for key in value.keys())
    elif isinstance(value, (list, tuple)):
        description["length"] = len(value)
    return description


def _describe_sample(sample: Any) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return {"sample": _describe_value(sample)}
    return {str(key): _describe_value(value) for key, value in sorted(sample.items())}


def _load_dataset(repo_id: str, root: str | None, split: str | None) -> Any:
    dataset_cls = _import_lerobot_dataset()
    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if root:
        kwargs["root"] = root
    if split:
        kwargs["split"] = split
    try:
        return dataset_cls(**kwargs)
    except TypeError:
        kwargs.pop("split", None)
        return dataset_cls(**kwargs)


def _metadata(dataset: Any) -> dict[str, Any]:
    meta = getattr(dataset, "meta", None)
    payload: dict[str, Any] = {}
    for name in ("repo_id", "root", "fps", "features", "tasks", "episodes", "stats"):
        if hasattr(dataset, name):
            payload[name] = _jsonable(getattr(dataset, name))
        elif meta is not None and hasattr(meta, name):
            payload[name] = _jsonable(getattr(meta, name))
    return payload


def inspect_dataset(
    repo_id: str,
    output_dir: Path,
    root: str | None = None,
    split: str | None = None,
    sample_index: int = 0,
    allow_download: bool = False,
) -> Path:
    _set_offline_if_needed(allow_download)
    try:
        dataset = _load_dataset(repo_id, root, split)
        length = len(dataset)
        sample = dataset[sample_index] if length else {}
    except Exception as exc:  # pragma: no cover - depends on external LeRobot data/cache
        raise SystemExit(
            "ERROR: failed to load LeRobot dataset. By default downloads are disabled. "
            "Set LEROBOT_ALLOW_DOWNLOAD=1 only on a machine where downloads are intended, "
            "or set LEROBOT_DATASET_ROOT to an existing local dataset/cache. "
            f"Details: {exc}"
        ) from exc

    meta = _metadata(dataset)
    features = meta.get("features")
    if not isinstance(features, dict):
        features = _jsonable(getattr(dataset, "features", {}))

    payload = {
        "schema_version": "1.0",
        "backend": "lerobot",
        "repo_id": repo_id,
        "root": root,
        "split": split,
        "sample_index": sample_index,
        "length": length,
        "fps": meta.get("fps"),
        "features": features if isinstance(features, dict) else {"raw": features},
        "sample": _describe_sample(sample),
        "metadata": meta,
        "allow_download": allow_download,
        "validation_status": "passed",
        "notes": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_profile.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LEROBOT_DATASET_PROFILE {output_path}")
    print(f"SUMMARY repo_id={repo_id} length={length} sample_index={sample_index}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a LeRobot dataset without training.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--split")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    inspect_dataset(
        repo_id=args.repo_id,
        root=args.root,
        split=args.split,
        sample_index=args.sample_index,
        output_dir=args.output_dir,
        allow_download=args.allow_download,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
