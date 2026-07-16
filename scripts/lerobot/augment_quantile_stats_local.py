#!/usr/bin/env python3
"""Safely augment a local LeRobot dataset with quantile statistics.

LeRobot pi05 uses quantile normalization and therefore requires q01/q99
statistics in ``meta/stats.json``. Some public datasets were published before
these quantile fields became mandatory. The upstream helper can compute them,
but it also pushes the modified dataset back to the Hugging Face Hub.

This project helper deliberately writes only to the local dataset directory.
By default it computes quantiles only for the features that pi05 actually
normalizes with ``QUANTILES`` (``action`` and ``observation.state``). This
avoids decoding all videos just to compute visual quantiles, because pi05 maps
``VISUAL`` features to ``IDENTITY`` normalization.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from lerobot.datasets import DEFAULT_QUANTILES, LeRobotDataset, aggregate_stats, get_feature_stats, write_stats
from lerobot.utils.utils import init_logging


def quantile_keys() -> list[str]:
    return [f"q{int(q * 100):02d}" for q in DEFAULT_QUANTILES]


def has_quantile_stats(stats: dict[str, dict[str, Any]] | None, feature_keys: list[str]) -> bool:
    """Return true when every selected feature has the expected quantiles."""

    if not stats:
        return False

    expected = set(quantile_keys())
    selected_features = [stats.get(key) for key in feature_keys]
    if not selected_features or any(not isinstance(feature_stats, dict) for feature_stats in selected_features):
        return False

    return all(expected.issubset(feature_stats.keys()) for feature_stats in selected_features)


def parse_feature_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("--features must contain at least one feature key")
    return keys


def parquet_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")
    return files


def compute_selected_quantiles_from_parquet(root: Path, feature_keys: list[str]) -> dict[str, dict[str, list[float]]]:
    """Compute q01/q10/q50/q90/q99 for selected non-video parquet columns."""

    chunks: dict[str, list[np.ndarray]] = {key: [] for key in feature_keys}
    for path in parquet_files(root):
        table = pq.read_table(path, columns=feature_keys)
        for key in feature_keys:
            values = table[key].to_pylist()
            chunks[key].append(np.asarray(values, dtype=np.float64))

    quantiles: dict[str, dict[str, list[float]]] = {}
    for key, arrays in chunks.items():
        data = np.concatenate(arrays, axis=0)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        feature_quantiles: dict[str, list[float]] = {}
        for quantile in DEFAULT_QUANTILES:
            q_key = f"q{int(quantile * 100):02d}"
            q_value = np.quantile(data, quantile, axis=0)
            feature_quantiles[q_key] = q_value.tolist()
        quantiles[key] = feature_quantiles

    return quantiles


def load_stats(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot stats file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_stats_json(path: Path, stats: dict[str, dict[str, Any]]) -> None:
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_single_episode(dataset: LeRobotDataset, episode_idx: int) -> dict[str, dict[str, Any]]:
    logging.info("Computing stats for episode %s", episode_idx)

    start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    collected_data: dict[str, list[torch.Tensor]] = {}
    for idx in range(start_idx, end_idx):
        item = dataset[idx]
        for key, value in item.items():
            if key not in dataset.features:
                continue
            if dataset.features[key]["dtype"] == "string":
                continue
            if not isinstance(value, torch.Tensor):
                continue
            collected_data.setdefault(key, []).append(value)

    ep_stats: dict[str, dict[str, Any]] = {}
    for key, data_list in collected_data.items():
        data = torch.stack(data_list).cpu().numpy()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        feature_stats = get_feature_stats(
            data,
            axis=axes_to_reduce,
            keepdims=keepdims,
            quantile_list=DEFAULT_QUANTILES,
        )

        if dataset.features[key]["dtype"] in ["image", "video"]:
            feature_stats = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in feature_stats.items()}

        ep_stats[key] = feature_stats

    return ep_stats


def compute_quantile_stats_for_dataset(dataset: LeRobotDataset) -> dict[str, dict[str, Any]]:
    logging.info("Computing quantile statistics for %s episodes", dataset.num_episodes)

    episode_stats_list: list[dict[str, dict[str, Any]]] = []
    has_videos = len(dataset.meta.video_keys) > 0

    if has_videos:
        logging.info("Dataset contains video keys; processing episodes sequentially for decoder safety")
        for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
            episode_stats_list.append(process_single_episode(dataset, episode_idx))
    else:
        logging.info("Dataset has no video keys; processing episodes in parallel")
        max_workers = min(dataset.num_episodes, 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(process_single_episode, dataset, episode_idx): episode_idx
                for episode_idx in range(dataset.num_episodes)
            }
            episode_results: dict[int, dict[str, dict[str, Any]]] = {}
            with tqdm(total=dataset.num_episodes, desc="Processing episodes") as pbar:
                for future in concurrent.futures.as_completed(future_to_episode):
                    episode_idx = future_to_episode[future]
                    episode_results[episode_idx] = future.result()
                    pbar.update(1)

        episode_stats_list.extend(episode_results[idx] for idx in sorted(episode_results))

    if not episode_stats_list:
        raise ValueError("No episode data found for computing statistics")

    logging.info("Aggregating statistics from %s episodes", len(episode_stats_list))
    return aggregate_stats(episode_stats_list)


def write_manifest(
    output_path: Path,
    *,
    repo_id: str,
    root: Path,
    stats_path: Path,
    skipped: bool,
    feature_keys: list[str],
    mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "backend": "lerobot",
        "operation": "augment_quantile_stats_local",
        "repo_id": repo_id,
        "dataset_root": str(root),
        "stats_path": str(stats_path),
        "skipped": skipped,
        "mode": mode,
        "features": feature_keys,
        "quantile_keys": quantile_keys(),
        "notes": [
            "This operation writes only to the local LeRobot dataset directory.",
            "It intentionally does not push modified metadata to the Hugging Face Hub.",
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment a local LeRobot dataset with q01/q99 stats")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. lerobot/svla_so100_pickplace")
    parser.add_argument("--root", required=True, help="Local LeRobot dataset directory")
    parser.add_argument(
        "--features",
        default="action,observation.state",
        help=(
            "Comma-separated non-video features to augment. Default matches pi05 QUANTILES features: "
            "action,observation.state"
        ),
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Use the slow upstream-style path and compute quantiles for all dataset features, including videos",
    )
    parser.add_argument(
        "--video-backend",
        default=os.environ.get("LEROBOT_DATASET_VIDEO_BACKEND", "pyav"),
        help="LeRobot video backend used while reading local videos; default: env LEROBOT_DATASET_VIDEO_BACKEND or pyav",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if quantile stats already exist")
    parser.add_argument(
        "--manifest",
        default="runs/artifact_manifests/lerobot_quantile_stats_manifest.json",
        help="Where to write a small local evidence manifest",
    )
    args = parser.parse_args()

    init_logging()
    root = Path(args.root).expanduser().resolve()
    feature_keys = parse_feature_keys(args.features)
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path

    stats_path = root / "meta" / "stats.json"
    if not args.all_features:
        logging.info(
            "Loading local stats repo_id=%s root=%s features=%s",
            args.repo_id,
            root,
            ",".join(feature_keys),
        )
        stats = load_stats(stats_path)
        if not args.overwrite and has_quantile_stats(stats, feature_keys):
            logging.info("Selected features already contain all required quantile statistics; skipping")
            write_manifest(
                manifest_path,
                repo_id=args.repo_id,
                root=root,
                stats_path=stats_path,
                skipped=True,
                feature_keys=feature_keys,
                mode="selected_parquet",
            )
            print(f"LEROBOT_QUANTILE_STATS_READY skipped=true stats_path={stats_path}")
            return 0

        missing_features = [key for key in feature_keys if key not in stats]
        if missing_features:
            raise KeyError(f"Selected features are missing from stats.json: {missing_features}")

        logging.info("Computing selected feature quantiles from parquet without decoding videos")
        selected_quantiles = compute_selected_quantiles_from_parquet(root, feature_keys)
        for key, quantile_stats in selected_quantiles.items():
            stats[key].update(quantile_stats)
        write_stats_json(stats_path, stats)
        write_manifest(
            manifest_path,
            repo_id=args.repo_id,
            root=root,
            stats_path=stats_path,
            skipped=False,
            feature_keys=feature_keys,
            mode="selected_parquet",
        )
        print(f"LEROBOT_QUANTILE_STATS_READY skipped=false stats_path={stats_path}")
        return 0

    logging.info(
        "Loading local LeRobot dataset repo_id=%s root=%s video_backend=%s",
        args.repo_id,
        root,
        args.video_backend,
    )
    dataset = LeRobotDataset(repo_id=args.repo_id, root=root, video_backend=args.video_backend)

    all_feature_keys = list(dataset.meta.stats.keys()) if dataset.meta.stats else []
    if not args.overwrite and has_quantile_stats(dataset.meta.stats, all_feature_keys):
        logging.info("Dataset already contains all required quantile statistics; skipping")
        write_manifest(
            manifest_path,
            repo_id=args.repo_id,
            root=root,
            stats_path=stats_path,
            skipped=True,
            feature_keys=all_feature_keys,
            mode="all_features",
        )
        print(f"LEROBOT_QUANTILE_STATS_READY skipped=true stats_path={stats_path}")
        return 0

    logging.info("Dataset is missing quantile statistics; computing q01/q10/q50/q90/q99 locally")
    new_stats = compute_quantile_stats_for_dataset(dataset)
    dataset.meta.stats = new_stats
    write_stats(new_stats, dataset.meta.root)
    write_manifest(
        manifest_path,
        repo_id=args.repo_id,
        root=root,
        stats_path=stats_path,
        skipped=False,
        feature_keys=all_feature_keys,
        mode="all_features",
    )
    print(f"LEROBOT_QUANTILE_STATS_READY skipped=false stats_path={stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
