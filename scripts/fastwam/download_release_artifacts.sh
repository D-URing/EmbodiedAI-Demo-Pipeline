#!/usr/bin/env bash
# 下载 custom FastWAM 路线需要的模型资产。
#
# 这个脚本由 make download-fastwam-artifacts / make prepare-assets-custom-fastwam 调用。
# 它会下载两类东西：
#   1. FastWAM release checkpoint 和 dataset stats；
#   2. 训练/预计算 text cache 必需的 Wan2.2 VAE、Wan2.2 T5 text encoder、Wan2.1 tokenizer。
#
# 默认优先使用 /home/scut/hfd.sh；如果存在 modelscope，也可用于 Wan runtime assets。
# 所有文件都落到项目内 models/，不会提交进 git。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EMBODIED_MODEL_ROOT="${EMBODIED_MODEL_ROOT:-$REPO_ROOT/models}"
EMBODIED_RUN_ROOT="${EMBODIED_RUN_ROOT:-$REPO_ROOT/runs}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

FASTWAM_RELEASE_REPO_ID="${FASTWAM_RELEASE_REPO_ID:-yuanty/fastwam}"
FASTWAM_RELEASE_LOCAL_DIR="${FASTWAM_RELEASE_LOCAL_DIR:-$EMBODIED_MODEL_ROOT/custom/fastwam/release}"
FASTWAM_RELEASE_FILES="${FASTWAM_RELEASE_FILES:-libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json}"
FASTWAM_DOWNLOAD_RUNTIME_ASSETS="${FASTWAM_DOWNLOAD_RUNTIME_ASSETS:-1}"
FASTWAM_WAN_MODEL_ID="${FASTWAM_WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
FASTWAM_WAN_MODEL_LOCAL_DIR="${FASTWAM_WAN_MODEL_LOCAL_DIR:-$EMBODIED_MODEL_ROOT/Wan-AI/Wan2.2-TI2V-5B}"
FASTWAM_WAN_MODEL_FILES="${FASTWAM_WAN_MODEL_FILES:-Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth}"
FASTWAM_TOKENIZER_MODEL_ID="${FASTWAM_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
FASTWAM_TOKENIZER_LOCAL_DIR="${FASTWAM_TOKENIZER_LOCAL_DIR:-$EMBODIED_MODEL_ROOT/Wan-AI/Wan2.1-T2V-1.3B}"
FASTWAM_TOKENIZER_INCLUDE="${FASTWAM_TOKENIZER_INCLUDE:-google/umt5-xxl/**}"
FASTWAM_RUNTIME_DOWNLOADER="${FASTWAM_RUNTIME_DOWNLOADER:-auto}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_CLI_BIN="${HF_CLI_BIN:-}"
HFD_BIN_WAS_SET="${HFD_BIN+x}"
HFD_BIN="${HFD_BIN:-/home/scut/hfd.sh}"
HFD_THREADS="${HFD_THREADS:-10}"
HFD_JOBS="${HFD_JOBS:-2}"
HFD_TOOL="${HFD_TOOL:-aria2c}"

if [[ -f "$HFD_BIN" ]]; then
  DOWNLOADER_KIND="hfd"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
elif [[ -n "$HFD_BIN_WAS_SET" ]]; then
  echo "HFD_BIN=$HFD_BIN does not exist. Unset HFD_BIN to fall back to hf/huggingface-cli." >&2
  exit 127
elif [[ -n "$HF_CLI_BIN" ]]; then
  if ! command -v "$HF_CLI_BIN" >/dev/null 2>&1; then
    echo "HF_CLI_BIN=$HF_CLI_BIN is not executable or not on PATH." >&2
    exit 127
  fi
  DOWNLOADER_KIND="hf_cli"
  HF_DOWNLOAD_CMD=("$HF_CLI_BIN" download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOADER_KIND="hf_cli"
  HF_DOWNLOAD_CMD=(huggingface-cli download)
elif command -v hf >/dev/null 2>&1; then
  DOWNLOADER_KIND="hf_cli"
  HF_DOWNLOAD_CMD=(hf download)
else
  echo "A Hugging Face downloader is required." >&2
  echo "Expected /home/scut/hfd.sh, HFD_BIN=/path/to/hfd.sh, 'huggingface-cli', or the newer 'hf' command on PATH." >&2
  exit 127
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required to write artifact manifests. Set PYTHON_BIN=/path/to/python if needed." >&2
  exit 127
fi

mkdir -p "$FASTWAM_RELEASE_LOCAL_DIR" "$EMBODIED_RUN_ROOT/artifact_manifests"

read -r -a release_files <<< "$FASTWAM_RELEASE_FILES"
if [[ "${#release_files[@]}" -eq 0 ]]; then
  echo "FASTWAM_RELEASE_FILES must contain at least one filename." >&2
  exit 2
fi
read -r -a wan_model_files <<< "$FASTWAM_WAN_MODEL_FILES"
read -r -a tokenizer_includes <<< "$FASTWAM_TOKENIZER_INCLUDE"

MANIFEST_PATH="$EMBODIED_RUN_ROOT/artifact_manifests/fastwam_release_artifacts_manifest.json"

print_download_failure_help() {
  cat >&2 <<EOF

[error] FastWAM release download failed.
[target] $FASTWAM_RELEASE_LOCAL_DIR
[manifest-if-success] $MANIFEST_PATH

This usually means the current cluster node cannot reach Hugging Face.
Quick checks:
  command -v curl >/dev/null 2>&1 && curl -I "\${HF_ENDPOINT:-https://huggingface.co}"
  env | grep -E '^(HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|HF_ENDPOINT)='

Possible fixes:
  1. Run this command on a login/compute node with outbound network.
  2. Configure the cluster proxy, for example:
       export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
       export HTTP_PROXY=http://<proxy-host>:<proxy-port>
  3. If your cluster uses a Hugging Face mirror:
       export HF_ENDPOINT=https://<your-hf-mirror>
  4. Or download the files elsewhere and copy them into:
       $FASTWAM_RELEASE_LOCAL_DIR
EOF
}

echo "[artifact] family=fastwam_release"
echo "[artifact] repo=$FASTWAM_RELEASE_REPO_ID"
echo "[artifact] files=${release_files[*]}"
echo "[artifact] local_dir=$FASTWAM_RELEASE_LOCAL_DIR"
echo "[artifact] manifest=$MANIFEST_PATH"
if [[ "$DOWNLOADER_KIND" == "hfd" ]]; then
  echo "[artifact] downloader=hfd"
  echo "[artifact] hfd_bin=$HFD_BIN"
  echo "[artifact] hfd_tool=$HFD_TOOL"
  echo "[artifact] hfd_threads=$HFD_THREADS"
  echo "[artifact] hfd_jobs=$HFD_JOBS"
else
  echo "[artifact] downloader=hf_cli"
  echo "[artifact] hf_cli=${HF_DOWNLOAD_CMD[*]}"
fi
echo "[artifact] hf_endpoint=${HF_ENDPOINT:-https://huggingface.co}"
echo "[artifact] hf_home=$HF_HOME"
echo "[artifact] hf_hub_cache=$HUGGINGFACE_HUB_CACHE"
echo "[download] FastWAM release: $FASTWAM_RELEASE_REPO_ID -> $FASTWAM_RELEASE_LOCAL_DIR"
if [[ "$DOWNLOADER_KIND" == "hfd" ]]; then
  if ! HF_ENDPOINT="$HF_ENDPOINT" \
    bash "$HFD_BIN" "$FASTWAM_RELEASE_REPO_ID" \
      --include "${release_files[@]}" \
      --local-dir "$FASTWAM_RELEASE_LOCAL_DIR" \
      --tool "$HFD_TOOL" \
      -x "$HFD_THREADS" \
      -j "$HFD_JOBS"; then
    print_download_failure_help
    exit 1
  fi
else
  if ! HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
    "${HF_DOWNLOAD_CMD[@]}" "$FASTWAM_RELEASE_REPO_ID" \
      "${release_files[@]}" \
      --local-dir "$FASTWAM_RELEASE_LOCAL_DIR"; then
    print_download_failure_help
    exit 1
  fi
fi

FASTWAM_RELEASE_REPO_ID="$FASTWAM_RELEASE_REPO_ID" \
FASTWAM_RELEASE_LOCAL_DIR="$FASTWAM_RELEASE_LOCAL_DIR" \
FASTWAM_RELEASE_FILES="$FASTWAM_RELEASE_FILES" \
"$PYTHON_BIN" - "$MANIFEST_PATH" <<PY
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

files = os.environ["FASTWAM_RELEASE_FILES"].split()
local_dir = os.environ["FASTWAM_RELEASE_LOCAL_DIR"]

manifest = {
    "artifact_family": "fastwam_release",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repo_id": os.environ["FASTWAM_RELEASE_REPO_ID"],
    "local_dir": local_dir,
    "files": [{"name": name, "path": f"{local_dir.rstrip('/')}/{name}"} for name in files],
    "next_env": {
        "FASTWAM_MODEL_BASE": os.path.dirname(local_dir.rstrip("/")),
        "FASTWAM_RELEASE_CKPT": f"{local_dir.rstrip('/')}/libero_uncond_2cam224.pt",
        "FASTWAM_RELEASE_DATASET_STATS": f"{local_dir.rstrip('/')}/libero_uncond_2cam224_dataset_stats.json",
    },
}

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
    f.write("\\n")
PY

echo "[manifest] $MANIFEST_PATH"
echo
echo "Next FastWAM overlay env:"
echo "  export FASTWAM_MODEL_BASE=\"$(dirname "$FASTWAM_RELEASE_LOCAL_DIR")\""
echo "  export FASTWAM_RELEASE_CKPT=\"$FASTWAM_RELEASE_LOCAL_DIR/libero_uncond_2cam224.pt\""
echo "  export FASTWAM_RELEASE_DATASET_STATS=\"$FASTWAM_RELEASE_LOCAL_DIR/libero_uncond_2cam224_dataset_stats.json\""

runtime_downloader_kind=""
if [[ "$FASTWAM_RUNTIME_DOWNLOADER" == "auto" ]]; then
  if command -v modelscope >/dev/null 2>&1; then
    runtime_downloader_kind="modelscope"
  else
    runtime_downloader_kind="$DOWNLOADER_KIND"
  fi
else
  runtime_downloader_kind="$FASTWAM_RUNTIME_DOWNLOADER"
fi

download_runtime_files() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local files=("$@")
  mkdir -p "$local_dir"

  case "$runtime_downloader_kind" in
    modelscope)
      command -v modelscope >/dev/null 2>&1 || { echo "modelscope is required for FASTWAM_RUNTIME_DOWNLOADER=modelscope" >&2; exit 127; }
      modelscope download "$repo_id" "${files[@]}" --local_dir "$local_dir"
      ;;
    hfd)
      HF_ENDPOINT="$HF_ENDPOINT" \
        bash "$HFD_BIN" "$repo_id" \
          --include "${files[@]}" \
          --local-dir "$local_dir" \
          --tool "$HFD_TOOL" \
          -x "$HFD_THREADS" \
          -j "$HFD_JOBS"
      ;;
    hf_cli)
      HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
        "${HF_DOWNLOAD_CMD[@]}" "$repo_id" "${files[@]}" --local-dir "$local_dir"
      ;;
    *)
      echo "FASTWAM_RUNTIME_DOWNLOADER must be auto|modelscope|hfd|hf_cli, got $runtime_downloader_kind" >&2
      exit 2
      ;;
  esac
}

download_runtime_include() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local includes=("$@")
  mkdir -p "$local_dir"

  case "$runtime_downloader_kind" in
    modelscope)
      command -v modelscope >/dev/null 2>&1 || { echo "modelscope is required for FASTWAM_RUNTIME_DOWNLOADER=modelscope" >&2; exit 127; }
      modelscope download "$repo_id" --include "${includes[@]}" --local_dir "$local_dir"
      ;;
    hfd)
      HF_ENDPOINT="$HF_ENDPOINT" \
        bash "$HFD_BIN" "$repo_id" \
          --include "${includes[@]}" \
          --local-dir "$local_dir" \
          --tool "$HFD_TOOL" \
          -x "$HFD_THREADS" \
          -j "$HFD_JOBS"
      ;;
    hf_cli)
      HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
        "${HF_DOWNLOAD_CMD[@]}" "$repo_id" --include "${includes[@]}" --local-dir "$local_dir"
      ;;
    *)
      echo "FASTWAM_RUNTIME_DOWNLOADER must be auto|modelscope|hfd|hf_cli, got $runtime_downloader_kind" >&2
      exit 2
      ;;
  esac
}

if [[ "$FASTWAM_DOWNLOAD_RUNTIME_ASSETS" == "1" ]]; then
  echo
  echo "[download] FastWAM runtime Wan assets"
  echo "[artifact] runtime_downloader=$runtime_downloader_kind"
  echo "[artifact] wan_model=$FASTWAM_WAN_MODEL_ID -> $FASTWAM_WAN_MODEL_LOCAL_DIR"
  echo "[artifact] wan_files=${wan_model_files[*]}"
  download_runtime_files "$FASTWAM_WAN_MODEL_ID" "$FASTWAM_WAN_MODEL_LOCAL_DIR" "${wan_model_files[@]}"

  echo "[artifact] tokenizer_model=$FASTWAM_TOKENIZER_MODEL_ID -> $FASTWAM_TOKENIZER_LOCAL_DIR"
  echo "[artifact] tokenizer_include=${tokenizer_includes[*]}"
  download_runtime_include "$FASTWAM_TOKENIZER_MODEL_ID" "$FASTWAM_TOKENIZER_LOCAL_DIR" "${tokenizer_includes[@]}"

  echo
  echo "FastWAM runtime assets ready:"
  echo "  $FASTWAM_WAN_MODEL_LOCAL_DIR/Wan2.2_VAE.pth"
  echo "  $FASTWAM_WAN_MODEL_LOCAL_DIR/models_t5_umt5-xxl-enc-bf16.pth"
  echo "  $FASTWAM_TOKENIZER_LOCAL_DIR/google/umt5-xxl/"
else
  echo
  echo "FASTWAM_DOWNLOAD_RUNTIME_ASSETS=0, skipped Wan VAE/text encoder/tokenizer assets."
fi
