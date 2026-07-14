#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

XPOLICYLAB_REPO="${XPOLICYLAB_REPO:-https://github.com/XPolicyLab/XPolicyLab.git}"
XPOLICYLAB_COMMIT="${XPOLICYLAB_COMMIT:-fe71eb54675cef495fea817a637386a4f4529153}"
REFERENCE_CACHE_DIR="${REFERENCE_CACHE_DIR:-$REPO_ROOT/upstreams}"
TARGET_DIR="${TARGET_DIR:-$REFERENCE_CACHE_DIR/XPolicyLab}"

mkdir -p "$(dirname "$TARGET_DIR")"

if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone --no-checkout "$XPOLICYLAB_REPO" "$TARGET_DIR"
fi

git -C "$TARGET_DIR" fetch --depth 1 origin "$XPOLICYLAB_COMMIT" || \
  git -C "$TARGET_DIR" fetch --depth 1 origin main
git -C "$TARGET_DIR" checkout --detach "$XPOLICYLAB_COMMIT"

echo "XPolicyLab reference checkout is ready:"
echo "  path: $TARGET_DIR"
echo "  commit: $XPOLICYLAB_COMMIT"
echo
echo "Reference files to inspect first:"
echo "  policy/demo_policy/README.md"
echo "  policy/demo_policy/model.py"
echo "  policy/demo_policy/deploy.py"
echo "  policy/demo_policy/deploy.yml"
echo
echo "Manual upstream debug command, after preparing the upstream environments:"
echo "  cd $TARGET_DIR/policy/demo_policy"
echo "  EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>"
