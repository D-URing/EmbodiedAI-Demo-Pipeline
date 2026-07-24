#!/usr/bin/env bash
set -euo pipefail

case "${EMBODIED_ENV:-core}" in
  core|lerobot|imagewam)
    ;;
  *)
    echo "EMBODIED_ENV must be one of: core, lerobot, imagewam" >&2
    exit 2
    ;;
esac

export VIRTUAL_ENV="/opt/venvs/${EMBODIED_ENV:-core}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

if [[ "${EMBODIED_ENV:-core}" == "imagewam" ]]; then
  export IMAGEWAM_WORKDIR="${IMAGEWAM_WORKDIR:-${PROJECT_ROOT:-/opt/embodied-ai-demo-pipeline}/upstreams/ImageWAM}"
  export IMAGEWAM_FLUX2_SRC="${IMAGEWAM_FLUX2_SRC:-${IMAGEWAM_WORKDIR}/third_party/flux2}"
  export IMAGEWAM_LIBERO_SRC="${IMAGEWAM_LIBERO_SRC:-${IMAGEWAM_WORKDIR}/third_party/LIBERO}"
  export PYTHONPATH="${IMAGEWAM_LIBERO_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
fi

exec "$@"
