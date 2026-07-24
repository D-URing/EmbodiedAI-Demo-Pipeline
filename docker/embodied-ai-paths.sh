#!/bin/sh

if [ -t 1 ] \
  && [ -z "${EMBODIED_PATHS_NOTICE_SHOWN:-}" ] \
  && [ -r /etc/embodied-ai/paths.txt ]; then
  EMBODIED_PATHS_NOTICE_SHOWN=1
  export EMBODIED_PATHS_NOTICE_SHOWN
  printf '\n'
  cat /etc/embodied-ai/paths.txt
  printf '\n'
fi
