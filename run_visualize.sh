#!/usr/bin/env bash
set -euo pipefail

NB="visualize_cd4mt.ipynb"
OUT="visualize_cd4mt.out.ipynb"

if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate cdp10
else
  echo "micromamba not found in PATH. Please ensure it's installed." >&2
  exit 1
fi

if command -v jupyter >/dev/null 2>&1; then
  jupyter nbconvert --to notebook --execute "$NB" --output "$OUT" \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0
elif command -v papermill >/dev/null 2>&1; then
  papermill "$NB" "$OUT"
else
  echo "Neither jupyter nor papermill found. Install one in env 'cdp10'." >&2
  exit 2
fi

echo "Done. Executed notebook saved to: $OUT"

