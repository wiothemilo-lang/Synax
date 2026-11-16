#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv; run 'make install' first" >&2
  exit 2
fi

.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python -m synax.train.pretrain --config configs/tiny.yaml --synthetic --steps 50
