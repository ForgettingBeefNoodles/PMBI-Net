#!/usr/bin/env bash
set -euo pipefail
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
command -v python >/dev/null || { echo "Activate your Python environment" >&2; exit 1; }
python run.py --config configs/pmbi_hparams.py --exp_key PEMS04_96 --model PMBI_Net --is_training 1 "$@"
