#!/usr/bin/env bash
# D-M exploration with a shared reference range of [0.2, 0.8].
# Positional arguments and environment overrides are forwarded to the core runner.
set -euo pipefail

quantile_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export QUANTILE_EVAL_VARIANTS=${QUANTILE_EVAL_VARIANTS-D E F F0 G H I J K L M}
export QUANTILE_EVAL_MIN_LEVEL=${QUANTILE_EVAL_MIN_LEVEL:-0.2}
export QUANTILE_EVAL_MAX_LEVEL=${QUANTILE_EVAL_MAX_LEVEL:-0.8}
export QUANTILE_EVAL_TEMPERATURE=${QUANTILE_EVAL_TEMPERATURE:-0.35}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'D-M defaults: variants D E F F0 G H I J K L M; reference range [0.2,0.8]; temperature 0.35.\n'
    printf 'All defaults can be overridden with the QUANTILE_* variables below.\n\n'
fi
exec bash "$quantile_script_dir/eval_quantile_ablations.sh" "$@"
