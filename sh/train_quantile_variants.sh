#!/usr/bin/env bash
# Train one QuantileHead quantile ablation from the common base checkpoint.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash sh/train_quantile_variants.sh VARIANT [--key=value ...]

VARIANT (required):
  standard              Original monotone multi-quantile model (control).
  median_only           Same architecture; supervise only Q(0.5).
  stop_median_gradient  Detach the median anchor in non-median quantiles.
  independent           Predict non-median quantiles independently; crossings allowed.

Defaults match the current AutoDL training recipe: 2 GPUs, 6000 steps,
batch_size=64, 16 action prompt tokens, QUANTILES normalization, loss mask=0.1.
All variants use the quantile regression head and deterministic median inference.
Each invocation trains one model; existing output directories are rejected.
The script activates conda environment lerobot06, clears offline/proxy variables,
and sets the HF mirror and cache directories under /root/autodl-tmp automatically.

Environment overrides:
  QUANTILE_TRAIN_SEED             Training seed (default: 1000)
  QUANTILE_TRAIN_NUM_PROCESSES    GPU process count (default: 2)
  CUDA_VISIBLE_DEVICES           Visible GPUs (default: 0,1)
  QUANTILE_TRAIN_DATASET_REPO_ID  Dataset repo ID (default: libero)
  QUANTILE_TRAIN_DATASET_ROOT     Dataset path (default: /root/autodl-fs/libero)
  QUANTILE_TRAIN_PRETRAINED_PATH  Common base checkpoint
                                (default: /root/autodl-tmp/hf-cache/checkpoints/pi05_libero_base)
  QUANTILE_TRAIN_OUTPUT_ROOT      Parent output directory (default: /root/autodl-fs/outputs)
  QUANTILE_TRAIN_CONDA_SH         Optional path to etc/profile.d/conda.sh
  QUANTILE_TRAIN_RUNNER           active (default: activated lerobot06) or uv (uv run --no-sync)
  QUANTILE_TRAIN_DRY_RUN          1 previews setup/command without activating conda,
                                creating files, or training

Additional arguments must use --key=value. Ordinary settings, including --seed
and --output_dir, override the defaults. Variant/head/inference mode and fresh-run
settings are fixed so this wrapper cannot silently train a different model or resume.

Examples:
  bash sh/train_quantile_variants.sh median_only
  QUANTILE_TRAIN_SEED=1001 bash sh/train_quantile_variants.sh independent
  QUANTILE_TRAIN_DRY_RUN=1 bash sh/train_quantile_variants.sh stop_median_gradient
  QUANTILE_TRAIN_RUNNER=active bash sh/train_quantile_variants.sh standard --steps=6000
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

prepare_training_environment() {
    local quantile_conda_sh=${QUANTILE_TRAIN_CONDA_SH:-}
    local quantile_conda_base=
    local quantile_conda_candidate
    if [[ -z $quantile_conda_sh ]]; then
        if [[ -n ${CONDA_EXE:-} && -x $CONDA_EXE ]]; then
            quantile_conda_base=$("$CONDA_EXE" info --base) || die "Cannot locate conda base via CONDA_EXE"
        elif command -v conda >/dev/null 2>&1; then
            quantile_conda_base=$(conda info --base) || die "Cannot locate conda base"
        fi
        if [[ -n $quantile_conda_base ]]; then
            quantile_conda_sh=$quantile_conda_base/etc/profile.d/conda.sh
        else
            for quantile_conda_candidate in \
                "${HOME:-/root}/miniconda3" "${HOME:-/root}/anaconda3" /opt/conda; do
                if [[ -r $quantile_conda_candidate/etc/profile.d/conda.sh ]]; then
                    quantile_conda_sh=$quantile_conda_candidate/etc/profile.d/conda.sh
                    break
                fi
            done
        fi
    fi
    [[ -n $quantile_conda_sh && -r $quantile_conda_sh ]] || \
        die "Cannot find conda.sh. Set QUANTILE_TRAIN_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh"

    # Noninteractive bash does not load conda's shell function. Some conda
    # activation hooks also reference unset variables, so suspend nounset here.
    set +u
    source "$quantile_conda_sh" || die "Cannot initialize conda from $quantile_conda_sh"
    conda activate lerobot06 || die "Cannot activate conda environment lerobot06"
    set -u
    hash -r

    unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_HOME=/root/autodl-tmp/hf-cache
    export HF_DATASETS_CACHE=/root/autodl-tmp/hf-cache/datasets
    export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
    export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache
    export TORCH_HOME=/root/autodl-tmp/torch-cache
    export TMPDIR=/root/autodl-tmp/tmp
    mkdir -p -- "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$TMPDIR" || \
        die "Cannot create cache or temporary directories under /root/autodl-tmp"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
quantile_train_variant=$1
shift
case "$quantile_train_variant" in
    standard|median_only|stop_median_gradient|independent) ;;
    *) die "Unknown variant: $quantile_train_variant" ;;
esac

quantile_train_seed=${QUANTILE_TRAIN_SEED:-1000}
quantile_train_processes=${QUANTILE_TRAIN_NUM_PROCESSES:-2}
quantile_train_runner=${QUANTILE_TRAIN_RUNNER:-active}
quantile_train_dry_run=${QUANTILE_TRAIN_DRY_RUN:-0}
quantile_train_output_override=
quantile_train_extra_args=()
for quantile_train_arg in "$@"; do
    [[ $quantile_train_arg == --*=* ]] || die "Use --key=value for overrides: $quantile_train_arg"
    case "$quantile_train_arg" in
        --seed=*) quantile_train_seed=${quantile_train_arg#*=} ;;
        --output_dir=*)
            quantile_train_output_override=${quantile_train_arg#*=}
            [[ -n $quantile_train_output_override ]] || die "output_dir cannot be empty"
            ;;
        --policy.type=*|--policy.marginal_quantile_variant=*|--policy.use_pure_regression_head=*|\
        --policy.action_generation_mode=*|--policy.marginal_quantile_inference_mode=*|\
        --policy.path=*|--config_path=*|--resume=*)
            die "This wrapper fixes the selected variant and starts fresh; unsupported override: $quantile_train_arg"
            ;;
        *) quantile_train_extra_args+=("$quantile_train_arg") ;;
    esac
done
[[ $quantile_train_seed =~ ^[0-9]+$ ]] || die "Training seed must be a nonnegative integer"
[[ $quantile_train_processes =~ ^[1-9][0-9]*$ ]] || die "GPU process count must be a positive integer"
[[ $quantile_train_dry_run == 0 || $quantile_train_dry_run == 1 ]] || die "QUANTILE_TRAIN_DRY_RUN must be 0 or 1"

quantile_train_output=${quantile_train_output_override:-${QUANTILE_TRAIN_OUTPUT_ROOT:-/root/autodl-fs/outputs}/libero_quantile_${quantile_train_variant}_seed${quantile_train_seed}}
[[ ! -e $quantile_train_output && ! -L $quantile_train_output ]] || die "Output already exists: $quantile_train_output. Choose a new --output_dir."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

case "$quantile_train_runner" in
    uv) quantile_train_command=(uv run --no-sync accelerate launch) ;;
    active) quantile_train_command=(accelerate launch) ;;
    *) die "QUANTILE_TRAIN_RUNNER must be uv or active" ;;
esac
if (( quantile_train_processes > 1 )); then
    quantile_train_command+=(--multi_gpu)
fi
quantile_train_command+=(--num_processes "$quantile_train_processes" -m lerobot.scripts.lerobot_train)
quantile_train_args=(
    "--dataset.repo_id=${QUANTILE_TRAIN_DATASET_REPO_ID:-libero}"
    --dataset.video_backend=pyav
    "--dataset.root=${QUANTILE_TRAIN_DATASET_ROOT:-/root/autodl-fs/libero}"
    --policy.type=quantile_head
    "--output_dir=$quantile_train_output"
    "--job_name=quantile_head_quantile_${quantile_train_variant}_seed${quantile_train_seed}"
    --resume=false
    --policy.push_to_hub=false
    "--policy.pretrained_path=${QUANTILE_TRAIN_PRETRAINED_PATH:-/root/autodl-tmp/hf-cache/checkpoints/pi05_libero_base}"
    --policy.action_generation_mode=regression
    --policy.use_pure_regression_head=false
    "--policy.marginal_quantile_variant=$quantile_train_variant"
    --policy.marginal_quantile_inference_mode=median
    --policy.marginal_quantile_num_knots=21
    --policy.marginal_quantile_min_level=0.025
    --policy.marginal_quantile_initial_gap=0.02
    --policy.soft_prompt_len=16
    --policy.train_soft_prompt=true
    --policy.action_loss_mask_ratio=0.1
    '--policy.normalization_mapping={"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}'
    --policy.compile_model=true
    --policy.gradient_checkpointing=true
    --policy.train_expert_only=true
    --policy.detach_action_from_vlm=false
    --policy.device=cuda
    --wandb.enable=false
    --steps=6000
    --batch_size=64
    "--seed=$quantile_train_seed"
)

# Replace matching defaults instead of relying on how the CLI handles duplicate flags.
for quantile_train_arg in ${quantile_train_extra_args[@]+"${quantile_train_extra_args[@]}"}; do
    quantile_train_key=${quantile_train_arg%%=*}
    quantile_train_replaced=0
    for quantile_train_index in "${!quantile_train_args[@]}"; do
        if [[ ${quantile_train_args[$quantile_train_index]%%=*} == "$quantile_train_key" ]]; then
            quantile_train_args[$quantile_train_index]=$quantile_train_arg
            quantile_train_replaced=1
            break
        fi
    done
    if [[ $quantile_train_replaced == 0 ]]; then
        quantile_train_args+=("$quantile_train_arg")
    fi
done
quantile_train_command+=("${quantile_train_args[@]}")
printf 'Variant: %s\nOutput: %s\n' "$quantile_train_variant" "$quantile_train_output"
printf 'Environment: conda activate lerobot06; offline/proxy variables cleared\n'
printf 'HF endpoint: https://hf-mirror.com; HF/Torch/tmp directories: /root/autodl-tmp\nCommand:\n'
printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
printf '%q ' "${quantile_train_command[@]}"
printf '\n'
if [[ $quantile_train_dry_run == 1 ]]; then
    exit 0
fi
prepare_training_environment
command -v "${quantile_train_command[0]}" >/dev/null || die "Missing executable: ${quantile_train_command[0]}"
exec "${quantile_train_command[@]}"
