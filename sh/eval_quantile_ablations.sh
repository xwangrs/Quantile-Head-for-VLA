#!/usr/bin/env bash
# Evaluate one quantile checkpoint with selected decoders, sequentially on one device.
# Usage: bash sh/eval_quantile_ablations.sh [CHECKPOINT] [OUTPUT_ROOT] [EPISODES_PER_TASK] [SAMPLER_SEED ...]
# With no arguments, run A/B/C on libero_quantile_action_prompt, 50 episodes per task and sampler seed 0.
# Example: bash sh/eval_quantile_ablations.sh /path/to/pretrained_model outputs/quantile_sampling 50 0
# Optional: QUANTILE_EVAL_VARIANTS="D E F F0 G H I J K L M" QUANTILE_DRY_RUN=1
# A/L are deterministic and run only for the first seed; other variants run for each seed.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    printf 'Usage: bash %s [CHECKPOINT] [OUTPUT_ROOT] [EPISODES_PER_TASK=50] [SAMPLER_SEED ...; default=0]\n' "$0"
    printf 'Default checkpoint: /root/autodl-fs/outputs/libero_quantile_action_prompt/checkpoints/last/pretrained_model\n'
    printf 'Default output: /root/autodl-fs/quantile_sampling_YYYYMMDD_HHMMSS\n'
    printf 'Options: QUANTILE_EVAL_SEED=1000 QUANTILE_EVAL_VIZ=0 QUANTILE_DRY_RUN=0\n'
    printf 'QUANTILE_EVAL_VARIANTS selects an ordered, space-separated subset; default: %s\n' "${QUANTILE_EVAL_VARIANTS-A B C}"
    printf 'A: median; B: chunk-shared ranks with rho=0; C: chunk-shared ranks with rho=0.7.\n'
    printf 'Allowed variants: A B C C_all D E F F0 G H I J K L M. Empty lists and duplicates are rejected.\n'
    printf 'C excludes the gripper from sampling; C_all samples all action dimensions, including the gripper.\n'
    printf 'D: episode-fixed ranks; E: independent ranks per step; F/F0: Gaussian decoder, rho=0.7/0.\n'
    printf 'G/H/I: rho=0 with ranges [0.2,0.8]/[0.3,0.7]/[0.4,0.6]; J/K: rho=0 with temperature 0.15/1.\n'
    printf 'L: deterministic quantile expectation; M: rho=0 with fixed calibrated quantile widths.\n'
    printf 'Shared settings: QUANTILE_EVAL_MIN_LEVEL=%s QUANTILE_EVAL_MAX_LEVEL=%s QUANTILE_EVAL_TEMPERATURE=%s\n' \
        "${QUANTILE_EVAL_MIN_LEVEL:-0.1}" "${QUANTILE_EVAL_MAX_LEVEL:-0.9}" "${QUANTILE_EVAL_TEMPERATURE:-0.35}"
    printf 'L: QUANTILE_EXPECTATION_NODES=128. M: QUANTILE_FIXED_WIDTH_PATH=/path/to/fixed_width.json\n'
    printf 'Without a width file, M first calibrates with random initial states: QUANTILE_CALIBRATION_EPISODES=5 QUANTILE_CALIBRATION_SEED=2000\n'
    printf 'See sh/eval_quantile_dm.md for controlled comparisons.\n'
    exit 0
fi


quantile_checkpoint=${1:-/root/autodl-fs/outputs/libero_quantile_action_prompt/checkpoints/last/pretrained_model}
quantile_output_root=${2:-/root/autodl-fs/quantile_sampling_$(date +%Y%m%d_%H%M%S)}
quantile_episodes=${3:-50}
if [[ $# -ge 3 ]]; then
    shift 3
else
    set --
fi
if [[ $# -eq 0 ]]; then
    set -- 0
fi
quantile_eval_seed=${QUANTILE_EVAL_SEED:-1000}
quantile_variant_list=${QUANTILE_EVAL_VARIANTS-A B C}
quantile_reference_min=${QUANTILE_EVAL_MIN_LEVEL:-0.1}
quantile_reference_max=${QUANTILE_EVAL_MAX_LEVEL:-0.9}
quantile_reference_temperature=${QUANTILE_EVAL_TEMPERATURE:-0.35}
quantile_expectation_nodes=${QUANTILE_EXPECTATION_NODES:-128}
quantile_width_path=${QUANTILE_FIXED_WIDTH_PATH:-}
quantile_calibration_episodes=${QUANTILE_CALIBRATION_EPISODES:-5}
quantile_calibration_seed=${QUANTILE_CALIBRATION_SEED:-2000}
quantile_dry_run=${QUANTILE_DRY_RUN:-0}
quantile_viz=${QUANTILE_EVAL_VIZ:-0}

# Keep the caller's order without shell glob expansion or silently dropping extra lines.
if [[ $quantile_variant_list == *$'\n'* || $quantile_variant_list == *$'\r'* ]]; then
    printf 'QUANTILE_EVAL_VARIANTS must be a single space-separated line.\n' >&2
    exit 2
fi
IFS=' ' read -r -a quantile_variants <<< "$quantile_variant_list"
if [[ ${#quantile_variants[@]} -eq 0 ]]; then
    printf 'QUANTILE_EVAL_VARIANTS must contain at least one variant.\n' >&2
    exit 2
fi
quantile_seen_variants=' '
for quantile_variant in "${quantile_variants[@]}"; do
    case "$quantile_variant" in
        A|B|C|C_all|D|E|F|F0|G|H|I|J|K|L|M) ;;
        *) printf 'Unknown QUANTILE_EVAL_VARIANTS entry: %s\n' "$quantile_variant" >&2; exit 2 ;;
    esac
    if [[ $quantile_seen_variants == *" $quantile_variant "* ]]; then
        printf 'Duplicate QUANTILE_EVAL_VARIANTS entry: %s\n' "$quantile_variant" >&2
        exit 2
    fi
    quantile_seen_variants+="$quantile_variant "
done

for quantile_flag in "$quantile_dry_run" "$quantile_viz"; do
    if [[ $quantile_flag != 0 && $quantile_flag != 1 ]]; then
        printf 'QUANTILE_DRY_RUN and QUANTILE_EVAL_VIZ must be 0 or 1.\n' >&2
        exit 2
    fi
done
if [[ ! $quantile_eval_seed =~ ^[0-9]+$ ]]; then
    printf 'QUANTILE_EVAL_SEED must be a non-negative integer.\n' >&2
    exit 2
fi
quantile_float_pattern='^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
for quantile_number in "$quantile_reference_min" "$quantile_reference_max" "$quantile_reference_temperature"; do
    if [[ ! $quantile_number =~ $quantile_float_pattern ]]; then
        printf 'Quantile min/max/temperature must be finite, non-negative numbers: %s\n' "$quantile_number" >&2
        exit 2
    fi
done
if ! awk -v lo="$quantile_reference_min" -v hi="$quantile_reference_max" -v temp="$quantile_reference_temperature" \
    'BEGIN { exit !(lo >= 0 && lo < 0.5 && hi > 0.5 && hi <= 1 && temp >= 0 && temp < 1e308) }'; then
    printf 'Require 0 <= QUANTILE_EVAL_MIN_LEVEL < 0.5 < QUANTILE_EVAL_MAX_LEVEL <= 1 and finite temperature >= 0.\n' >&2
    exit 2
fi
if [[ $quantile_seen_variants == *' L '* ]] && \
    { [[ ! $quantile_expectation_nodes =~ ^[1-9][0-9]*$ ]] || (( quantile_expectation_nodes < 2 )); }; then
    printf 'QUANTILE_EXPECTATION_NODES must be an integer >= 2.\n' >&2
    exit 2
fi
if [[ $quantile_seen_variants == *' M '* ]]; then
    if [[ -z $quantile_width_path ]]; then
        if [[ ! $quantile_calibration_episodes =~ ^[1-9][0-9]*$ || ! $quantile_calibration_seed =~ ^[0-9]+$ ]]; then
            printf 'QUANTILE_CALIBRATION_EPISODES must be positive; QUANTILE_CALIBRATION_SEED must be non-negative integers.\n' >&2
            exit 2
        fi
    elif [[ $quantile_dry_run != 1 && ! -f $quantile_width_path ]]; then
        printf 'QUANTILE_FIXED_WIDTH_PATH does not exist: %s\n' "$quantile_width_path" >&2
        exit 2
    fi
fi

if [[ ! $quantile_episodes =~ ^[1-9][0-9]*$ ]]; then
    printf 'EPISODES_PER_TASK must be a positive integer.\n' >&2
    exit 2
fi
quantile_seen_seeds=' '
for quantile_sampler_seed in "$@"; do
    if [[ ! $quantile_sampler_seed =~ ^[0-9]+$ ]]; then
        printf 'SAMPLER_SEED must be a non-negative integer: %s\n' "$quantile_sampler_seed" >&2
        exit 2
    fi
    if [[ $quantile_seen_seeds == *" $quantile_sampler_seed "* ]]; then
        printf 'Duplicate SAMPLER_SEED: %s\n' "$quantile_sampler_seed" >&2
        exit 2
    fi
    quantile_seen_seeds+="$quantile_sampler_seed "
done

if [[ $quantile_dry_run != 1 ]]; then
    # Keep the full run's progress and status beside its individual evaluation results.
    if [[ -e "$quantile_output_root/status.tsv" || -e "$quantile_output_root/run_all.log" ]]; then
        printf 'Use a new OUTPUT_ROOT; previous run logs already exist: %s\n' "$quantile_output_root" >&2
        exit 1
    fi
    mkdir -p "$quantile_output_root"
    printf 'Progress log: %s/run_all.log\n' "$quantile_output_root"
    exec > "$quantile_output_root/run_all.log" 2>&1
    printf 'All evaluation outputs: %s\n' "$quantile_output_root"
    printf 'Variants: %s\n' "${quantile_variants[*]}"
    printf 'B/C sample arm dimensions with the gripper fixed at its median; C_all also samples the gripper.\n'
    printf 'Checkpoint: %s\nEpisodes per task: %s\nSampler seeds: %s\n' "$quantile_checkpoint" "$quantile_episodes" "$*"
    printf 'Reference range: [%s, %s]; reference temperature: %s\n' \
        "$quantile_reference_min" "$quantile_reference_max" "$quantile_reference_temperature"
    if ! command -v lerobot-eval; then
        printf 'lerobot-eval was not found. Activate your evaluation environment and run again.\n' >&2
        exit 1
    fi
    printf 'variant\tsampler_seed\texit_code\teval_info\tlog\n' > "$quantile_output_root/status.tsv"
fi

quantile_any_failure=0
quantile_median_done=0
quantile_expectation_done=0
quantile_calibration_failed=0
if [[ $quantile_seen_variants == *' M '* && -z $quantile_width_path ]]; then
    quantile_calibration_dir="$quantile_output_root/width_calibration"
    quantile_width_path="$quantile_calibration_dir/fixed_width.json"
    quantile_calibration_cmd=(
        lerobot-eval
        --env.type=libero
        --env.task=libero_10
        --env.init_states=false
        --env.max_parallel_tasks=1
        --eval.batch_size=1
        "--eval.n_episodes=$quantile_calibration_episodes"
        "--seed=$quantile_calibration_seed"
        "--policy.path=$quantile_checkpoint"
        --policy.n_action_steps=10
        --policy.marginal_quantile_inference_mode=median
        "--policy.marginal_quantile_sampling_temperature=$quantile_reference_temperature"
        "--policy.marginal_quantile_sampling_min_level=$quantile_reference_min"
        "--policy.marginal_quantile_sampling_max_level=$quantile_reference_max"
        '--policy.marginal_quantile_sampling_exclude_joints=["gripper"]'
        --quantile_head_quantile_decoder=calibrate_width
        "--quantile_head_quantile_width_output_path=$quantile_width_path"
        "--output_dir=$quantile_calibration_dir"
    )
    if [[ $quantile_dry_run == 1 ]]; then
        printf '# Width calibration (random initial states; excluded from evaluation scores)\n'
        printf '%q ' "${quantile_calibration_cmd[@]}"
        printf '\n'
    else
        if ! mkdir "$quantile_calibration_dir"; then
            printf 'Use a new OUTPUT_ROOT; calibration directory already exists or cannot be created: %s\n' "$quantile_calibration_dir" >&2
            exit 1
        fi
        printf '%q ' "${quantile_calibration_cmd[@]}" > "$quantile_calibration_dir/command.txt"
        printf '\n' >> "$quantile_calibration_dir/command.txt"
        printf 'Calibrating fixed widths on random initial states; log: %s/eval.log\n' "$quantile_calibration_dir"
        if (unset LEROBOT_QUANTILE_HEAD_VIZ_DIR; "${quantile_calibration_cmd[@]}") > "$quantile_calibration_dir/eval.log" 2>&1; then
            quantile_calibration_exit=0
            if [[ ! -f $quantile_width_path ]]; then
                printf 'Calibration exited successfully but fixed_width.json is missing.\n' >> "$quantile_calibration_dir/eval.log"
                quantile_calibration_exit=1
            fi
        else
            quantile_calibration_exit=$?
        fi
        printf 'exit_code\twidth_file\tlog\n%s\t%s\t%s\n' "$quantile_calibration_exit" \
            "$quantile_width_path" "$quantile_calibration_dir/eval.log" > "$quantile_calibration_dir/calibration_status.tsv"
        if [[ $quantile_calibration_exit != 0 ]]; then
            quantile_calibration_failed=1
            quantile_any_failure=1
            printf 'Width calibration FAILED (exit %s); M will be skipped. Other variants will continue.\n' "$quantile_calibration_exit"
            tail -n 80 "$quantile_calibration_dir/eval.log" || true
        fi
    fi
fi

for quantile_sampler_seed in "$@"; do
    for quantile_variant in "${quantile_variants[@]}"; do
        if [[ $quantile_variant == A && $quantile_median_done == 1 ]]; then
            continue
        fi
        if [[ $quantile_variant == L && $quantile_expectation_done == 1 ]]; then
            continue
        fi
        quantile_decoder=policy
        quantile_inference_mode=temporal_sample
        quantile_rho=0.7
        quantile_exclude_joints='["gripper"]'
        quantile_min=$quantile_reference_min
        quantile_max=$quantile_reference_max
        quantile_temperature=$quantile_reference_temperature
        quantile_decoder_args=()
        case "$quantile_variant" in
            A) quantile_inference_mode=median; quantile_median_done=1 ;;
            B) quantile_rho=0.0 ;;
            C) ;; # Current policy: chunk-shared ranks with rho=0.7.
            C_all) quantile_exclude_joints='[]' ;; # Same as C, also sampling the gripper.
            D) quantile_rho=1.0 ;;
            E) quantile_decoder=independent_rank; quantile_rho=0.0 ;;
            F) quantile_decoder=gaussian_temporal ;;
            F0) quantile_decoder=gaussian_temporal; quantile_rho=0.0 ;;
            G) quantile_rho=0.0; quantile_min=0.2; quantile_max=0.8 ;;
            H) quantile_rho=0.0; quantile_min=0.3; quantile_max=0.7 ;;
            I) quantile_rho=0.0; quantile_min=0.4; quantile_max=0.6 ;;
            J) quantile_rho=0.0; quantile_temperature=0.15 ;;
            K) quantile_rho=0.0; quantile_temperature=1.0 ;;
            L)
                quantile_decoder=quantile_expectation
                quantile_rho=0.0
                quantile_expectation_done=1
                quantile_decoder_args=("--quantile_head_quantile_expectation_nodes=$quantile_expectation_nodes")
                ;;
            M)
                quantile_decoder=fixed_width
                quantile_rho=0.0
                quantile_decoder_args=("--quantile_head_quantile_fixed_width_path=$quantile_width_path")
                ;;
        esac

        quantile_run_dir="$quantile_output_root/seed_${quantile_sampler_seed}/${quantile_variant}"
        quantile_cmd=(
            lerobot-eval
            --env.type=libero
            --env.task=libero_10
            --env.init_states=true
            --env.max_parallel_tasks=1
            --eval.batch_size=1
            "--eval.n_episodes=$quantile_episodes"
            "--seed=$quantile_eval_seed"
            "--policy.path=$quantile_checkpoint"
            --policy.n_action_steps=10
            "--policy.marginal_quantile_inference_mode=$quantile_inference_mode"
            "--policy.marginal_quantile_sampling_temperature=$quantile_temperature"
            "--policy.marginal_quantile_sampling_min_level=$quantile_min"
            "--policy.marginal_quantile_sampling_max_level=$quantile_max"
            "--policy.marginal_quantile_sampling_ar_coefficient=$quantile_rho"
            "--policy.marginal_quantile_sampling_seed=$quantile_sampler_seed"
            "--policy.marginal_quantile_sampling_exclude_joints=$quantile_exclude_joints"
            "--quantile_head_quantile_decoder=$quantile_decoder"
            ${quantile_decoder_args[@]+"${quantile_decoder_args[@]}"}
            "--output_dir=$quantile_run_dir"
        )

        if [[ $quantile_dry_run == 1 ]]; then
            printf '%q ' "${quantile_cmd[@]}"
            printf '\n'
            continue
        fi

        mkdir -p "$quantile_output_root/seed_${quantile_sampler_seed}"
        # A fresh directory prevents overwriting a previous experiment's results/logs.
        if ! mkdir "$quantile_run_dir"; then
            printf 'Use a new OUTPUT_ROOT; run directory already exists or cannot be created: %s\n' "$quantile_run_dir" >&2
            exit 1
        fi
        printf '%q ' "${quantile_cmd[@]}" > "$quantile_run_dir/command.txt"
        printf '\n' >> "$quantile_run_dir/command.txt"
        printf 'Running %s, sampler seed %s; log: %s/eval.log\n' "$quantile_variant" "$quantile_sampler_seed" "$quantile_run_dir"
        if [[ $quantile_variant == M && $quantile_calibration_failed == 1 ]]; then
            printf 'M skipped because width calibration failed; see %s/eval.log.\n' "$quantile_calibration_dir" > "$quantile_run_dir/eval.log"
            quantile_exit_code=1
        elif (
            if [[ $quantile_viz == 1 ]]; then
                export LEROBOT_QUANTILE_HEAD_VIZ_DIR="$quantile_run_dir/action_diagnostics"
            else
                unset LEROBOT_QUANTILE_HEAD_VIZ_DIR
            fi
            "${quantile_cmd[@]}"
        ) > "$quantile_run_dir/eval.log" 2>&1; then
            quantile_exit_code=0
            if [[ ! -f "$quantile_run_dir/eval_info.json" ]]; then
                printf 'The command exited successfully but eval_info.json is missing.\n' >> "$quantile_run_dir/eval.log"
                quantile_exit_code=1
            fi
        else
            quantile_exit_code=$?
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$quantile_variant" "$quantile_sampler_seed" "$quantile_exit_code" \
            "$quantile_run_dir/eval_info.json" "$quantile_run_dir/eval.log" >> "$quantile_output_root/status.tsv"
        if [[ $quantile_exit_code == 0 ]]; then
            printf 'Finished %s: %s/eval_info.json\n' "$quantile_variant" "$quantile_run_dir"
        else
            quantile_any_failure=1
            printf 'FAILED %s (exit %s); see %s/eval.log. Continuing with the next setting.\n' \
                "$quantile_variant" "$quantile_exit_code" "$quantile_run_dir"
            printf 'Last 80 lines of the failed evaluation log:\n'
            tail -n 80 "$quantile_run_dir/eval.log" || true
        fi
    done
done

if [[ $quantile_dry_run != 1 ]]; then
    printf 'Completed the evaluation sequence. Status: %s/status.tsv\n' "$quantile_output_root"
fi
exit "$quantile_any_failure"
