# Reproducing Quantile Head

[Back to the project README](../README.md)

Run all commands from the repository root. This guide covers installation,
training, simulation evaluation, and the supported inference configurations.

## Installation and required assets

Follow the [official LeRobot installation guide](https://huggingface.co/docs/lerobot/installation)
and the [repository installation instructions](../README.md#installation).
Run the commands below in the activated environment with this repository installed.

Prepare these assets separately:

1. A LeRobot-format training dataset, including its videos, metadata, task text,
   and normalization statistics. Supply its actual repository ID and local root.
2. A compatible π₀.₅ base weight directory for initialization, or a trained
   `quantile_head` checkpoint for evaluation. A LeRobot pretrained directory uses
   `model.safetensors`; evaluation also needs its saved config and processors.
3. The tokenizer used by the processors: `google/paligemma-3b-pt-224`. It must be
   accessible through Hugging Face or already available in the local cache.
4. LIBERO task definitions, assets, and initial states for simulation evaluation.

Datasets, benchmark assets, and model weights are not bundled in this source
checkout. Offline runs require all model, tokenizer, dataset, and environment
assets to be cached beforehand.

State and action `QUANTILES` normalization requires `q01` and `q99` statistics.
This data normalization is separate from the quantile-regression objective. If an
older dataset lacks these statistics, the following utility updates its metadata:

```bash
python -m lerobot.scripts.augment_dataset_quantile_stats \
  --repo-id=your_dataset_id \
  --root=/path/to/lerobot_dataset
```

## Training

The manuscript freezes the VLM and trains the action expert, 16 prompt tokens, and
quantile head. It uses AdamW, a global batch size of 128 on two GPUs, and one epoch
on each training dataset. The prompt learning rate is `1e-4` with zero weight
decay; other trainable parameters use `2.5e-5` and weight decay `0.01`. Warmup lasts
200 steps for LIBERO and 666 for LIBERO-Plus, followed by cosine decay.

The training CLI uses a **step budget**, not an epoch argument. For the unfiltered,
non-streaming dataset below, compute one epoch from `total_frames` and the global
batch size. The final distributed batch may repeat examples to fill the batch.
If you select episodes or change the sampler, recalculate the budget from the
number of selected samples.

```bash
export QH_DATASET_ID=libero
export QH_DATASET_ROOT=/path/to/lerobot_libero
export QH_BASE_WEIGHTS=/path/to/pi05_base
export QH_TRAIN_SEED=1000
export QH_OUTPUT_DIR=outputs/quantile_head_libero_seed1000

QH_TRAIN_STEPS=$(python -c \
  'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["total_frames"] / 128))' \
  "$QH_DATASET_ROOT/meta/info.json")

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes=2 --mixed_precision=no -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$QH_DATASET_ID" \
  --dataset.root="$QH_DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --policy.type=quantile_head \
  --policy.pretrained_path="$QH_BASE_WEIGHTS" \
  --policy.push_to_hub=false \
  --policy.action_generation_mode=regression \
  --policy.use_pure_regression_head=false \
  --policy.marginal_quantile_variant=standard \
  --policy.marginal_quantile_num_knots=21 \
  --policy.marginal_quantile_min_level=0.025 \
  --policy.marginal_quantile_initial_gap=0.02 \
  --policy.action_loss_mask_ratio=0.1 \
  --policy.soft_prompt_len=16 \
  --policy.train_soft_prompt=true \
  --policy.train_expert_only=true \
  --policy.detach_action_from_vlm=false \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.marginal_quantile_inference_mode=median \
  --policy.optimizer_lr=2.5e-5 \
  --policy.optimizer_weight_decay=0.01 \
  --policy.optimizer_soft_prompt_lr=1e-4 \
  --policy.optimizer_soft_prompt_weight_decay=0.0 \
  --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps="$QH_TRAIN_STEPS" \
  --policy.scheduler_decay_lr=2.5e-6 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.device=cuda \
  --batch_size=64 \
  --steps="$QH_TRAIN_STEPS" \
  --seed="$QH_TRAIN_SEED" \
  --resume=false \
  --wandb.enable=false \
  --job_name=quantile_head_libero \
  --output_dir="$QH_OUTPUT_DIR"
```

Replace all asset paths and the dataset ID before running. `batch_size=64` is per
GPU, giving a global batch size of 128. The command leaves precision at the
implementation default (`float32`); `--policy.dtype=bfloat16` is supported on
compatible hardware. Record precision and runtime choices with each experiment.
Use a fresh output directory for each independent training seed.

For LIBERO-Plus training, use its dataset, recompute the one-epoch step budget, and
set `--policy.scheduler_warmup_steps=666`. The manuscript initializes the LIBERO
and LIBERO-Plus models from the same base checkpoint. The scripts in `sh/` retain
experiment-specific defaults such as 6,000 steps and local cluster paths; those
values do not automatically implement the manuscript's one-epoch budget.

The last saved policy is written under
`OUTPUT_DIR/checkpoints/last/pretrained_model`. Keep the complete checkpoint
folder, including processor configurations and statistics.

## LIBERO evaluation

Set `QH_CHECKPOINT` to a trained `quantile_head` pretrained-model directory. This
small run checks the four standard suites with 10 episodes per task:

```bash
export QH_CHECKPOINT=/path/to/quantile_head/checkpoints/last/pretrained_model
export MUJOCO_GL=egl

lerobot-eval \
  --policy.path="$QH_CHECKPOINT" \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.marginal_quantile_inference_mode=median \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=1000 \
  --output_dir=outputs/eval/quantile_head_median
```

For the main manuscript protocol, set `--eval.n_episodes=100` and evaluate each
of the three independently trained checkpoints. The evaluator applies this count
to each task: 100 episodes per task across ten tasks gives 1,000 episodes per suite
and trained model. Average task success rates within
each suite, then average the four suite rates, and finally average corresponding
rates across the three trained models. Changing only a sampler seed does not
create an independent training replicate. Evaluation writes `eval_info.json` and
rollout outputs under the chosen output directory.

The controller executes 10 actions before replanning; `chunk_size=50` is the
prediction horizon. Dataset features, action representation, camera names, and
normalization must agree with the checkpoint and environment. The inherited
[LIBERO documentation](../docs/source/libero.mdx) describes the environment; its
upstream model scores and example episode counts are not the Quantile Head paper
protocol.

LIBERO-Plus needs its separate fork and assets; see the
[environment guide](../docs/source/libero_plus.mdx) and
[Docker setup](../docker/Dockerfile.benchmark.libero_plus). A default LIBERO-Plus run
is not sufficient to reproduce all 10,030 paper instances. LIBERO-Pro likewise
requires its benchmark installation and perturbation protocol. Use the
[three benchmark commands](../README.md#evaluation) for the complete suite selections
and episode limits.

## Inference and ablations

`marginal_quantile_inference_mode=median` uses the central prediction. To sample
from a trained ordered head, replace that flag in the evaluation command with:

```bash
--policy.marginal_quantile_inference_mode=temporal_sample \
--policy.marginal_quantile_sampling_temperature=0.35 \
--policy.marginal_quantile_sampling_min_level=0.4 \
--policy.marginal_quantile_sampling_max_level=0.6 \
--policy.marginal_quantile_sampling_ar_coefficient=0.7 \
--policy.marginal_quantile_sampling_seed=0 \
--policy.marginal_quantile_sampling_exclude_joints='["gripper"]'
```

Each motion dimension receives its own quantile rank, shared across the predicted
chunk. The AR coefficient controls correlation between replanning steps: `0`
resamples ranks independently, while `1` holds each rank for the episode. The
example explicitly sets a narrow rank window and keeps the gripper at its median.
The class defaults are `[0.1, 0.9]`, temperature `0.35`, AR coefficient `0.9`, and
sampler seed `1000`; they are not a universal paper sampling configuration.

| Parameter | Supported values / purpose |
| --- | --- |
| `marginal_quantile_variant` | `standard`, `median_only`, `stop_median_gradient`, `independent` |
| `use_pure_regression_head` / `pure_regression_loss` | Point-regression baselines with `l1` or `l2` loss |
| `action_generation_mode` | `regression` or the retained `flow_matching` baseline |
| `marginal_quantile_inference_mode` | `median` or `temporal_sample` |
| `marginal_quantile_sampling_*` | Temperature, rank window, AR coefficient, seed, and excluded joints |
| `action_loss_mask_ratio` | Fraction of valid coordinate labels dropped during training |
| `soft_prompt_len` / `train_expert_only` | Prompt length and whether to freeze the VLM |

`median_only` and `independent` support median inference only. A head variant is a
training/architecture choice and must not be changed merely to evaluate a saved
checkpoint. The current implementation does not expose every manuscript ablation
as a configuration option; for example, there is no symmetric-gap variant.

Additional evaluation-only decoders are selected with the **top-level** option
`--quantile_head_quantile_decoder`: `policy`, `independent_rank`,
`gaussian_temporal`, `quantile_expectation`, `fixed_width`, and `calibrate_width`.
The Gaussian and fixed-width decoders reuse predictions from the quantile model;
they are not independently trained models. Fixed-width decoding requires a
separate calibration file. The [policy guide](source/quantile_head.mdx) documents
the decoder options. Use the direct [benchmark evaluation commands](../README.md#evaluation)
for the paper's default median policy.


## Policy name and checkpoint format

Use `--policy.type=quantile_head` for new training runs and the new Python imports:

```python
from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig
from lerobot.policies.quantile_head.modeling_quantile_head import QuantileHeadPolicy
```

This is a breaking rename of the modified policy previously stored under `pi05`.
Old `pi05` policy types, Python import paths, processor names, and legacy
compatibility-only configuration fields are not supported. Use checkpoints saved
with the current `quantile_head` config and processors for evaluation. Initializing
new training from compatible π₀.₅ backbone weights is a separate supported use
case; it does not make old experiment checkpoints interchangeable with this format.

Update evaluation flags from `pi05_quantile_*` to `quantile_head_quantile_*` and
visualization environment variables from `LEROBOT_PI05_VIZ_*` to
`LEROBOT_QUANTILE_HEAD_VIZ_*`. The current configuration in
[`configuration_quantile_head.py`](../src/lerobot/policies/quantile_head/configuration_quantile_head.py)
is the source of truth for supported fields and validation.
