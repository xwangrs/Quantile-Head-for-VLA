# Quantile Head for Vision-Language-Action Models

**Xuan Wang · Yinan Wu · Haoran Duan · Jungong Han**<br>
Department of Automation, Tsinghua University

[arXiv:2609.34061](https://arxiv.org/abs/2609.34061) · [Method](#method) · [Results](#results) · [Installation](#installation) · [Training](#training) · [Evaluation](#evaluation)

Quantile Head predicts **ordered marginal action quantiles in one forward pass**.
Joint quantile supervision trains a strong median policy, while the same head
supports quantile sampling at inference without retraining. Built on the π₀.₅
backbone, it combines a frozen vision-language model with learnable prompts and a
trainable action expert.

## Method

![Quantile Head architecture: a frozen VLM, learnable prompts, and an action expert predict ordered action quantiles trained with masked pinball loss.](docs/assets/quantile-head-architecture.png)

*Architecture from Figure 2 of the paper.*

- **Predict a median and positive gaps.** The head accumulates independently
  learned left and right gaps around the median to produce 21 noncrossing
  quantiles for each action coordinate.
- **Train all quantiles together.** Masked pinball loss supervises the quantiles
  with a shared coordinate mask, randomly withholding 10% of valid action labels.
  Sixteen learnable prompt tokens adapt the frozen VLM features for control.
- **Decode in one pass.** The default policy executes the median. Optional
  sampling interpolates the predicted marginal quantiles and can correlate
  quantile ranks across replanning steps.

<details>
<summary>Attention and loss masks (Figure 3)</summary>

![Block self-attention among VLM, prompt, and action tokens; coordinate-label masks shared across all predicted quantiles.](docs/assets/quantile-head-masks.png)

Prompts attend to VLM and prompt features; action tokens attend to all three
groups. During training, each example receives a newly sampled coordinate-label
mask shared across all quantiles. The head models one marginal distribution per
action coordinate; sampling specifies how ranks are coupled across coordinates
and time.

</details>

## Results

The tables below compare Quantile Head with other methods on **LIBERO,
LIBERO-Pro, LIBERO-Plus, and two real-robot tasks** (Tables 1–3 of the paper).
Success rates are percentages (higher is better). Quantile Head uses
**deterministic median decoding**. External baselines retain their original
training and evaluation protocols.

**Bold** and <ins>underlined</ins> values in Tables 1–2 mark the best and
second-best results. Quantile Head benchmark rates average three training seeds.
LIBERO uses 1,000 episodes per suite and trained model; LIBERO-Plus covers all
10,030 benchmark instances per model; LIBERO-Pro uses 100 episodes per task and
model.

### Table 1. LIBERO and LIBERO-Pro

**(a) LIBERO**

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy<sup>a</sup> | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA<sup>a</sup> | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀-FAST<sup>a</sup> | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀<sup>a</sup> | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
| VLANeXt | <ins>99.0</ins> | 99.2 | 96.6 | 94.8 | 97.4 |
| SimVLA | **99.6** | <ins>99.8</ins> | <ins>98.6</ins> | 96.4 | 98.6 |
| WLA-0 | <ins>99.0</ins> | **100.0** | 97.8 | 97.6 | 98.6 |
| InternVLA-A1.5 | 98.6 | <ins>99.8</ins> | <ins>98.6</ins> | <ins>98.4</ins> | <ins>98.9</ins> |
| Quantile Head | <ins>99.0</ins> | **100.0** | **99.6** | **98.6** | **99.3** |

Long denotes LIBERO-10. Avg. is the four-suite mean; source-reported precision is retained. <sup>a</sup>Results are from Kim et al., *Fine-tuning Vision-Language-Action Models: Optimizing Speed and Success* (OpenVLA-OFT), Table I. The OpenVLA-OFT row is also from that paper. The π₀.₅ reference is the official openpi 30k-step checkpoint reported in its LIBERO README. Other baselines follow their original papers.

**(b) LIBERO-Pro**

| Method | Object | Position | Semantic | Task | Avg. |
|---|---:|---:|---:|---:|---:|
| OpenVLA | <ins>93.0</ins> | 0.0 | 97.3 | 0.0 | 47.6 |
| π₀ | 90.5 | 0.0 | 90.5 | 0.0 | 45.3 |
| π₀.₅ | **96.0** | <ins>20.8</ins> | 95.8 | 0.8 | <ins>53.3</ins> |
| MolmoAct | 76.0 | 1.5 | 85.8 | 1.5 | 41.2 |
| X-VLA | 79.0 | 0.8 | 96.8 | 6.8 | 45.8 |
| OpenVLA-OFT<sup>b</sup> | 27.6 | 5.8 | 59.0 | 0.6 | 23.3 |
| X-VLA (CR eval.)<sup>b</sup> | 78.4 | 1.6 | 82.9 | 16.4 | 44.8 |
| VLA-Adapter<sup>b</sup> | 73.8 | 0.0 | 90.8 | <ins>19.8</ins> | 46.1 |
| SimVLA | 81.5 | 8.3 | **98.8** | 6.0 | 48.6 |
| SUREFlow | 68.5 | 0.0 | 87.8 | 0.0 | 39.1 |
| **Quantile Head** | 85.8 | **31.3** | <ins>97.8</ins> | **26.0** | **60.2** |

Each perturbation category averages Spatial, Object, Goal, and Long; Avg. averages the four categories, excluding Original and Environment. Quantile Head uses its LIBERO-trained checkpoint without Pro fine-tuning. The first five baselines use the official LIBERO-Pro leaderboard. <sup>b</sup>Results are from Wu et al., *Continuous Reasoning for Vision-Language-Action*. SimVLA and SUREFlow use their original reports.

### Table 2. LIBERO-Plus

**(a) Zero-shot transfer** — LIBERO-trained policies without further training.

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA<sup>a</sup> | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| π₀-FAST<sup>a</sup> | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| π₀<sup>a</sup> | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| OpenVLA-OFT<sup>a</sup> | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀.₅<sup>b</sup> | 75.8 | 79.4 | 83.3 | 95.5 | <ins>95.0</ins> | 89.6 | 87.0 | <ins>85.7</ins> |
| VLANeXt | <ins>90.4</ins> | 65.7 | 81.8 | 95.9 | 82.5 | 94.1 | 80.8 | 83.9 |
| GAM | 83.1 | 70.0 | <ins>84.8</ins> | **97.2** | 94.3 | <ins>95.3</ins> | 79.1 | 85.5 |
| GE-Act 2.0 | **94.1** | 50.7 | 81.4 | 94.0 | 60.5 | **95.5** | 83.1 | 80.4 |
| ACoT-VLA<sup>b</sup> | 68.9 | <ins>80.3</ins> | 84.1 | 95.6 | 93.1 | 81.5 | **88.3** | 83.6 |
| Quantile Head | 71.6 | **80.7** | **87.6** | <ins>96.1</ins> | **97.2** | 94.8 | <ins>87.8</ins> | **87.1** |

**(b) Supervised fine-tuning** — uses LIBERO-Plus training data.

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| π₀<sup>b</sup> | 79.6 | 21.1 | 72.5 | 84.7 | 86.2 | 68.3 | 69.4 | 67.4 |
| π₀.₅<sup>b</sup> | 70.3 | 41.7 | 81.1 | <ins>97.3</ins> | 94.6 | 71.8 | <ins>84.9</ins> | 75.7 |
| OpenVLA-OFT+ | <ins>92.8</ins> | 30.3 | <ins>85.8</ins> | 94.9 | 93.9 | 89.3 | 77.6 | 79.6 |
| MemoryVLA<sup>d</sup> | 91.4 | 48.6 | 79.4 | 95.2 | 95.3 | 94.0 | 75.7 | 81.9 |
| MemoryVLA++ | **96.8** | 49.7 | 71.0 | 96.6 | <ins>97.0</ins> | **96.0** | 78.6 | 82.7 |
| ACoT-VLA<sup>b</sup> | 91.2 | <ins>62.5</ins> | 80.3 | 95.1 | 91.5 | 88.3 | <ins>84.9</ins> | <ins>84.1</ins> |
| Quantile Head (SFT) | 88.6 | **77.0** | **86.2** | **97.4** | **97.3** | <ins>94.1</ins> | **87.3** | **89.1** |

Source superscripts are local to Table 2: <sup>a</sup>Fei et al., *LIBERO-Plus*; <sup>b</sup>Zhong et al., *ACoT-VLA*; <sup>d</sup>Shi et al., *MemoryVLA++*. Unmarked baselines use their original papers. The π₀ and π₀.₅ SFT results from ACoT-VLA freeze the LLM backbone. Baseline totals follow each source's aggregation. Quantile Head pools all 10,030 benchmark instances for Total and reports the mean over three training seeds; Total is instance-weighted, not an unweighted mean of categories or suites.

### Table 3. Real-robot evaluation

| Real-robot task | π₀.₅ | Quantile Head | Gain (percentage points) |
|---|---:|---:|---:|
| Place apple on yellow plate | 31 | 52 | +21 |
| Remove cuboid from blue plate | 86 | 98 | +12 |
| Average | 58.5 | 75.0 | +16.5 |

The π₀.₅ baseline is fully fine-tuned. Both methods use the same demonstration data, optimization settings, and evaluation conditions.

## Installation

Set up the environment by following the **[official LeRobot installation guide](https://huggingface.co/docs/lerobot/installation)**,
including Python, PyTorch/CUDA, and video dependencies. At the source-install
step, install this repository in the activated environment:

```bash
git clone https://github.com/xwangrs/Quantile-Head-for-VLA.git
cd Quantile-Head-for-VLA
pip install -e ".[training,evaluation,pi,libero]"
```

For simulation assets and benchmark dependencies, follow the official
[LIBERO](https://huggingface.co/docs/lerobot/libero),
[LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO), and
[LIBERO-Plus](https://huggingface.co/docs/lerobot/libero_plus) instructions.
The benchmarks share the Python package name `libero`; use a separate environment
for each benchmark and install its matching package and assets. In the Pro and
Plus environments, replace `hf-libero` with the corresponding fork and add that
fork's repository root to `PYTHONPATH` as described in its installation guide.
Set `LIBERO_CONFIG_PATH` to a separate directory containing that benchmark's
`config.yaml`, with the correct `assets`, `bddl_files`, `init_states`, and
`datasets` paths.

## Training

Both runs start from the same compatible π₀.₅ base weights and train for one epoch
with a global batch size of 128 (64 per GPU on two GPUs). The VLM is frozen;
the action expert, quantile head, and 16 prompt tokens are trained. Replace the
dataset IDs and local paths below with your prepared LeRobot-format assets.
Each dataset must contain `meta/info.json`, its videos, and `q01`/`q99` state and
action statistics.

Run either block from the repository root in the activated LeRobot environment.
Use the same base-weight directory for both runs.

### LIBERO training

Use 200 warmup steps.

```bash
export QH_BASE_WEIGHTS=/path/to/pi05_base
export QH_TRAIN_SEED=1000
export QH_LIBERO_DATASET_ID=your_namespace/libero
export QH_LIBERO_DATASET_ROOT=/path/to/lerobot_libero
QH_LIBERO_STEPS=$(python -c \
  'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["total_frames"] / 128))' \
  "$QH_LIBERO_DATASET_ROOT/meta/info.json")

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes=2 --mixed_precision=no \
  -m lerobot.scripts.lerobot_train \
  --policy.type=quantile_head \
  --policy.pretrained_path="$QH_BASE_WEIGHTS" \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --wandb.enable=false \
  --dataset.repo_id="$QH_LIBERO_DATASET_ID" \
  --dataset.root="$QH_LIBERO_DATASET_ROOT" \
  --policy.scheduler_warmup_steps=200 \
  --policy.scheduler_decay_steps="$QH_LIBERO_STEPS" \
  --steps="$QH_LIBERO_STEPS" \
  --seed="$QH_TRAIN_SEED" \
  --job_name=quantile_head_libero \
  --output_dir="outputs/quantile_head_libero_seed${QH_TRAIN_SEED}"
```

### LIBERO-Plus training

Use 666 warmup steps.

```bash
export QH_BASE_WEIGHTS=/path/to/pi05_base
export QH_TRAIN_SEED=1000
export QH_PLUS_DATASET_ID=your_namespace/libero_plus
export QH_PLUS_DATASET_ROOT=/path/to/lerobot_libero_plus
QH_PLUS_STEPS=$(python -c \
  'import json, math, sys; print(math.ceil(json.load(open(sys.argv[1]))["total_frames"] / 128))' \
  "$QH_PLUS_DATASET_ROOT/meta/info.json")

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes=2 --mixed_precision=no \
  -m lerobot.scripts.lerobot_train \
  --policy.type=quantile_head \
  --policy.pretrained_path="$QH_BASE_WEIGHTS" \
  --policy.push_to_hub=false \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --dataset.video_backend=pyav \
  --batch_size=64 \
  --wandb.enable=false \
  --dataset.repo_id="$QH_PLUS_DATASET_ID" \
  --dataset.root="$QH_PLUS_DATASET_ROOT" \
  --policy.scheduler_warmup_steps=666 \
  --policy.scheduler_decay_steps="$QH_PLUS_STEPS" \
  --steps="$QH_PLUS_STEPS" \
  --seed="$QH_TRAIN_SEED" \
  --job_name=quantile_head_libero_plus \
  --output_dir="outputs/quantile_head_libero_plus_seed${QH_TRAIN_SEED}"
```

The current `quantile_head` defaults provide the standard ordered 21-quantile
head, 16 trainable prompts, 10% label masking, median decoding, a 50-action
prediction horizon, and `QUANTILES` state/action normalization. They also provide
the paper's AdamW settings: learning rate `2.5e-5` and weight decay `0.01` for
ordinary trainable parameters, and `1e-4` with zero weight decay for prompts.
The commands use the default `float32` policy and explicitly disable launcher
mixed precision.

The epoch budget assumes the complete, non-streaming dataset; the last distributed
batch can repeat samples to fill the batch. If you filter episodes or change the
sampler, recompute the budget for the selected samples. Set a new training seed
and use a fresh output directory for each independent replicate. The last saved
model is under `OUTPUT_DIR/checkpoints/last/pretrained_model`.

For complete asset and checkpoint requirements, see the
[training and evaluation guide](docs/reproduction.md).

## Evaluation

Run each block in the corresponding benchmark environment prepared above. Use a
saved `quantile_head` checkpoint, including its config, processors, and statistics.
All three evaluations use median decoding and execute ten actions before replanning.

### LIBERO evaluation

Evaluate all four suites with 100 episodes per task (1,000 per suite):

```bash
QH_CHECKPOINT=outputs/quantile_head_libero_seed1000/checkpoints/last/pretrained_model

MUJOCO_GL=egl lerobot-eval \
  --policy.path="$QH_CHECKPOINT" \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.marginal_quantile_inference_mode=median \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.init_states=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=100 \
  --seed=1000 \
  --output_dir=outputs/eval/libero_seed1000
```

### LIBERO-Pro evaluation

With the official LIBERO-Pro fork and its BDDL/init-state files installed, use
`--env.type=libero` and the perturbation suite names below. The suffixes `object`,
`swap`, `lan`, and `task` select object, position, semantic, and task perturbations.
The loop sets each base suite's episode limit explicitly.

```bash
QH_CHECKPOINT=outputs/quantile_head_libero_seed1000/checkpoints/last/pretrained_model
QH_EVAL_OUTPUT=outputs/eval/libero_pro_seed1000

(
  set -e
  for qh_spec in libero_spatial:280 libero_object:280 libero_goal:300 libero_10:800; do
    qh_suite=${qh_spec%:*}
    qh_max_steps=${qh_spec#*:}
    MUJOCO_GL=egl lerobot-eval \
      --policy.path="$QH_CHECKPOINT" \
      --policy.device=cuda \
      --policy.n_action_steps=10 \
      --policy.marginal_quantile_inference_mode=median \
      --env.type=libero \
      --env.task="${qh_suite}_object,${qh_suite}_swap,${qh_suite}_lan,${qh_suite}_task" \
      --env.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
      --env.episode_length="$qh_max_steps" \
      --env.init_states=true \
      --env.max_parallel_tasks=1 \
      --eval.batch_size=1 \
      --eval.n_episodes=100 \
      --seed=1000 \
      --output_dir="$QH_EVAL_OUTPUT/$qh_suite"
  done
)
```

This covers the four perturbation categories in Table 1(b), with 100 episodes per
task. Average the four base-suite rates within each category, then average the
four category rates.

### LIBERO-Plus evaluation

With the official LIBERO-Plus fork and assets installed, evaluate every registered
instance once. The four suites contain 2,402, 2,518, 2,591, and 2,519 instances,
respectively, for 10,030 in total. LIBERO-Plus Long uses a 520-step episode limit.

```bash
# Use the LIBERO-trained checkpoint for zero-shot evaluation.
QH_CHECKPOINT=outputs/quantile_head_libero_seed1000/checkpoints/last/pretrained_model
QH_EVAL_OUTPUT=outputs/eval/libero_plus_zero_shot_seed1000

(
  set -e
  for qh_spec in libero_spatial:280 libero_object:280 libero_goal:300 libero_10:520; do
    qh_suite=${qh_spec%:*}
    qh_max_steps=${qh_spec#*:}
    MUJOCO_GL=egl lerobot-eval \
      --policy.path="$QH_CHECKPOINT" \
      --policy.device=cuda \
      --policy.n_action_steps=10 \
      --policy.marginal_quantile_inference_mode=median \
      --env.type=libero_plus \
      --env.task="$qh_suite" \
      --env.episode_length="$qh_max_steps" \
      --env.init_states=true \
      --env.max_parallel_tasks=1 \
      --eval.batch_size=1 \
      --eval.n_episodes=1 \
      --seed=1000 \
      --output_dir="$QH_EVAL_OUTPUT/$qh_suite"
  done
)
```

For supervised fine-tuning evaluation, set `QH_CHECKPOINT` to
`outputs/quantile_head_libero_plus_seed1000/checkpoints/last/pretrained_model`
and use a separate output directory such as `outputs/eval/libero_plus_sft_seed1000`.
Compute Total by pooling successful episodes across all 10,030 instances. The
evaluator saves per-instance results in `eval_info.json`; the seven perturbation
category totals require grouping these results by the benchmark's categories.

The paper's benchmark results average three independently trained checkpoints.
Repeat the evaluations for each checkpoint with a separate output directory, then
average their corresponding success rates.

## Acknowledgments and license

This repository builds on [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
and its implementation of [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi).
We thank these projects and the LIBERO, LIBERO-Plus, and LIBERO-Pro benchmark
authors. The code retains the upstream [Apache License 2.0](LICENSE) and source
copyright notices. Weights, datasets, and benchmark assets retain their respective
licenses.
