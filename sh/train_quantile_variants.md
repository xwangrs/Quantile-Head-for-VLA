# QuantileHead quantile regression 训练消融

通过 `--policy.marginal_quantile_variant` 选择模型，默认 `standard` 保持原来的 QR。四种配置都使用 regression 和 quantile head；三个新模型需要从**同一个原始 base checkpoint 重新训练**，不能给已训练好的 QR checkpoint 换一个参数当作训练消融。

| 参数值 | 训练内容 | 用来回答的问题 |
| --- | --- | --- |
| `standard` | 原来的有序多分位数模型，中位数加减累积正间隔，监督全部分位数 | 完整 QR 对照 |
| `median_only` | 保留相同结构，只监督中间的 `Q(0.5)`，损失为 `0.5 × L1` | 额外分位数的训练信号是否有用 |
| `stop_median_gradient` | 非中心分位数构造时对中位数锚点 `detach()`；中心仍正常训练，间隔分支仍更新共享特征 | 非中心分位数直接推动中位数的梯度是否重要 |
| `independent` | 中心仍由原来的动作投影预测，非中心分位数独立输出，不强制单调 | 有序参数化及中位数耦合是否有贡献 |

`median_only` 的间隔分支没有得到监督；`independent` 允许分位数交叉。这两个模式只支持 `median` 推理。`stop_median_gradient` 仍有有序分位数，但本轮主要比较统一使用中位数动作。

## 直接运行脚本

在仓库根目录、已经安装训练依赖的 GPU 环境执行。脚本每次只训练一个模型，模型名是必填参数：

```bash
bash sh/train_quantile_variants.sh median_only
bash sh/train_quantile_variants.sh stop_median_gradient
bash sh/train_quantile_variants.sh independent
```

这些命令前台运行，依次执行即可。默认设置沿用当前 AutoDL recipe：

- base：`/root/autodl-tmp/hf-cache/checkpoints/pi05_libero_base`
- 数据：`/root/autodl-fs/libero`，`repo_id=libero`
- 双 GPU `0,1`，`steps=6000`、`batch_size=64`
- `soft_prompt_len=16`，`train_expert_only=true`，启用 compile 与 gradient checkpointing
- STATE/ACTION 的 `QUANTILES` 数据归一化，`action_loss_mask_ratio=0.1`
- 21 个分位数，训练分位范围 `[0.025, 0.975]`；这个范围与推理采样的 `[0.2, 0.8]` 或 `[0.3, 0.7]` 是不同设置
- 训练 seed 为 `1000`（当前训练配置的默认值），评测使用中位数

结果分别写入：

```text
/root/autodl-fs/outputs/libero_quantile_median_only_seed1000
/root/autodl-fs/outputs/libero_quantile_stop_median_gradient_seed1000
/root/autodl-fs/outputs/libero_quantile_independent_seed1000
```

脚本会自动初始化 conda 并激活 `lerobot06`，然后使用该环境中的 `accelerate` 启动训练。无需提前手动激活环境或执行 `unset` / `export`。环境准备顺序如下，并自动创建缓存和临时目录：

```bash
conda activate lerobot06
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf-cache/datasets
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache
export TORCH_HOME=/root/autodl-tmp/torch-cache
export TMPDIR=/root/autodl-tmp/tmp
```

脚本通过 `CONDA_EXE`、PATH 中的 `conda` 或常见安装目录定位 conda。如果安装在其他目录且无法自动发现，可指定 `QUANTILE_TRAIN_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh`。保留 `QUANTILE_TRAIN_RUNNER=uv` 作为显式使用项目 uv 环境的选项，通常无需设置。脚本的环境设置只影响本次训练及其子进程，不会修改当前终端的环境。

脚本拒绝已存在的输出目录。换训练 seed 或指定新目录即可建立另一次独立实验。普通训练参数使用 `--key=value` 覆盖；模型类型、head、推理模式和 fresh-run 设置由脚本固定，不能从尾部参数改变：

```bash
# 另一个训练 seed；对各模型使用同一组 seed。
QUANTILE_TRAIN_SEED=1001 bash sh/train_quantile_variants.sh median_only

# 改数据/base路径，并指定新的输出目录。
QUANTILE_TRAIN_DATASET_ROOT=/path/to/libero \
QUANTILE_TRAIN_PRETRAINED_PATH=/path/to/pi05_libero_base \
bash sh/train_quantile_variants.sh independent \
  --output_dir=/path/to/outputs/independent_run1

# 只预览环境准备和最终命令，不激活 conda、不创建目录、不启动训练。
QUANTILE_TRAIN_DRY_RUN=1 bash sh/train_quantile_variants.sh stop_median_gradient
bash sh/train_quantile_variants.sh --help
```

## 等价的完整训练命令

不使用脚本时，先在当前终端执行上面的 conda 和环境设置，再运行以下命令训练 `median_only`。换成 `stop_median_gradient` 或 `independent` 时，同时修改 `marginal_quantile_variant`、`output_dir`、`job_name`。若要重新训练完整 QR 对照，使用 `standard` 并换独立目录。

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes 2 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=libero \
  --dataset.video_backend=pyav \
  --dataset.root=/root/autodl-fs/libero \
  --policy.type=quantile_head \
  --output_dir=/root/autodl-fs/outputs/libero_quantile_median_only_seed1000 \
  --job_name=quantile_head_quantile_median_only_seed1000 \
  --resume=false \
  --policy.push_to_hub=false \
  --policy.pretrained_path=/root/autodl-tmp/hf-cache/checkpoints/pi05_libero_base \
  --policy.action_generation_mode=regression \
  --policy.use_pure_regression_head=false \
  --policy.marginal_quantile_variant=median_only \
  --policy.marginal_quantile_inference_mode=median \
  --policy.marginal_quantile_num_knots=21 \
  --policy.marginal_quantile_min_level=0.025 \
  --policy.marginal_quantile_initial_gap=0.02 \
  --policy.soft_prompt_len=16 \
  --policy.train_soft_prompt=true \
  --policy.action_loss_mask_ratio=0.1 \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only=true \
  --policy.detach_action_from_vlm=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=6000 \
  --batch_size=64 \
  --seed=1000
```

## 对照与评测

先核对原始 QR 和 L1 实验保存的训练配置。这里的 6000 steps、seed=1000 等是现有 recipe 的默认值；如果原实验不同，应将所有新实验对齐到原训练的预算、训练 seed、数据、prompt、归一化、随机 loss mask、batch size、GPU 数、精度、优化器和 checkpoint 选择规则。数据的 `QUANTILES` 归一化与 quantile regression 损失相互独立，不要为了做 L1 对照而改变归一化。`median_only` 使用真实的中位数 pinball loss，即 `0.5 × L1`；解释与已有 L1 的差异时需考虑该损失尺度。

相同训练 seed 下，三个变体保持公共模块的初始化随机数顺序；独立分位数头的初始化与有序头的初始预测对齐。这有助于控制初始化因素，但仍应对关键比较重复多个**训练 seed**。推理采样 seed 不能替代训练 seed。

统一使用中位数评测，示例沿用每任务 50 局的 LIBERO-10 规模。手动评测前，也需在当前终端执行上面的 conda 和环境设置：

```bash
lerobot-eval \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=50 \
  --policy.path=/root/autodl-fs/outputs/libero_quantile_median_only_seed1000/checkpoints/last/pretrained_model \
  --policy.marginal_quantile_inference_mode=median \
  --policy.n_action_steps=10 \
  --env.max_parallel_tasks=1 \
  --output_dir=/root/autodl-fs/tests/quantile_median_only_seed1000
```

其他模型只替换 checkpoint 和输出目录，不覆盖 checkpoint 保存的 `marginal_quantile_variant`。环境初始状态、episode 数和其余评测设置与原 A 对齐。`[0.2, 0.8]` / `[0.3, 0.7]` 采样范围对这轮中位数评测没有作用。

训练指标 `marginal_quantile_supervised_knots` 表示受监督的分位数数量（`median_only` 为 1，其他为全部 knots）；`marginal_quantile_crossing_fraction` 表示相邻分位数发生交叉的比例，重点观察 `independent`。交叉指标和任务成功率一起看：独立头退化且有明显交叉支持有序结构有作用，但不能单凭一次成功率差异认定因果。

这些额外指标由模型 `forward` 返回；普通终端训练日志不会逐项显示。脚本默认关闭 W&B，如需记录它们，在已配置 W&B 的训练环境中追加 `--wandb.enable=true`，或在 Python 中读取 `forward` 返回的指标字典。
