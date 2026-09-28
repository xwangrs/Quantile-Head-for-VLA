# 实验脚本

所有 `.sh` 实验脚本集中在这里。以下命令都在仓库根目录执行。

| 脚本 | 用途 | 说明 |
| --- | --- | --- |
| [train_quantile_variants.sh](train_quantile_variants.sh) | 训练三个 QR 消融模型或完整 QR 对照，每次一个模型 | [训练说明](train_quantile_variants.md) |
| [eval_quantile_dm.sh](eval_quantile_dm.sh) | 默认评测 D–M 扩展配置，跳过 A/B/C 和 C_all，包含 `[0.3, 0.7]` 范围实验 | [评测说明](eval_quantile_dm.md) |
| [eval_quantile_ablations.sh](eval_quantile_ablations.sh) | 通用评测入口，默认 A/B/C，可用 `QUANTILE_EVAL_VARIANTS` 选择配置 | [配置说明](eval_quantile_dm.md) |

## 训练

训练脚本自动激活 `lerobot06`，清理离线和代理变量，设置 HF 镜像及 AutoDL 缓存路径。

```bash
bash sh/train_quantile_variants.sh median_only
bash sh/train_quantile_variants.sh stop_median_gradient
bash sh/train_quantile_variants.sh independent
```

## 评测

评测脚本使用当前终端环境；运行前需激活 `lerobot06` 并准备环境，详见[环境设置](train_quantile_variants.md)。

```bash
# 评测 A/B/C 之外的扩展配置，使用默认 checkpoint。
bash sh/eval_quantile_dm.sh

# 或指定 checkpoint、输出目录、每任务局数和采样 seed。
bash sh/eval_quantile_dm.sh /path/to/pretrained_model outputs/quantile_dm 50 0
```

## 预览

以下命令只打印设置，不启动训练或评测，也不创建输出目录：

```bash
QUANTILE_TRAIN_DRY_RUN=1 bash sh/train_quantile_variants.sh median_only
QUANTILE_DRY_RUN=1 bash sh/eval_quantile_dm.sh
```
