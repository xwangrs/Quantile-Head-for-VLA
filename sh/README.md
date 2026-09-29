# 实验脚本

所有 `.sh` 实验脚本集中在这里。以下命令都在仓库根目录执行。

| 脚本 | 用途 | 说明 |
| --- | --- | --- |
| [train_quantile_variants.sh](train_quantile_variants.sh) | 训练三个 QR 消融模型或完整 QR 对照，每次一个模型 | [训练说明](train_quantile_variants.md) |

## 训练

训练脚本自动激活 `lerobot06`，清理离线和代理变量，设置 HF 镜像及 AutoDL 缓存路径。

```bash
bash sh/train_quantile_variants.sh median_only
bash sh/train_quantile_variants.sh stop_median_gradient
bash sh/train_quantile_variants.sh independent
```

## 评测

使用 `lerobot-eval` 直接评测训练后的 checkpoint，命令见仓库首页的[评测说明](../README.md#evaluation)。

## 预览

以下命令只打印训练设置，不启动训练，也不创建输出目录：

```bash
QUANTILE_TRAIN_DRY_RUN=1 bash sh/train_quantile_variants.sh median_only
```
