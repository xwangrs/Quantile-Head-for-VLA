# Quantile Head (`quantile_head`)

This is the paper's policy implementation. It extends the PI0.5 backbone from [OpenPI](https://github.com/Physical-Intelligence/openpi) with a marginal quantile action head and action-side soft prompts.

The policy learns ordered quantile knots with a pinball loss. It supports deterministic median actions and temporal quantile sampling. The point-regression baseline and quantile supervision ablations are configured separately for fresh training runs.

| Interface | Name |
| --- | --- |
| Policy type | `quantile_head` |
| Configuration | `QuantileHeadConfig` |
| Policy | `QuantileHeadPolicy` |
| Core model | `QuantileHeadPytorch` |
| Package | `lerobot.policies.quantile_head` |

See the [policy guide](./quantile_head.mdx) for parameters, training, evaluation, and relative actions, and the [repository README](../../README.md) for the paper workflow.

## Training

```bash
lerobot-train \
  --policy.type=quantile_head \
  --policy.pretrained_path=lerobot/pi05_base \
  --dataset.repo_id=your_org/your_dataset \
  --policy.push_to_hub=false \
  --output_dir=outputs/quantile_head
```

The upstream model ID `lerobot/pi05_base` supplies the backbone weights. New training initializes the quantile head and action prompt when absent from the base model. Evaluation and resume use complete checkpoints saved with `type="quantile_head"`. Old policy names and removed configuration fields are not supported.

For relative actions, add `--policy.use_relative_actions=true` and provide statistics computed in the relative-action space. The default `relative_exclude_joints=["gripper"]` keeps matching coordinates absolute. See the policy guide for the preprocessing order.

## Upstream attribution

The following references describe the upstream backbone and its implementation, not the paper's Quantile Head contribution:

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
  title        = {π₀.₅: a Vision-Language-Action Model with Open-World Generalization},
  author       = {Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
  year         = {2025},
  eprint       = {2504.16054},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2504.16054},
}
```

## License

The implementation retains the Apache 2.0 license and the original OpenPI and LeRobot notices.
