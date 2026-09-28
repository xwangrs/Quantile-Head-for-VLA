#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Requires: pip install 'lerobot[evaluation]' plus the policy extra (e.g. lerobot[pi])
          and the environment extra (e.g. lerobot[pusht]) if evaluating in simulation.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import csv
import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, NotRequired, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.quantile_head_eval_decoders import (
    configure_quantile_head_eval_decoder,
    describe_quantile_head_quantile_sampler,
    export_quantile_head_quantile_width_calibration,
)
from lerobot.types import PolicyAction
from lerobot.utils.constants import (
    ACTION,
    DONE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STR,
    REWARD,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)

QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV = "LEROBOT_QUANTILE_HEAD_VIZ_DIR"
QUANTILE_HEAD_ACTION_DISTRIBUTION_SAMPLES_ENV = "LEROBOT_QUANTILE_HEAD_VIZ_SAMPLES"
QUANTILE_HEAD_ACTION_DISTRIBUTION_HORIZON_ENV = "LEROBOT_QUANTILE_HEAD_VIZ_HORIZON"


class QuantileHeadActionDistribution(TypedDict):
    marginal_samples_normalized: Tensor
    marginal_samples_env: Tensor
    median_actions_normalized: Tensor
    median_actions_env: Tensor
    quantile_knots_normalized: Tensor
    quantile_knots_env: Tensor
    quantile_levels: Tensor
    action_names: list[str]


class ExecutedActionStatistics(TypedDict):
    actions_env: np.ndarray
    delta_actions_env: np.ndarray
    episode_lengths: np.ndarray
    action_correlation: np.ndarray
    delta_correlation: np.ndarray
    action_names: list[str]


def _quantile_head_action_names(config: Any, action_dim: int) -> list[str]:
    action_names = config.action_feature_names
    if action_names and len(action_names) == action_dim:
        return list(action_names)
    if action_dim == 7:
        return ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"]
    return [f"action_{index}" for index in range(action_dim)]


def _quantile_head_task_output_dir(output_dir: Path, task_name: str) -> Path:
    task_slug = "_".join(task_name.split())
    task_slug = "".join(
        character if character.isascii() and (character.isalnum() or character in "-_") else "_"
        for character in task_slug
    ).strip("_")
    task_digest = hashlib.sha256(task_name.encode()).hexdigest()[:8]
    task_dir = output_dir / f"{task_slug[:64] or 'task'}_{task_digest}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _pearson_correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Compute a Pearson matrix without warnings for constant action dimensions."""
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {values.shape}")

    action_dim = values.shape[1]
    correlation = np.full((action_dim, action_dim), np.nan, dtype=np.float64)
    finite_values = values[np.isfinite(values).all(axis=1)]
    if finite_values.shape[0] < 2:
        return correlation

    centered = finite_values - finite_values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    varying = norms > np.finfo(np.float64).eps
    if not varying.any():
        return correlation

    normalized = centered[:, varying] / norms[varying]
    varying_correlation = np.clip(normalized.T @ normalized, -1.0, 1.0)
    correlation[np.ix_(varying, varying)] = varying_correlation
    return correlation


def compute_executed_action_statistics(
    episode_actions: Sequence[np.ndarray],
    action_names: Sequence[str],
) -> ExecutedActionStatistics:
    """Compute action and within-episode action-delta correlations on environment-scale commands."""
    if not episode_actions:
        raise ValueError("At least one episode is required to compute action statistics")

    arrays = [np.asarray(actions, dtype=np.float64) for actions in episode_actions]
    action_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else -1
    if action_dim < 1:
        raise ValueError(f"Expected episode actions with shape (steps, action_dim), got {arrays[0].shape}")
    if len(action_names) != action_dim:
        raise ValueError(f"Expected {action_dim} action names, got {len(action_names)}")
    for episode_index, actions in enumerate(arrays):
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise ValueError(
                f"Episode {episode_index} has shape {actions.shape}; expected (steps, {action_dim})"
            )
        if actions.shape[0] == 0:
            raise ValueError(f"Episode {episode_index} contains no actions")
        if not np.isfinite(actions).all():
            raise ValueError(f"Episode {episode_index} contains non-finite actions")

    actions_env = np.concatenate(arrays, axis=0)
    delta_parts = [np.diff(actions, axis=0) for actions in arrays if actions.shape[0] > 1]
    delta_actions_env = (
        np.concatenate(delta_parts, axis=0) if delta_parts else np.empty((0, action_dim), dtype=np.float64)
    )

    return ExecutedActionStatistics(
        actions_env=actions_env,
        delta_actions_env=delta_actions_env,
        episode_lengths=np.asarray([len(actions) for actions in arrays], dtype=np.int64),
        action_correlation=_pearson_correlation_matrix(actions_env),
        delta_correlation=_pearson_correlation_matrix(delta_actions_env),
        action_names=list(action_names),
    )


def _extract_executed_action_episodes(
    actions: Tensor,
    done_indices: Tensor,
    max_episodes: int,
) -> list[np.ndarray]:
    """Slice a rollout batch at each first-done index, optionally dropping an extra final-batch item."""
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape (batch, steps, action_dim), got {actions.shape}")
    if done_indices.ndim != 1 or done_indices.shape[0] != actions.shape[0]:
        raise ValueError(
            "done_indices must contain one index per action batch item, "
            f"got {done_indices.shape} for actions {actions.shape}"
        )
    if max_episodes < 0:
        raise ValueError(f"max_episodes must be non-negative, got {max_episodes}")

    episodes = []
    for episode_actions, done_index_tensor in zip(
        actions[:max_episodes], done_indices[:max_episodes], strict=True
    ):
        done_index = int(done_index_tensor.item())
        if not 0 <= done_index < episode_actions.shape[0]:
            raise ValueError(
                f"Done index {done_index} is outside an action sequence of length {episode_actions.shape[0]}"
            )
        episodes.append(episode_actions[: done_index + 1].detach().cpu().numpy())
    return episodes


def _save_correlation_csv(path: Path, correlation: np.ndarray, action_names: Sequence[str]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["dimension", *action_names])
        for action_name, row in zip(action_names, correlation, strict=True):
            writer.writerow([action_name, *row.tolist()])


def _save_correlation_plot(
    path: Path,
    correlation: np.ndarray,
    action_names: Sequence[str],
    title: str,
    sample_count: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "QuantileHead action-statistics visualization requires matplotlib. "
            "Install it with `uv pip install matplotlib`."
        ) from exc

    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    colormap = plt.get_cmap("coolwarm").with_extremes(bad="lightgray")
    image = axis.imshow(np.ma.masked_invalid(correlation), vmin=-1.0, vmax=1.0, cmap=colormap)
    axis.set_xticks(range(len(action_names)), action_names, rotation=45, ha="right")
    axis.set_yticks(range(len(action_names)), action_names)
    axis.set_title(f"{title}\nPearson correlation; samples={sample_count}")

    for row_index in range(len(action_names)):
        for column_index in range(len(action_names)):
            value = correlation[row_index, column_index]
            label = f"{value:.2f}" if np.isfinite(value) else "N/A"
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.55 else "black"
            axis.text(column_index, row_index, label, ha="center", va="center", color=text_color)

    figure.colorbar(image, ax=axis, label="correlation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_executed_action_statistics(
    episode_actions: Sequence[np.ndarray],
    action_names: Sequence[str],
    output_dir: Path,
    task_name: str,
) -> dict[str, Path]:
    """Save executed actions and Pearson matrices, including episode-safe ``R_delta``."""
    statistics = compute_executed_action_statistics(episode_actions, action_names)
    task_dir = _quantile_head_task_output_dir(output_dir, task_name)

    data_path = task_dir / "executed_actions.npz"
    r_action_csv_path = task_dir / "R_action.csv"
    r_delta_csv_path = task_dir / "R_delta.csv"
    r_action_plot_path = task_dir / "R_action.png"
    r_delta_plot_path = task_dir / "R_delta.png"

    np.savez_compressed(
        data_path,
        actions_env=statistics["actions_env"],
        delta_actions_env=statistics["delta_actions_env"],
        episode_lengths=statistics["episode_lengths"],
        episode_index=np.repeat(
            np.arange(len(statistics["episode_lengths"]), dtype=np.int64),
            statistics["episode_lengths"],
        ),
        delta_episode_index=np.repeat(
            np.arange(len(statistics["episode_lengths"]), dtype=np.int64),
            np.maximum(statistics["episode_lengths"] - 1, 0),
        ),
        r_action=statistics["action_correlation"],
        r_delta=statistics["delta_correlation"],
        action_names=np.asarray(statistics["action_names"]),
        scale="environment",
        correlation="pearson",
        delta_definition="action[t] - action[t-1] within each episode",
    )
    _save_correlation_csv(r_action_csv_path, statistics["action_correlation"], statistics["action_names"])
    _save_correlation_csv(r_delta_csv_path, statistics["delta_correlation"], statistics["action_names"])
    _save_correlation_plot(
        r_action_plot_path,
        statistics["action_correlation"],
        statistics["action_names"],
        f"{task_name} | executed actions",
        statistics["actions_env"].shape[0],
    )
    _save_correlation_plot(
        r_delta_plot_path,
        statistics["delta_correlation"],
        statistics["action_names"],
        f"{task_name} | executed action deltas",
        statistics["delta_actions_env"].shape[0],
    )

    return {
        "data": data_path,
        "r_action_csv": r_action_csv_path,
        "r_delta_csv": r_delta_csv_path,
        "r_action_plot": r_action_plot_path,
        "r_delta_plot": r_delta_plot_path,
    }


def _unwrap_quantile_head_policy(policy: nn.Module) -> nn.Module:
    """Return the QuantileHead policy under an optional PEFT wrapper."""
    get_base_model = getattr(policy, "get_base_model", None)
    base_policy = get_base_model() if callable(get_base_model) else policy

    config = getattr(base_policy, "config", None)
    if config is None or getattr(config, "type", None) != "quantile_head":
        raise ValueError(
            f"{QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV} is only supported for QuantileHead policies, "
            f"got {getattr(config, 'type', type(base_policy).__name__)}"
        )
    if config.action_generation_mode != "regression":
        raise ValueError(
            "QuantileHead action-distribution visualization requires action_generation_mode='regression'"
        )
    if config.use_pure_regression_head:
        raise ValueError(
            "QuantileHead action-distribution visualization requires a quantile head, but use_pure_regression_head=True. "
            f"Unset {QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV} to evaluate this deterministic policy."
        )
    return base_policy


def _postprocess_action_chunk(
    action: Tensor,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> Tensor:
    """Convert a normalized action chunk to the exact action scale sent to the environment."""
    processed_action = postprocessor(action)
    action_transition = env_postprocessor({ACTION: processed_action})
    env_action = action_transition[ACTION]
    if not isinstance(env_action, Tensor):
        raise TypeError(f"Expected a tensor action after postprocessing, got {type(env_action).__name__}")
    return env_action


@torch.inference_mode()
def sample_quantile_head_action_distribution(
    policy: nn.Module,
    batch: dict[str, Tensor],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    *,
    n_samples: int = 4096,
    sample_batch_size: int = 256,
    seed: int = 1234,
) -> QuantileHeadActionDistribution:
    """Sample independent marginals for diagnostics and return the median policy action."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if sample_batch_size < 1:
        raise ValueError(f"sample_batch_size must be positive, got {sample_batch_size}")

    quantile_head_policy = _unwrap_quantile_head_policy(policy)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    if tokens.shape[0] != 1:
        raise ValueError(
            "QuantileHead action-distribution visualization requires --eval.batch_size=1, "
            f"got batch size {tokens.shape[0]}"
        )

    config = quantile_head_policy.config
    action_dim = config.output_features[ACTION].shape[0]
    cuda_devices: list[int] = []
    if tokens.device.type == "cuda":
        device_index = tokens.device.index
        cuda_devices = [torch.cuda.current_device() if device_index is None else device_index]

    samples_normalized_parts = []
    samples_env_parts = []
    with torch.random.fork_rng(devices=cuda_devices):
        if tokens.device.type == "cuda":
            with torch.cuda.device(tokens.device):
                torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

        images, image_masks = quantile_head_policy._preprocess_images(batch)
        time_tensor = torch.zeros(tokens.shape[0], dtype=torch.float32, device=tokens.device)
        quantile_knots = quantile_head_policy.model.regress_quantiles_from_zero_query(
            images,
            image_masks,
            tokens,
            masks,
            time_tensor,
        )
        quantile_knots = quantile_knots[:, :, :action_dim, :].float()
        _, chunk_size, returned_action_dim, _ = quantile_knots.shape
        if returned_action_dim != action_dim or chunk_size != config.chunk_size:
            raise RuntimeError(
                "Unexpected QuantileHead quantile shape: "
                f"got {tuple(quantile_knots.shape)}, expected chunk/action dimensions "
                f"({config.chunk_size}, {action_dim})"
            )

        execution_horizon = min(config.n_action_steps, chunk_size)
        execution_knots = quantile_knots[:, :execution_horizon]
        for start in range(0, n_samples, sample_batch_size):
            current_batch_size = min(sample_batch_size, n_samples - start)
            repeated_knots = execution_knots.expand(current_batch_size, -1, -1, -1)
            marginal_levels = torch.rand(
                current_batch_size,
                execution_horizon,
                action_dim,
                device=quantile_knots.device,
                dtype=torch.float32,
            )
            normalized_actions = quantile_head_policy.model.interpolate_marginal_quantiles(
                repeated_knots,
                marginal_levels,
            ).float()
            env_actions = _postprocess_action_chunk(
                normalized_actions,
                postprocessor,
                env_postprocessor,
            )
            samples_normalized_parts.append(normalized_actions.cpu())
            samples_env_parts.append(env_actions.float().cpu())

        # Treat the quantile index as a batch dimension so action_dim stays last for postprocessing.
        knot_batch = quantile_knots[0, :execution_horizon].permute(2, 0, 1).contiguous()
        quantile_knots_env = _postprocess_action_chunk(
            knot_batch,
            postprocessor,
            env_postprocessor,
        ).permute(1, 2, 0)
        median_actions_normalized = quantile_head_policy.model.median_actions_from_quantiles(execution_knots)
        median_actions_env = _postprocess_action_chunk(
            median_actions_normalized,
            postprocessor,
            env_postprocessor,
        )

    samples_normalized = torch.cat(samples_normalized_parts, dim=0)
    samples_env = torch.cat(samples_env_parts, dim=0)
    quantile_knots_normalized = quantile_knots[0, :execution_horizon].cpu()
    quantile_knots_env = quantile_knots_env.float().cpu()
    median_actions_normalized = median_actions_normalized[0].float().cpu()
    median_actions_env = median_actions_env[0].float().cpu()
    quantile_levels = quantile_head_policy.model.quantile_levels.detach().float().cpu()

    expected_sample_shape = (n_samples, execution_horizon, action_dim)
    if samples_env.shape != expected_sample_shape:
        raise RuntimeError(
            f"Unexpected sampled action shape {tuple(samples_env.shape)}; expected {expected_sample_shape}"
        )
    if (
        not torch.isfinite(samples_env).all()
        or not torch.isfinite(quantile_knots_env).all()
        or not torch.isfinite(median_actions_env).all()
    ):
        raise RuntimeError("QuantileHead action-distribution visualization produced non-finite values")

    action_names = _quantile_head_action_names(config, action_dim)

    return QuantileHeadActionDistribution(
        marginal_samples_normalized=samples_normalized,
        marginal_samples_env=samples_env,
        median_actions_normalized=median_actions_normalized,
        median_actions_env=median_actions_env,
        quantile_knots_normalized=quantile_knots_normalized,
        quantile_knots_env=quantile_knots_env,
        quantile_levels=quantile_levels,
        action_names=list(action_names),
    )


def save_quantile_head_action_distribution(
    policy: nn.Module,
    batch: dict[str, Tensor],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    output_dir: Path,
    task_name: str,
    *,
    n_samples: int = 4096,
    horizon_step: int = 0,
) -> dict[str, Path]:
    """Save QuantileHead marginal diagnostics, median actions, and their numeric data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "QuantileHead action-distribution visualization requires matplotlib. "
            "Install it with `uv pip install matplotlib`."
        ) from exc

    distribution = sample_quantile_head_action_distribution(
        policy,
        batch,
        postprocessor,
        env_postprocessor,
        n_samples=n_samples,
    )
    quantile_head_policy = _unwrap_quantile_head_policy(policy)
    config = quantile_head_policy.config
    policy_decoder = getattr(
        quantile_head_policy,
        "_quantile_head_eval_decoder_name",
        config.marginal_quantile_inference_mode,
    )
    samples_env = distribution["marginal_samples_env"]
    quantile_knots_env = distribution["quantile_knots_env"]
    quantile_levels = distribution["quantile_levels"]
    action_names = distribution["action_names"]
    execution_horizon = samples_env.shape[1]
    action_dim = samples_env.shape[2]
    if not 0 <= horizon_step < execution_horizon:
        raise ValueError(f"horizon_step must be in [0, {execution_horizon - 1}], got {horizon_step}")

    task_dir = _quantile_head_task_output_dir(output_dir, task_name)

    data_path = task_dir / "quantile_head_distribution.npz"
    np.savez_compressed(
        data_path,
        marginal_samples_normalized=distribution["marginal_samples_normalized"].numpy(),
        marginal_samples_env=samples_env.numpy(),
        median_actions_normalized=distribution["median_actions_normalized"].numpy(),
        median_actions_env=distribution["median_actions_env"].numpy(),
        quantile_knots_normalized=distribution["quantile_knots_normalized"].numpy(),
        quantile_knots_env=quantile_knots_env.numpy(),
        quantile_levels=quantile_levels.numpy(),
        action_names=np.asarray(action_names),
        policy_decoder=policy_decoder,
        diagnostic_sampling="independent_marginals",
        diagnostic_distribution="learned_quantiles",
        sampling_temperature=config.marginal_quantile_sampling_temperature,
        sampling_min_level=config.marginal_quantile_sampling_min_level,
        sampling_max_level=config.marginal_quantile_sampling_max_level,
        sampling_ar_coefficient=config.marginal_quantile_sampling_ar_coefficient,
        sampling_seed=config.marginal_quantile_sampling_seed,
        sampling_exclude_joints=np.asarray(config.marginal_quantile_sampling_exclude_joints),
    )

    rows = (action_dim + 1) // 2
    title = f"{task_name}\nindependent marginal diagnostic samples; policy decoder={policy_decoder}"
    middle_quantile = int(torch.argmin(torch.abs(quantile_levels - 0.5)))

    histogram_figure, histogram_axes = plt.subplots(
        rows,
        2,
        figsize=(14, 3.2 * rows),
        constrained_layout=True,
    )
    for dim, axis in enumerate(np.asarray(histogram_axes).reshape(-1)):
        if dim >= action_dim:
            axis.axis("off")
            continue

        values = samples_env[:, horizon_step, dim].numpy()
        lower = float(quantile_knots_env[horizon_step, dim, 0])
        median = float(quantile_knots_env[horizon_step, dim, middle_quantile])
        upper = float(quantile_knots_env[horizon_step, dim, -1])
        axis.hist(values, bins=60, density=True, alpha=0.75)
        axis.axvspan(
            lower,
            upper,
            alpha=0.12,
            label=f"q{quantile_levels[0]:.3f}-q{quantile_levels[-1]:.3f}",
        )
        axis.axvline(median, color="crimson", linewidth=2, label="learned median")
        axis.set_title(f"{action_names[dim]} | horizon={horizon_step}")
        axis.set_xlabel("environment-scale action")
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
        axis.legend()

    histogram_figure.suptitle(title)
    histogram_path = task_dir / f"marginals_h{horizon_step:02d}.png"
    histogram_figure.savefig(histogram_path, dpi=180)
    plt.close(histogram_figure)

    cdf_figure, cdf_axes = plt.subplots(
        rows,
        2,
        figsize=(14, 3.2 * rows),
        constrained_layout=True,
    )
    for dim, axis in enumerate(np.asarray(cdf_axes).reshape(-1)):
        if dim >= action_dim:
            axis.axis("off")
            continue

        axis.plot(
            quantile_knots_env[horizon_step, dim].numpy(),
            quantile_levels.numpy(),
            marker="o",
            markersize=3,
        )
        axis.set_title(f"{action_names[dim]} | horizon={horizon_step}")
        axis.set_xlabel("environment-scale action")
        axis.set_ylabel("cumulative probability")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)

    cdf_figure.suptitle(f"{task_name}\nlearned marginal quantile knots")
    cdf_path = task_dir / f"cdf_h{horizon_step:02d}.png"
    cdf_figure.savefig(cdf_path, dpi=180)
    plt.close(cdf_figure)

    return {
        "histogram": histogram_path,
        "cdf": cdf_path,
        "data": data_path,
    }


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    return_task_name: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.
        (optional) "task_name": The natural-language task description shared by the vectorized
            environments. Included only when ``return_task_name`` is true.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        return_task_name: Whether to include the task description in the return dictionary. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    task_name = ""

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from sub-environments (prefer natural language description).
        # env.call() works with both SyncVectorEnv and AsyncVectorEnv.
        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                observation["task"] = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                observation["task"] = [""] * env.num_envs
        task_name = str(observation["task"][0])

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)

        distribution_dir = os.environ.get(QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV)
        if distribution_dir and step == 0:
            visualized_tasks = getattr(policy, "_quantile_head_visualized_tasks", set())
            if task_name not in visualized_tasks:
                output_paths = save_quantile_head_action_distribution(
                    policy,
                    observation,
                    postprocessor,
                    env_postprocessor,
                    Path(distribution_dir),
                    task_name,
                    n_samples=int(os.environ.get(QUANTILE_HEAD_ACTION_DISTRIBUTION_SAMPLES_ENV, "4096")),
                    horizon_step=int(os.environ.get(QUANTILE_HEAD_ACTION_DISTRIBUTION_HORIZON_ENV, "0")),
                )
                visualized_tasks.add(task_name)
                policy._quantile_head_visualized_tasks = visualized_tasks
                logging.info("Saved QuantileHead action-distribution outputs: %s", output_paths)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        elif "is_success" in info:
            is_success = info["is_success"]
            successes = (
                is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * env.num_envs
            )
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_task_name:
        ret["task_name"] = task_name
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos
    action_stats_dir_value = os.environ.get(QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV)
    action_stats_output_dir = Path(action_stats_dir_value) if action_stats_dir_value else None
    executed_action_episodes: list[np.ndarray] = []
    action_stats_task_name: str | None = None

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif hasattr(env, "call"):
            # Here we must render all frames and discard any we don't need.
            # Covers AsyncVectorEnv and _LazyAsyncVectorEnv (which wraps one).
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            return_task_name=action_stats_output_dir is not None,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        if action_stats_output_dir is not None:
            rollout_task_name = str(rollout_data.get("task_name", ""))
            if action_stats_task_name is None:
                action_stats_task_name = rollout_task_name
            elif rollout_task_name != action_stats_task_name:
                raise RuntimeError(
                    "Expected one task per eval_policy call, got "
                    f"{action_stats_task_name!r} and {rollout_task_name!r}"
                )

            remaining_episodes = n_episodes - len(executed_action_episodes)
            executed_action_episodes.extend(
                _extract_executed_action_episodes(
                    rollout_data[ACTION],
                    done_indices,
                    remaining_episodes,
                )
            )

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    if action_stats_output_dir is not None:
        if not executed_action_episodes:
            raise RuntimeError("No executed actions were collected for QuantileHead action statistics")
        quantile_head_policy = _unwrap_quantile_head_policy(policy)
        action_dim = executed_action_episodes[0].shape[1]
        action_names = _quantile_head_action_names(quantile_head_policy.config, action_dim)
        action_stats_paths = save_executed_action_statistics(
            executed_action_episodes,
            action_names,
            action_stats_output_dir,
            action_stats_task_name or "task",
        )
        info["action_stats_paths"] = {name: str(path) for name, path in action_stats_paths.items()}
        logging.info("Saved QuantileHead executed-action statistics: %s", action_stats_paths)

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.quantile_head_quantile_decoder != "policy" and cfg.env.max_parallel_tasks != 1:
        raise ValueError("QuantileHead evaluation decoder ablations require --env.max_parallel_tasks=1")

    if os.environ.get(QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV):
        if cfg.env.type != "libero":
            raise ValueError(
                f"{QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV} currently supports only --env.type=libero"
            )
        if cfg.policy.type != "quantile_head":
            raise ValueError(
                f"{QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV} requires a QuantileHead policy, got {cfg.policy.type}"
            )
        if cfg.eval.batch_size != 1:
            raise ValueError(f"{QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV} requires --eval.batch_size=1")
        if cfg.env.max_parallel_tasks != 1:
            raise ValueError(
                f"{QUANTILE_HEAD_ACTION_DISTRIBUTION_DIR_ENV} requires --env.max_parallel_tasks=1"
            )

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()
    decoder_metadata = configure_quantile_head_eval_decoder(
        policy,
        cfg.quantile_head_quantile_decoder,
        expectation_nodes=cfg.quantile_head_quantile_expectation_nodes,
        fixed_width_path=cfg.quantile_head_quantile_fixed_width_path,
    )
    if decoder_metadata is not None:
        logging.info("QuantileHead evaluation decoder override: %s", decoder_metadata)
    sampler_metadata = describe_quantile_head_quantile_sampler(policy)
    if sampler_metadata is not None:
        logging.info("QuantileHead resolved quantile sampler: %s", sampler_metadata)

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Calibration uses separate rollouts and freezes its statistics before any M evaluation.
    if cfg.quantile_head_quantile_decoder == "calibrate_width":
        width_path = cfg.quantile_head_quantile_width_output_path or Path(cfg.output_dir) / "fixed_width.json"
        info["quantile_head_quantile_width_calibration"] = export_quantile_head_quantile_width_calibration(
            policy, width_path
        )
        info["phase"] = "calibration"

    # Save info
    if decoder_metadata is not None:
        info["quantile_head_quantile_ablation"] = decoder_metadata
    if sampler_metadata is not None:
        info["quantile_head_quantile_sampler"] = sampler_metadata
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]
    action_stats_paths: NotRequired[dict[str, str]]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    metrics = TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )
    if "action_stats_paths" in task_result:
        metrics["action_stats_paths"] = task_result["action_stats_paths"]
    return metrics


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        prefetch_thread: threading.Thread | None = None
        for i, (task_group, task_id, env) in enumerate(tasks):
            if prefetch_thread is not None:
                prefetch_thread.join()
                prefetch_thread = None

            try:
                tg, tid, metrics = task_runner(task_group, task_id, env)
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
            finally:
                env.close()
                # Prefetch next task's workers *after* closing current env to prevent
                # GPU memory overlap between consecutive tasks.
                if i + 1 < len(tasks):
                    next_env = tasks[i + 1][2]
                    if hasattr(next_env, "_ensure"):
                        prefetch_thread = threading.Thread(target=next_env._ensure, daemon=True)
                        prefetch_thread.start()
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id, env)
            for fut in cf.as_completed(fut2meta):
                tg, tid, env = fut2meta[fut]
                try:
                    tg, tid, metrics = fut.result()
                    _accumulate_to(tg, metrics)
                    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                finally:
                    env.close()

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
