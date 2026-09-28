# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Evaluation-only QuantileHead quantile ablations, with no changes to model weights or training."""

import json
import math
from pathlib import Path
from statistics import NormalDist
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.utils.constants import ACTION

GAUSSIAN_FIT_LEVELS = (0.1, 0.9)
GAUSSIAN_CENTRAL_WIDTH = 2.0 * NormalDist().inv_cdf(GAUSSIAN_FIT_LEVELS[1])


def describe_quantile_head_quantile_sampler(policy: nn.Module) -> dict[str, Any] | None:
    """Record the resolved inference mask without advancing RNG or changing the policy."""
    get_base_model = getattr(policy, "get_base_model", None)
    base_policy = get_base_model() if callable(get_base_model) else policy
    config = getattr(base_policy, "config", None)
    if (
        getattr(config, "type", None) != "quantile_head"
        or getattr(config, "action_generation_mode", None) != "regression"
        or config.use_pure_regression_head
    ):
        return None

    action_dim = int(config.output_features[ACTION].shape[0])
    mask = base_policy._marginal_quantile_sampling_exclusion_mask(action_dim).cpu().tolist()
    names = config.action_feature_names
    decoder = getattr(
        base_policy, "_quantile_head_eval_decoder_name", config.marginal_quantile_inference_mode
    )
    sampling_enabled = (
        decoder not in {"median", "quantile_expectation", "calibrate_width"}
        and config.marginal_quantile_sampling_temperature > 0
    )
    soft_prompt_len = config.soft_prompt_len
    return {
        "decoder": decoder,
        "marginal_quantile_variant": config.marginal_quantile_variant,
        "action_dim": action_dim,
        "soft_prompt_len": soft_prompt_len,
        "soft_prompt_location": "action_suffix" if soft_prompt_len > 0 else None,
        "action_feature_names": list(names) if names is not None else None,
        "coordinate_names_complete": names is not None and len(names) >= action_dim,
        "sampling_exclude_joints": list(config.marginal_quantile_sampling_exclude_joints),
        "sampling_exclusion_mask": mask,
        "excluded_action_indices": [index for index, excluded in enumerate(mask) if excluded],
        "sampling_enabled": sampling_enabled,
        "sampled_action_indices": [
            index for index, excluded in enumerate(mask) if sampling_enabled and not excluded
        ],
        "sampling_temperature": config.marginal_quantile_sampling_temperature,
        "sampling_min_level": config.marginal_quantile_sampling_min_level,
        "sampling_max_level": config.marginal_quantile_sampling_max_level,
        "sampling_ar_coefficient": config.marginal_quantile_sampling_ar_coefficient,
        "sampling_seed": config.marginal_quantile_sampling_seed,
    }


def _median_without_sampling(policy: Any, knots: Tensor, *, independent: bool) -> Tensor:
    shape = knots.shape[:-1] if independent else (knots.shape[0], knots.shape[2])
    policy._last_marginal_quantile_levels = torch.full(shape, 0.5, dtype=torch.float32)
    return policy.model.median_actions_from_quantiles(knots)


def _keep_excluded_medians(policy: Any, actions: Tensor, knots: Tensor) -> Tensor:
    excluded = policy._marginal_quantile_sampling_exclusion_mask(knots.shape[2])
    excluded = excluded.to(device=actions.device)[None, None, :]
    median = policy.model.median_actions_from_quantiles(knots)
    return torch.where(excluded, median, actions)


@torch.no_grad()
def _decode_independent_rank(policy: Any, knots: Tensor) -> Tensor:
    """E: fresh independent ranks for every batch item, horizon step and action coordinate."""
    temperature = float(policy.config.marginal_quantile_sampling_temperature)
    if temperature == 0.0:
        return _median_without_sampling(policy, knots, independent=True)

    latent = torch.randn(
        knots.shape[:-1], generator=policy._get_marginal_quantile_sampling_generator(), dtype=torch.float32
    )
    # Same rank marginal as the policy sampler; only the temporal coupling changes.
    uniform_levels = 0.5 * (1.0 + torch.erf(temperature * latent / math.sqrt(2.0)))
    centered_rank = 2.0 * uniform_levels - 1.0
    min_level = float(policy.config.marginal_quantile_sampling_min_level)
    max_level = float(policy.config.marginal_quantile_sampling_max_level)
    levels = torch.where(
        centered_rank < 0,
        0.5 + centered_rank * (0.5 - min_level),
        0.5 + centered_rank * (max_level - 0.5),
    )
    excluded = policy._marginal_quantile_sampling_exclusion_mask(knots.shape[2])
    levels[..., excluded] = 0.5
    policy._last_marginal_quantile_levels = levels.clone()
    # Do not read or update AR state: rho has no role in this ablation.
    actions = policy.model.interpolate_marginal_quantiles(
        knots, levels.to(device=knots.device, dtype=knots.dtype)
    )
    return _keep_excluded_medians(policy, actions, knots)


@torch.no_grad()
def _decode_gaussian_temporal(policy: Any, knots: Tensor) -> Tensor:
    """F: match the median and 10--90% width, keeping the policy's exact AR rank process."""
    if float(policy.config.marginal_quantile_sampling_temperature) == 0.0:
        return _median_without_sampling(policy, knots, independent=False)

    # Exactly one call, including for collapsed quantiles, preserves the baseline RNG/AR trajectory.
    levels = policy._sample_temporally_coherent_quantile_levels(knots.shape[0], knots.shape[2])
    working_knots = knots if knots.dtype == torch.float64 else knots.float()
    anchors = [
        policy.model.interpolate_marginal_quantiles(
            working_knots,
            torch.full(knots.shape[:-1], level, device=knots.device, dtype=working_knots.dtype),
        )
        for level in GAUSSIAN_FIT_LEVELS
    ]
    scale = (anchors[1] - anchors[0]).clamp_min(0.0) / GAUSSIAN_CENTRAL_WIDTH
    median = policy.model.median_actions_from_quantiles(working_knots)

    # The sampler runs on CPU. Invert there in float32, including on MPS or bfloat16 models.
    # Endpoint clipping is numerical only; it avoids inf and 0 * inf for zero-width coordinates.
    rank_eps = torch.finfo(torch.float32).eps
    # Match the original decoder's rank quantization before a stable inverse CDF.
    inverse_cdf_levels = levels.cpu().to(dtype=knots.dtype).float()
    normal_ranks = torch.special.ndtri(inverse_cdf_levels.clamp(rank_eps, 1.0 - rank_eps))
    normal_ranks = normal_ranks.to(device=knots.device, dtype=working_knots.dtype)[:, None, :]
    actions = median + scale * normal_ranks
    return _keep_excluded_medians(policy, actions, knots)


def _central_width(policy: Any, knots: Tensor) -> Tensor:
    """Measure the learned 10--90% width, independently of the sampling rank bounds."""
    working_knots = knots if knots.dtype == torch.float64 else knots.float()
    anchors = [
        policy.model.interpolate_marginal_quantiles(
            working_knots, torch.full_like(working_knots[..., 0], level)
        )
        for level in GAUSSIAN_FIT_LEVELS
    ]
    return (anchors[1] - anchors[0]).clamp_min(0.0)


def _expectation_weights(policy: Any, num_nodes: int) -> Tensor:
    """Integrate the actual tempered rank law using deterministic probability-space quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(num_nodes)
    probabilities = torch.from_numpy((nodes + 1.0) / 2.0)
    weights = torch.from_numpy(weights / 2.0)
    latent = torch.special.ndtri(probabilities)
    temperature = float(policy.config.marginal_quantile_sampling_temperature)
    centered_rank = torch.erf(temperature * latent / math.sqrt(2.0))
    ranks = torch.where(
        centered_rank < 0,
        0.5 + centered_rank * (0.5 - policy.config.marginal_quantile_sampling_min_level),
        0.5 + centered_rank * (policy.config.marginal_quantile_sampling_max_level - 0.5),
    )
    # Interpolation is linear in the knots. Integrate its basis once, then each
    # prediction only needs a weighted sum; no RNG or per-node GPU loop is needed.
    num_knots = policy.model.quantile_levels.numel()
    basis = torch.eye(num_knots, dtype=torch.float64)[:, None, None, :]
    basis = basis.expand(num_knots, num_nodes, 1, num_knots)
    values = policy.model.interpolate_marginal_quantiles(
        basis, ranks[None, :, None].expand(num_knots, num_nodes, 1)
    )
    return (values[..., 0] * weights).sum(dim=1)


@torch.no_grad()
def _decode_quantile_expectation(policy: Any, knots: Tensor) -> Tensor:
    """L: deterministic expectation under the configured tempered, bounded rank law."""
    if float(policy.config.marginal_quantile_sampling_temperature) == 0.0:
        return _median_without_sampling(policy, knots, independent=False)
    working_knots = knots if knots.dtype == torch.float64 else knots.float()
    weights = policy._quantile_head_expectation_weights.to(device=knots.device, dtype=working_knots.dtype)
    actions = (working_knots * weights).sum(dim=-1)
    # An expectation has no single sampled rank. Do not advance RNG or AR state.
    policy._last_marginal_quantile_levels = None
    return _keep_excluded_medians(policy, actions, knots)


@torch.no_grad()
def _decode_fixed_width(policy: Any, knots: Tensor) -> Tensor:
    """M: retain the center and normalized quantile shape, replacing state-dependent width."""
    if float(policy.config.marginal_quantile_sampling_temperature) == 0.0:
        return _median_without_sampling(policy, knots, independent=False)
    width = _central_width(policy, knots)
    reference = policy._quantile_head_fixed_width.to(device=knots.device, dtype=width.dtype)
    if reference.shape[-1] != knots.shape[2] or (
        reference.ndim == 2 and reference.shape[0] != knots.shape[1]
    ):
        raise ValueError(
            f"Fixed width shape {tuple(reference.shape)} does not match action chunk "
            f"{tuple(knots.shape[1:3])}; use a [D] vector or an exact [H, D] matrix"
        )
    levels = policy._sample_temporally_coherent_quantile_levels(knots.shape[0], knots.shape[2])
    levels = levels.to(device=knots.device, dtype=knots.dtype)[:, None, :].expand(*knots.shape[:-1])
    sampled = policy.model.interpolate_marginal_quantiles(knots, levels)
    median = policy.model.median_actions_from_quantiles(knots)
    # Collapsed intervals contain no usable shape; never amplify numerical residuals.
    epsilon = torch.finfo(width.dtype).eps
    offset = (sampled - median) / width.clamp_min(epsilon)
    actions = median + torch.where(width > epsilon, offset * reference, 0.0)
    return _keep_excluded_medians(policy, actions, knots)


@torch.no_grad()
def _decode_calibrate_width(policy: Any, knots: Tensor) -> Tensor:
    """Collect widths on separate median rollouts without changing their action policy."""
    width = _central_width(policy, knots).detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(width).all():
        raise ValueError("Cannot calibrate fixed widths from non-finite quantile predictions")
    square_sum = width.square().sum(dim=0)
    previous = policy._quantile_head_width_square_sum
    if previous is not None and previous.shape != square_sum.shape:
        raise ValueError("Action chunk shape changed during fixed-width calibration")
    policy._quantile_head_width_square_sum = square_sum if previous is None else previous + square_sum
    policy._quantile_head_width_observation_count += width.shape[0]
    return _median_without_sampling(policy, knots, independent=False)


def export_quantile_head_quantile_width_calibration(
    policy: nn.Module, output_path: str | Path
) -> dict[str, Any]:
    """Freeze calibration statistics for reuse across M evaluation seeds."""
    get_base_model = getattr(policy, "get_base_model", None)
    base_policy = get_base_model() if callable(get_base_model) else policy
    if getattr(base_policy, "_quantile_head_eval_decoder_name", None) != "calibrate_width":
        raise ValueError("Width calibration export requires decoder='calibrate_width'")
    count = base_policy._quantile_head_width_observation_count
    if count == 0:
        raise ValueError("No action predictions were collected for fixed-width calibration")
    widths = (base_policy._quantile_head_width_square_sum / count).sqrt()
    data = {
        "format_version": 1,
        "width_quantiles": list(GAUSSIAN_FIT_LEVELS),
        "method": "root_mean_square_quantile_width",
        "observation_count": count,
        "horizon": widths.shape[0],
        "action_dim": widths.shape[1],
        "widths": widths.tolist(),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"path": str(path), **data}


def configure_quantile_head_eval_decoder(
    policy: nn.Module,
    decoder: str,
    *,
    expectation_nodes: int = 128,
    fixed_width_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Override only the loaded instance's decoder; ``policy`` leaves normal inference untouched.

    Explicit ablations take precedence over the checkpoint's median/temporal_sample setting.
    Existing temperature, rank bounds, excluded joints and sampler seed still apply. The
    original reset and action queue remain in use, and no parameter/config tensors are changed.
    """
    decoders = {
        "independent_rank": _decode_independent_rank,
        "gaussian_temporal": _decode_gaussian_temporal,
        "quantile_expectation": _decode_quantile_expectation,
        "fixed_width": _decode_fixed_width,
        "calibrate_width": _decode_calibrate_width,
    }
    if decoder == "policy":
        return None
    if decoder not in decoders:
        raise ValueError(
            f"Unknown QuantileHead evaluation decoder {decoder!r}; choose from {['policy', *decoders]}"
        )

    get_base_model = getattr(policy, "get_base_model", None)
    base_policy = get_base_model() if callable(get_base_model) else policy
    config = getattr(base_policy, "config", None)
    if (
        getattr(config, "type", None) != "quantile_head"
        or getattr(config, "action_generation_mode", None) != "regression"
    ):
        raise ValueError(
            "QuantileHead evaluation decoders require a QuantileHead policy with action_generation_mode='regression'"
        )
    if config.use_pure_regression_head:
        raise ValueError(
            f"QuantileHead evaluation decoder {decoder!r} requires a quantile head, but use_pure_regression_head=True. "
            "Use quantile_head_quantile_decoder='policy' to evaluate a pure regression policy."
        )
    variant = config.marginal_quantile_variant
    if variant in {"median_only", "independent"}:
        raise ValueError(
            f"marginal_quantile_variant={variant!r} requires median inference; "
            f"the {decoder!r} distribution decoder is not supported. Use quantile_head_quantile_decoder='policy'."
        )

    temperature = float(config.marginal_quantile_sampling_temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("marginal_quantile_sampling_temperature must be finite and non-negative")
    min_level = float(config.marginal_quantile_sampling_min_level)
    max_level = float(config.marginal_quantile_sampling_max_level)
    if not 0 <= min_level < 0.5 < max_level <= 1:
        raise ValueError(
            "QuantileHead evaluation rank bounds must satisfy 0 <= min_level < 0.5 < max_level <= 1"
        )

    rho = float(config.marginal_quantile_sampling_ar_coefficient)
    if decoder in {"gaussian_temporal", "fixed_width", "calibrate_width"}:
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("marginal_quantile_sampling_ar_coefficient must be finite and in [0, 1]")
        learned_levels = base_policy.model.quantile_levels
        if (
            learned_levels is None
            or learned_levels[0].item() > GAUSSIAN_FIT_LEVELS[0] + 1e-6
            or learned_levels[-1].item() < GAUSSIAN_FIT_LEVELS[1] - 1e-6
        ):
            raise ValueError(f"{decoder} requires learned quantile levels covering [0.1, 0.9]")

    if decoder == "quantile_expectation":
        if (
            isinstance(expectation_nodes, bool)
            or not isinstance(expectation_nodes, int)
            or expectation_nodes < 2
        ):
            raise ValueError("expectation_nodes must be an integer >= 2")
        base_policy._quantile_head_expectation_weights = _expectation_weights(base_policy, expectation_nodes)
    elif decoder == "fixed_width":
        if fixed_width_path is None:
            raise ValueError(
                "fixed_width requires quantile_head_quantile_fixed_width_path pointing to a calibration JSON"
            )
        with Path(fixed_width_path).open(encoding="utf-8") as stream:
            calibration = json.load(stream)
        if not isinstance(calibration, dict) or "widths" not in calibration:
            raise ValueError("Fixed-width JSON must contain a 'widths' array")
        widths = torch.as_tensor(calibration["widths"], dtype=torch.float64)
        if (
            widths.ndim not in {1, 2}
            or widths.numel() == 0
            or not torch.isfinite(widths).all()
            or (widths < 0).any()
        ):
            raise ValueError("Fixed widths must be a nonempty, finite, non-negative [D] or [H, D] array")
        if widths.shape[-1] != int(config.output_features[ACTION].shape[0]):
            raise ValueError("Fixed-width action dimension does not match the loaded policy")
        base_policy._quantile_head_fixed_width = widths
    elif decoder == "calibrate_width":
        base_policy._quantile_head_width_square_sum = None
        base_policy._quantile_head_width_observation_count = 0

    base_policy._decode_regression_quantiles = MethodType(decoders[decoder], base_policy)
    base_policy._quantile_head_eval_decoder_name = decoder
    metadata: dict[str, Any] = {
        "decoder": decoder,
        "policy_inference_mode": config.marginal_quantile_inference_mode,
        "sampling_temperature": temperature,
        "sampling_min_level": min_level,
        "sampling_max_level": max_level,
        "configured_ar_coefficient": rho,
        "effective_ar_coefficient": rho if decoder in {"gaussian_temporal", "fixed_width"} else None,
        "sampling_seed": config.marginal_quantile_sampling_seed,
        "sampling_exclude_joints": list(config.marginal_quantile_sampling_exclude_joints),
        "rank_coupling": (
            "chunk_shared_ar"
            if decoder in {"gaussian_temporal", "fixed_width"}
            else "independent_per_step"
            if decoder == "independent_rank"
            else "deterministic"
        ),
    }
    if decoder == "gaussian_temporal":
        metadata.update(
            gaussian_fit_levels=list(GAUSSIAN_FIT_LEVELS),
            gaussian_center="predicted_median",
            gaussian_scale="(Q(0.9)-Q(0.1))/(2*Phi_inverse(0.9))",
            gaussian_inverse_cdf_rank_epsilon=torch.finfo(torch.float32).eps,
        )
    elif decoder == "quantile_expectation":
        metadata.update(
            expectation_nodes=expectation_nodes,
            integration="gauss_legendre_uniform_latent_probability",
            quantile_weights=base_policy._quantile_head_expectation_weights.tolist(),
        )
    elif decoder == "fixed_width":
        metadata.update(
            fixed_width_path=str(fixed_width_path),
            fixed_widths=base_policy._quantile_head_fixed_width.tolist(),
            width_quantiles=list(GAUSSIAN_FIT_LEVELS),
        )
    elif decoder == "calibrate_width":
        metadata.update(phase="calibration", width_quantiles=list(GAUSSIAN_FIT_LEVELS))
    return metadata
