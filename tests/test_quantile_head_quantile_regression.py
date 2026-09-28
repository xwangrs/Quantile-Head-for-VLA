#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig  # noqa: E402
from lerobot.policies.quantile_head.modeling_quantile_head import (  # noqa: E402
    QuantileHeadPolicy,
    QuantileHeadPytorch,
    build_monotonic_quantile_knots,
    extend_quantile_knots_to_unit_interval,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from tests.quantile_head_test_utils import make_quantile_head_config_stub  # noqa: E402


def test_monotonic_quantile_knots_preserve_median():
    torch.manual_seed(0)
    median = torch.randn(2, 3, 4)
    raw_gaps = torch.randn(2, 3, 4, 20)

    knots = build_monotonic_quantile_knots(median, raw_gaps, initial_gap=0.02)

    assert knots.shape == (2, 3, 4, 21)
    assert torch.all(knots[..., 1:] > knots[..., :-1])
    torch.testing.assert_close(knots[..., 10], median)


def test_quantile_knots_are_linearly_extended_to_probability_zero_and_one():
    quantile_knots = torch.tensor([[[[-1.0, 0.0, 1.0]]]])
    quantile_levels = torch.tensor([0.1, 0.5, 0.9])

    extended_knots, extended_levels = extend_quantile_knots_to_unit_interval(
        quantile_knots,
        quantile_levels,
    )

    torch.testing.assert_close(extended_knots, torch.tensor([[[[-1.25, -1.0, 0.0, 1.0, 1.25]]]]))
    torch.testing.assert_close(extended_levels, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]))


def test_pinball_loss_shrinks_all_gaps_for_a_memorized_single_target():
    raw_gaps = torch.zeros(1, 1, 1, 20, requires_grad=True)
    median = torch.zeros_like(raw_gaps[..., 0])
    quantile_knots = build_monotonic_quantile_knots(median, raw_gaps, initial_gap=0.02)
    quantile_levels = torch.linspace(0.025, 0.975, quantile_knots.shape[-1])
    errors = -quantile_knots
    levels = quantile_levels.view(1, 1, 1, -1)
    pinball_loss = torch.maximum(levels * errors, (levels - 1.0) * errors).mean()

    pinball_loss.backward()

    assert raw_gaps.grad is not None
    assert torch.isfinite(raw_gaps.grad).all()
    # Gradient descent subtracts this positive gradient, reducing every raw gap.
    assert (raw_gaps.grad > 0).all()


def test_quantile_inference_defaults_to_deterministic_median():
    config = QuantileHeadConfig()

    assert config.marginal_quantile_inference_mode == "median"
    assert config.marginal_quantile_sampling_temperature == pytest.approx(0.35)
    assert config.marginal_quantile_sampling_min_level == pytest.approx(0.1)
    assert config.marginal_quantile_sampling_max_level == pytest.approx(0.9)
    assert config.marginal_quantile_sampling_ar_coefficient == pytest.approx(0.9)
    assert config.marginal_quantile_sampling_seed == 1000
    assert config.marginal_quantile_sampling_exclude_joints == ["gripper"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"marginal_quantile_inference_mode": "random"}, "inference_mode"),
        ({"marginal_quantile_sampling_temperature": -0.1}, "temperature"),
        ({"marginal_quantile_sampling_min_level": 0.5}, "min_level"),
        ({"marginal_quantile_sampling_max_level": 0.5}, "max_level"),
        ({"marginal_quantile_sampling_ar_coefficient": 1.1}, "ar_coefficient"),
        ({"marginal_quantile_sampling_seed": -1}, "sampling_seed"),
        ({"marginal_quantile_sampling_exclude_joints": [""]}, "exclude_joints"),
    ],
)
def test_quantile_inference_config_rejects_invalid_values(overrides, match):
    with pytest.raises(ValueError, match=match):
        QuantileHeadConfig(**overrides)


def _make_loss_mask_policy(mask_ratio: float):
    policy = QuantileHeadPolicy.__new__(QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = make_quantile_head_config_stub(action_loss_mask_ratio=mask_ratio)
    return policy


def test_action_loss_mask_excludes_padding_and_masks_only_valid_coordinates():
    policy = _make_loss_mask_policy(mask_ratio=0.25)
    policy.train()
    values = torch.zeros(3, 4, 2)
    action_is_pad = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, True],
            [True, True, True, True],
        ]
    )

    torch.manual_seed(4)
    keep_mask = policy.make_action_loss_keep_mask(values, action_is_pad)
    padding_keep_mask = policy.make_action_padding_keep_mask(values, action_is_pad)

    assert not keep_mask[~padding_keep_mask].any()
    valid_counts = padding_keep_mask.flatten(1).sum(dim=1)
    expected_kept_counts = valid_counts - torch.minimum(
        torch.ceil(valid_counts.float() * 0.25).long(),
        (valid_counts - 1).clamp_min(0),
    )
    torch.testing.assert_close(keep_mask.flatten(1).sum(dim=1), expected_kept_counts)


def test_action_loss_mask_in_eval_excludes_only_padding():
    policy = _make_loss_mask_policy(mask_ratio=0.25)
    policy.eval()
    values = torch.zeros(2, 3, 2)
    action_is_pad = torch.tensor([[False, True, True], [False, False, True]])

    keep_mask = policy.make_action_loss_keep_mask(values, action_is_pad)
    expected = ~action_is_pad.unsqueeze(-1).expand_as(values)

    torch.testing.assert_close(keep_mask, expected)


class _FixedFlowLossModel(torch.nn.Module):
    def __init__(self, losses: torch.Tensor):
        super().__init__()
        self.losses = losses

    @staticmethod
    def sample_noise(shape, device):
        return torch.zeros(shape, device=device)

    @staticmethod
    def sample_time(batch_size, device):
        return torch.zeros(batch_size, device=device)

    def forward(self, images, img_masks, tokens, masks, actions, noise, time):
        del images, img_masks, tokens, masks, actions, noise, time
        return self.losses


class _FixedRegressionModel(QuantileHeadPytorch):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = make_quantile_head_config_stub(
            marginal_quantile_min_level=0.1,
            marginal_quantile_num_knots=3,
        )
        self.register_buffer("quantile_levels", torch.tensor([0.1, 0.5, 0.9]), persistent=False)

    def forward_regression(self, images, img_masks, tokens, masks, time, actions):
        del images, img_masks, tokens, masks, time
        knots = torch.tensor([-1.0, 0.0, 1.0], device=actions.device)
        return knots.view(1, 1, 1, 3).expand(*actions.shape, 3)

    def regress_quantiles_from_zero_query(self, images, img_masks, tokens, masks, time):
        del images, img_masks, masks, time
        knots = torch.tensor([-1.0, 0.25, 1.0], device=tokens.device)
        return knots.view(1, 1, 1, 3).expand(tokens.shape[0], 3, 1, 3)


def _make_quantile_decoder_policy(
    *,
    inference_mode: str = "temporal_sample",
    temperature: float = 0.35,
    min_level: float = 0.1,
    max_level: float = 0.9,
    ar_coefficient: float = 0.9,
    seed: int = 1000,
    action_feature_names: list[str] | None = None,
    exclude_joints: list[str] | None = None,
):
    policy = QuantileHeadPolicy.__new__(QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = make_quantile_head_config_stub(
        n_action_steps=3,
        marginal_quantile_inference_mode=inference_mode,
        marginal_quantile_sampling_temperature=temperature,
        marginal_quantile_sampling_min_level=min_level,
        marginal_quantile_sampling_max_level=max_level,
        marginal_quantile_sampling_ar_coefficient=ar_coefficient,
        marginal_quantile_sampling_seed=seed,
        marginal_quantile_sampling_exclude_joints=exclude_joints or [],
        action_feature_names=action_feature_names,
    )
    policy.model = _FixedRegressionModel()
    policy._marginal_quantile_sampling_generator = torch.Generator(device="cpu")
    policy._marginal_quantile_sampling_generator.manual_seed(seed)
    policy.reset()
    return policy


def _identity_quantile_knots(batch_size: int, horizon: int, action_dim: int):
    """Return knots whose piecewise-linear inverse CDF is Q(u) = u."""
    return torch.tensor([0.1, 0.5, 0.9]).view(1, 1, 1, 3).expand(batch_size, horizon, action_dim, 3)


def test_median_decoder_returns_the_explicit_middle_knot():
    model = _FixedRegressionModel()
    quantile_knots = torch.tensor(
        [
            [[[-2.0, -0.5, 3.0], [-4.0, 1.5, 8.0]]],
            [[[-1.0, 0.25, 7.0], [-3.0, 2.5, 9.0]]],
        ]
    )

    actions = model.median_actions_from_quantiles(quantile_knots)

    torch.testing.assert_close(actions, quantile_knots[..., 1])


def test_median_decoder_rejects_wrong_number_of_knots():
    model = _FixedRegressionModel()

    with pytest.raises(ValueError, match="matching final dimensions"):
        model.median_actions_from_quantiles(torch.zeros(1, 2, 1, 5))


def test_policy_median_decoder_is_exact_and_does_not_advance_any_rng():
    policy = _make_quantile_decoder_policy(inference_mode="median", temperature=100.0, seed=17)
    quantile_knots = torch.tensor(
        [
            [
                [[-2.0, -0.5, 3.0], [-4.0, 1.5, 8.0]],
                [[-1.0, 0.25, 7.0], [-3.0, 2.5, 9.0]],
            ]
        ]
    )
    sampler_rng_state = policy._marginal_quantile_sampling_generator.get_state().clone()
    global_rng_state = torch.random.get_rng_state().clone()

    actions = policy._decode_regression_quantiles(quantile_knots)

    assert torch.equal(actions, quantile_knots[..., 1])
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_rng_state)
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)
    assert torch.equal(policy._last_marginal_quantile_levels, torch.full((1, 2), 0.5))


def test_temporal_sample_with_zero_temperature_is_exact_median_without_rng_use():
    policy = _make_quantile_decoder_policy(
        temperature=0.0, seed=19, action_feature_names=["actions"], exclude_joints=["gripper"]
    )
    quantile_knots = _identity_quantile_knots(batch_size=2, horizon=4, action_dim=7)
    sampler_rng_state = policy._marginal_quantile_sampling_generator.get_state().clone()

    actions = policy._decode_regression_quantiles(quantile_knots)

    assert torch.equal(actions, torch.full((2, 4, 7), 0.5))
    assert torch.equal(policy._last_marginal_quantile_levels, torch.full((2, 7), 0.5))
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_rng_state)
    assert policy._marginal_quantile_sampling_latent is None


def test_temporal_sample_uses_one_bounded_quantile_per_dimension_for_the_whole_chunk():
    policy = _make_quantile_decoder_policy(
        temperature=0.7,
        min_level=0.2,
        max_level=0.85,
        seed=23,
    )
    quantile_knots = _identity_quantile_knots(batch_size=2, horizon=4, action_dim=3)

    actions = policy._decode_regression_quantiles(quantile_knots)
    levels = policy._last_marginal_quantile_levels

    assert levels is not None
    assert levels.shape == (2, 3)
    assert torch.all(levels >= 0.2)
    assert torch.all(levels <= 0.85)
    torch.testing.assert_close(actions, levels[:, None, :].expand(2, 4, 3))


def test_temporal_sample_seed_and_ar_replanning_are_reproducible():
    seed = 31
    rho = 0.75
    first_policy = _make_quantile_decoder_policy(seed=seed, ar_coefficient=rho)
    second_policy = _make_quantile_decoder_policy(seed=seed, ar_coefficient=rho)
    quantile_knots = _identity_quantile_knots(batch_size=2, horizon=3, action_dim=3)
    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(seed)
    expected_first_latent = torch.randn((2, 3), generator=reference_generator)
    innovation = torch.randn((2, 3), generator=reference_generator)
    expected_second_latent = rho * expected_first_latent + (1.0 - rho**2) ** 0.5 * innovation

    first_actions = first_policy._decode_regression_quantiles(quantile_knots)
    second_actions = second_policy._decode_regression_quantiles(quantile_knots)
    torch.testing.assert_close(first_actions, second_actions)
    torch.testing.assert_close(first_policy._marginal_quantile_sampling_latent, expected_first_latent)

    first_actions = first_policy._decode_regression_quantiles(quantile_knots)
    second_actions = second_policy._decode_regression_quantiles(quantile_knots)

    torch.testing.assert_close(first_actions, second_actions)
    torch.testing.assert_close(first_policy._marginal_quantile_sampling_latent, expected_second_latent)


def test_temporal_sample_keeps_excluded_gripper_at_the_median():
    policy = _make_quantile_decoder_policy(
        temperature=1.0,
        action_feature_names=["delta_x", "robot.gripper"],
        exclude_joints=["gripper"],
        seed=47,
    )
    quantile_knots = _identity_quantile_knots(batch_size=1, horizon=5, action_dim=2)

    actions = policy._decode_regression_quantiles(quantile_knots)

    assert torch.equal(actions[..., 1], torch.full((1, 5), 0.5))
    assert policy._last_marginal_quantile_levels is not None
    assert policy._last_marginal_quantile_levels[0, 1].item() == 0.5


@pytest.mark.parametrize("action_feature_names", [None, [], ["actions"]])
@pytest.mark.parametrize("rho", [0.0, 0.7, 1.0])
def test_temporal_sample_uses_last_dimension_as_legacy_gripper_fallback(action_feature_names, rho):
    policy = _make_quantile_decoder_policy(
        temperature=1.0,
        ar_coefficient=rho,
        action_feature_names=action_feature_names,
        exclude_joints=["gripper"],
        seed=53,
    )
    sample_all = _make_quantile_decoder_policy(
        temperature=1.0, ar_coefficient=rho, action_feature_names=action_feature_names, seed=53
    )
    quantile_knots = _identity_quantile_knots(batch_size=1, horizon=3, action_dim=7)

    for _ in range(2):
        actions = policy._decode_regression_quantiles(quantile_knots)
        all_actions = sample_all._decode_regression_quantiles(quantile_knots)

        assert torch.equal(actions[..., -1], torch.full((1, 3), 0.5))
        assert policy._last_marginal_quantile_levels[0, -1].item() == 0.5
        assert not torch.equal(all_actions[..., -1], actions[..., -1])
        assert sample_all._last_marginal_quantile_levels[0, -1].item() != 0.5
        assert torch.equal(actions[..., :-1], all_actions[..., :-1])
        assert torch.equal(
            policy._marginal_quantile_sampling_latent, sample_all._marginal_quantile_sampling_latent
        )
        assert torch.equal(
            policy._marginal_quantile_sampling_generator.get_state(),
            sample_all._marginal_quantile_sampling_generator.get_state(),
        )


@pytest.mark.parametrize(
    ("action_feature_names", "exclude_joints", "excluded_index"),
    [
        (["x", "y", "robot.gripper", "z", "rx", "ry", "rz"], ["gripper"], 2),
        (["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"], ["gripper"], None),
        (["x", "y", "z", "rx", "ry", "rz", "gripper"], [], None),
    ],
)
def test_temporal_sample_respects_complete_joint_names_and_empty_exclusions(
    action_feature_names, exclude_joints, excluded_index
):
    policy = _make_quantile_decoder_policy(
        action_feature_names=action_feature_names, exclude_joints=exclude_joints, seed=53
    )
    quantile_knots = _identity_quantile_knots(batch_size=1, horizon=3, action_dim=7)

    actions = policy._decode_regression_quantiles(quantile_knots)
    levels = policy._last_marginal_quantile_levels

    for index in range(7):
        if index == excluded_index:
            assert torch.equal(actions[..., index], quantile_knots[..., index, 1])
            assert levels[0, index].item() == 0.5
        else:
            assert not torch.equal(actions[..., index], quantile_knots[..., index, 1])
            assert levels[0, index].item() != 0.5


def test_reset_clears_temporal_sample_state_without_reseeding():
    policy = _make_quantile_decoder_policy(seed=59)
    quantile_knots = _identity_quantile_knots(batch_size=1, horizon=2, action_dim=2)
    policy._decode_regression_quantiles(quantile_knots)
    previous_latent = policy._marginal_quantile_sampling_latent.clone()
    sampler_rng_state = policy._marginal_quantile_sampling_generator.get_state().clone()

    policy.reset()

    assert policy._marginal_quantile_sampling_latent is None
    assert policy._last_marginal_quantile_levels is None
    assert torch.equal(policy._marginal_quantile_sampling_generator.get_state(), sampler_rng_state)

    policy._decode_regression_quantiles(quantile_knots)
    assert not torch.equal(policy._marginal_quantile_sampling_latent, previous_latent)


def test_regression_predict_action_chunk_always_executes_marginal_medians():
    policy = QuantileHeadPolicy.__new__(QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = make_quantile_head_config_stub(
        action_generation_mode="regression",
        output_features={ACTION: SimpleNamespace(shape=(1,))},
    )
    policy.model = _FixedRegressionModel()
    policy._preprocess_images = lambda batch: ([], [])
    batch = {
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }

    actions = policy.predict_action_chunk(batch)

    torch.testing.assert_close(actions, torch.full((2, 3, 1), 0.25))


def test_tail_extended_inverse_interpolates_without_outer_knot_spikes():
    model = _FixedRegressionModel()
    requested_levels = torch.tensor([0.02, 0.05, 0.099, 0.101, 0.3, 0.5, 0.8, 0.899, 0.901, 0.95, 0.98]).view(
        1, -1, 1
    )
    quantile_knots = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3)
    quantile_knots = quantile_knots.expand(1, requested_levels.shape[1], 1, 3)

    actions = model.interpolate_marginal_quantiles(quantile_knots, requested_levels)

    assert torch.all(actions[requested_levels < 0.1] < -1.0)
    assert torch.all(actions[requested_levels > 0.9] > 1.0)
    assert not torch.any(actions == -1.0)
    assert not torch.any(actions == 1.0)
    torch.testing.assert_close(actions[requested_levels == 0.5], torch.zeros(1))


def _make_fixed_forward_policy(action_generation_mode: str):
    policy = _make_loss_mask_policy(mask_ratio=0.0)
    policy.config.action_generation_mode = action_generation_mode
    policy.config.max_action_dim = 1
    policy.config.output_features = {ACTION: SimpleNamespace(shape=(1,))}

    def preprocess_images(batch):
        del batch
        return [], []

    def prepare_action(batch):
        return batch[ACTION]

    policy._preprocess_images = preprocess_images
    policy.prepare_action = prepare_action
    return policy


def _regression_batch(actions: torch.Tensor, action_is_pad: torch.Tensor):
    return {
        ACTION: actions,
        f"{ACTION}_is_pad": action_is_pad,
        OBS_LANGUAGE_TOKENS: torch.zeros(actions.shape[0], 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(actions.shape[0], 1, dtype=torch.bool),
    }


def test_regression_loss_and_diagnostics_ignore_padded_action_values():
    policy = _make_fixed_forward_policy("regression")
    policy.config.marginal_quantile_num_knots = 3
    policy.config.marginal_quantile_initial_gap = 1.0
    policy.model = _FixedRegressionModel()
    policy.train()

    action_is_pad = torch.tensor([[False, False, True], [False, True, True]])
    finite_batch = _regression_batch(
        torch.tensor([[[0.2], [0.3], [0.4]], [[-0.2], [-0.3], [-0.4]]]),
        action_is_pad,
    )
    extreme_batch = _regression_batch(
        torch.tensor([[[0.2], [0.3], [1e20]], [[-0.2], [-1e20], [1e20]]]),
        action_is_pad,
    )

    finite_loss, finite_metrics = policy.forward(finite_batch)
    extreme_loss, extreme_metrics = policy.forward(extreme_batch)

    torch.testing.assert_close(finite_loss, extreme_loss)
    for key in (
        "loss_marginal_quantile",
        "marginal_quantile_tail_fraction",
        "marginal_quantile_outer_coverage",
        "loss_median_mse",
        "marginal_quantile_mean_width",
        "marginal_quantile_mean_log_gap",
        "marginal_quantile_min_log_gap",
    ):
        assert finite_metrics[key] == pytest.approx(extreme_metrics[key])
    assert not any("copula" in key for key in finite_metrics)
    assert "loss_quantile_gap_barrier" not in finite_metrics
    assert "quantile_gap_barrier_strength" not in finite_metrics


def test_regression_loss_is_only_the_marginal_pinball_loss():
    policy = _make_fixed_forward_policy("regression")
    policy.config.marginal_quantile_num_knots = 3
    policy.model = _FixedRegressionModel()
    policy.train()
    batch = _regression_batch(
        torch.tensor([[[0.2], [0.3], [0.4]], [[-0.2], [-0.3], [-0.4]]]),
        torch.zeros(2, 3, dtype=torch.bool),
    )

    per_sample_loss, metrics = policy.forward(batch, reduction="none")
    mean_loss, _ = policy.forward(batch, reduction="mean")

    torch.testing.assert_close(mean_loss, per_sample_loss.mean())
    assert metrics["loss"] == pytest.approx(metrics["loss_marginal_quantile"])


def test_flow_loss_reductions_weight_samples_equally_with_heterogeneous_padding():
    policy = _make_fixed_forward_policy("flow_matching")
    policy.model = _FixedFlowLossModel(torch.tensor([[[2.0], [2.0], [100.0]], [[4.0], [100.0], [100.0]]]))
    batch = _regression_batch(
        torch.zeros(2, 3, 1),
        torch.tensor([[False, False, True], [False, True, True]]),
    )

    mean_loss, _ = policy.forward(batch, reduction="mean")
    per_sample_loss, _ = policy.forward(batch, reduction="none")

    torch.testing.assert_close(per_sample_loss, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(mean_loss, per_sample_loss.mean())
