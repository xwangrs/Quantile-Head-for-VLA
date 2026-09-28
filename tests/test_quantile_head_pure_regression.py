#!/usr/bin/env python

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

from collections import deque

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig  # noqa: E402
from lerobot.policies.quantile_head.modeling_quantile_head import (  # noqa: E402
    QuantileHeadPolicy,
    QuantileHeadPytorch,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


@pytest.mark.parametrize("loss_name", ["l1", "l2"])
def test_pure_regression_config_survives_round_trip_and_cli_override(tmp_path, loss_name):
    config = QuantileHeadConfig(device="cpu", use_pure_regression_head=True, pure_regression_loss=loss_name)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    assert restored.use_pure_regression_head
    assert restored.pure_regression_loss == loss_name

    overridden = PreTrainedConfig.from_pretrained(
        tmp_path,
        local_files_only=True,
        cli_overrides=["--use_pure_regression_head=true", f"--pure_regression_loss={loss_name}"],
    )
    assert overridden.use_pure_regression_head
    assert overridden.pure_regression_loss == loss_name


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"use_pure_regression_head": "true"}, "boolean"),
        ({"use_pure_regression_head": 1}, "boolean"),
        ({"use_pure_regression_head": True, "action_generation_mode": "flow_matching"}, "requires"),
        ({"pure_regression_loss": "huber"}, "pure_regression_loss"),
    ],
)
def test_pure_regression_config_rejects_invalid_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        QuantileHeadConfig(device="cpu", **overrides)


class _PointActionModel(QuantileHeadPytorch):
    """Exercise real zero-query entry points with a small, trainable action prediction."""

    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.prediction = torch.nn.Parameter(
            torch.tensor([[[1.0, -2.0, 1000.0], [3.0, -4.0, 1000.0], [5.0, -6.0, 1000.0]]])
        )
        self.calls = 0

    def _predict_action_values(self, images, img_masks, tokens, masks, action_queries, time):
        assert action_queries.shape == (tokens.shape[0], 3, 3)
        assert torch.equal(action_queries, torch.zeros_like(action_queries))
        assert torch.equal(time, torch.zeros_like(time))
        self.calls += 1
        return self.prediction.expand(tokens.shape[0], -1, -1)

    def _build_marginal_quantiles(self, *args):
        raise AssertionError("Pure regression must not construct quantiles")

    def sample_actions(self, *args, **kwargs):
        raise AssertionError("Pure regression must not run flow sampling")


def _make_policy(loss_name="l1", mask_ratio=0.0):
    policy = QuantileHeadPolicy.__new__(QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = QuantileHeadConfig(
        device="cpu",
        use_pure_regression_head=True,
        pure_regression_loss=loss_name,
        action_loss_mask_ratio=mask_ratio,
        max_action_dim=3,
        chunk_size=3,
        n_action_steps=2,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        # Old sampler settings must not affect point-regression inference.
        marginal_quantile_inference_mode="temporal_sample",
        marginal_quantile_sampling_temperature=100.0,
    )
    policy.model = _PointActionModel(policy.config)
    policy._preprocess_images = lambda batch: ([], [])
    policy._action_queue = deque()
    return policy


def _batch():
    return {
        ACTION: torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [float("nan"), float("inf")]], [[float("nan"), float("nan")]] * 3]
        ),
        f"{ACTION}_is_pad": torch.tensor([[False, False, True], [True, True, True]]),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    ("loss_name", "expected", "per_dim"), [("l1", 2.5, [2.0, 3.0]), ("l2", 7.5, [5.0, 10.0])]
)
def test_point_loss_reduction_padding_and_gradients(loss_name, expected, per_dim):
    policy = _make_policy(loss_name)
    per_sample, metrics = policy(_batch(), reduction="none")
    loss, _ = policy(_batch())

    torch.testing.assert_close(per_sample, torch.tensor([expected, 0.0]))
    torch.testing.assert_close(loss, per_sample.mean())
    assert metrics["loss_per_dim"] == pytest.approx(per_dim)
    assert metrics[f"loss_pure_regression_{loss_name}"] == pytest.approx(expected / 2)
    assert not any("quantile" in key for key in metrics)
    loss.backward()
    grad = policy.model.prediction.grad
    assert torch.isfinite(grad).all()
    assert grad[:, :2, :2].abs().sum() > 0
    assert not grad[:, 2].any()  # Padded horizon.
    assert not grad[..., 2].any()  # Padded action dimension.


def test_point_loss_applies_training_mask_and_disables_random_mask_in_eval():
    policy = _make_policy(mask_ratio=0.5)
    batch = _batch()
    torch.manual_seed(23)
    keep = policy.make_action_loss_keep_mask(batch[ACTION], batch[f"{ACTION}_is_pad"])
    selected_errors = policy.model.prediction[:, :, :2].expand(2, -1, -1).abs()[keep]
    torch.manual_seed(23)
    losses, metrics = policy(batch, reduction="none")
    assert keep.sum().item() == 2
    torch.testing.assert_close(losses, torch.stack([selected_errors.mean(), losses.new_zeros(())]))
    assert metrics["action_loss_masked_fraction"] == pytest.approx(10 / 12)

    policy.eval()
    rng_state = torch.random.get_rng_state().clone()
    losses, _ = policy(batch, reduction="none")
    torch.testing.assert_close(losses, torch.tensor([2.5, 0.0]))
    assert torch.equal(rng_state, torch.random.get_rng_state())


def test_point_inference_is_one_pass_deterministic_and_preserves_action_queue():
    policy = _make_policy()
    batch = _batch()
    rng_state = torch.random.get_rng_state().clone()
    actions = policy.predict_action_chunk(batch)
    repeated = policy.predict_action_chunk(batch)
    assert policy.model.calls == 2
    assert actions.shape == (2, 3, 2)
    torch.testing.assert_close(actions, repeated)
    torch.testing.assert_close(actions, policy.model.prediction[:, :, :2].expand(2, -1, -1))
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert not actions.requires_grad
    for step in range(3):
        selected = policy.select_action(batch)
        torch.testing.assert_close(selected, actions[:, step % 2])
    assert policy.model.calls == 4  # Two queue refills, each executing two of three predicted steps.
