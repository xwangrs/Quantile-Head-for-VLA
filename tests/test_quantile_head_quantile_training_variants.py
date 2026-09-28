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

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.quantile_head import modeling_quantile_head  # noqa: E402
from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig  # noqa: E402
from lerobot.policies.quantile_head.modeling_quantile_head import (  # noqa: E402
    QuantileHeadPolicy,
    QuantileHeadPytorch,
    build_monotonic_quantile_knots,
)
from lerobot.scripts.quantile_head_eval_decoders import configure_quantile_head_eval_decoder  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402

VARIANTS = ["standard", "median_only", "stop_median_gradient", "independent"]
from tests.quantile_head_test_utils import make_quantile_head_config_stub  # noqa: E402


@pytest.mark.parametrize("variant", VARIANTS)
def test_quantile_variant_config_round_trip_and_cli_override(tmp_path, variant):
    config = QuantileHeadConfig(device="cpu", marginal_quantile_variant=variant)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    assert restored.marginal_quantile_variant == variant

    overridden = PreTrainedConfig.from_pretrained(
        tmp_path,
        local_files_only=True,
        cli_overrides=[f"--marginal_quantile_variant={variant}"],
    )
    assert overridden.marginal_quantile_variant == variant


@pytest.mark.parametrize("variant", ["unknown", "", None, True])
def test_quantile_variant_rejects_unknown_names(variant):
    with pytest.raises(ValueError, match="marginal_quantile_variant"):
        QuantileHeadConfig(device="cpu", marginal_quantile_variant=variant)


@pytest.mark.parametrize("variant", VARIANTS[1:])
@pytest.mark.parametrize(
    "overrides", [{"use_pure_regression_head": True}, {"action_generation_mode": "flow_matching"}]
)
def test_quantile_variant_requires_quantile_regression(variant, overrides):
    with pytest.raises(ValueError, match="marginal_quantile_variant"):
        QuantileHeadConfig(device="cpu", marginal_quantile_variant=variant, **overrides)


@pytest.mark.parametrize("variant", ["median_only", "independent"])
def test_untrained_or_unordered_quantiles_cannot_enable_sampling(variant):
    with pytest.raises(ValueError, match="median"):
        QuantileHeadConfig(
            device="cpu",
            marginal_quantile_variant=variant,
            marginal_quantile_inference_mode="temporal_sample",
        )


@pytest.mark.parametrize("variant", ["median_only", "independent"])
@pytest.mark.parametrize(
    "decoder",
    ["independent_rank", "gaussian_temporal", "quantile_expectation", "fixed_width", "calibrate_width"],
)
def test_distribution_decoder_overrides_cannot_bypass_median_only_inference(variant, decoder):
    policy = torch.nn.Module()
    policy.config = make_quantile_head_config_stub(
        type="quantile_head", action_generation_mode="regression", marginal_quantile_variant=variant
    )

    with pytest.raises(ValueError, match="requires median inference"):
        configure_quantile_head_eval_decoder(policy, decoder)

    assert not hasattr(policy, "_decode_regression_quantiles")
    assert not hasattr(policy, "_quantile_head_eval_decoder_name")


@pytest.mark.parametrize("variant", ["median_only", "independent"])
def test_default_eval_decoder_stays_a_noop_for_median_only_variants(variant):
    policy = torch.nn.Module()
    policy.config = make_quantile_head_config_stub(
        type="quantile_head", action_generation_mode="regression", marginal_quantile_variant=variant
    )
    original_attributes = set(vars(policy))

    assert configure_quantile_head_eval_decoder(policy, "policy") is None
    assert set(vars(policy)) == original_attributes


@pytest.fixture
def tiny_head_model_factory(monkeypatch):
    """Exercise the real constructor without allocating a vision/language model."""

    class TinyBackbone(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(4, 4)

    monkeypatch.setattr(modeling_quantile_head, "PaliGemmaWithExpertModel", TinyBackbone)
    monkeypatch.setattr(modeling_quantile_head, "get_gemma_config", lambda variant: SimpleNamespace(width=4))

    def make_model(variant="standard", num_knots=5):
        return QuantileHeadPytorch(
            QuantileHeadConfig(
                device="cpu",
                dtype="float32",
                max_action_dim=3,
                chunk_size=3,
                n_action_steps=2,
                soft_prompt_len=2,
                marginal_quantile_num_knots=num_knots,
                marginal_quantile_variant=variant,
            )
        )

    return make_model


@pytest.mark.parametrize("variant", VARIANTS[1:])
@pytest.mark.parametrize("num_knots", [3, 7])
def test_variants_preserve_shared_initialization_rng_and_initial_predictions(
    tiny_head_model_factory, variant, num_knots
):
    torch.manual_seed(123)
    standard = tiny_head_model_factory(num_knots=num_knots)
    standard_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    ablation = tiny_head_model_factory(variant, num_knots)

    assert torch.equal(torch.random.get_rng_state(), standard_rng)
    standard_state = standard.state_dict()
    ablation_state = ablation.state_dict()
    extra_keys = set(ablation_state) - set(standard_state)
    expected_extra = {"quantile_delta_proj._independent_quantile_head"} if variant == "independent" else set()
    assert extra_keys == expected_extra
    assert set(standard_state).issubset(ablation_state)
    for name, value in standard_state.items():
        assert value.shape == ablation_state[name].shape
        if not name.startswith("quantile_delta_proj."):
            torch.testing.assert_close(value, ablation_state[name], rtol=0, atol=0)
    assert sum(p.numel() for p in standard.parameters()) == sum(p.numel() for p in ablation.parameters())

    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    torch.testing.assert_close(
        ablation._build_marginal_quantiles(features),
        standard._build_marginal_quantiles(features),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_outer_quantile_gradient_reaches_shared_features_with_expected_median_coupling(
    tiny_head_model_factory, variant
):
    model = tiny_head_model_factory(variant)
    # A trained gap head has nonzero feature weights; zero initialization would hide that path.
    with torch.no_grad():
        model.action_out_proj.weight.fill_(0.4)
        model.action_out_proj.bias.fill_(0.1)
        model.quantile_delta_proj.weight.fill_(0.2)
        model.quantile_delta_proj.bias.fill_(0.3)
    features = torch.ones(2, 3, 4, requires_grad=True)
    knots = model._build_marginal_quantiles(features)
    knots[..., -1].sum().backward()

    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    assert model.quantile_delta_proj.weight.grad.abs().sum() > 0
    median_grad = model.action_out_proj.weight.grad
    if variant in {"stop_median_gradient", "independent"}:
        assert median_grad is None or not median_grad.any()
    else:
        assert median_grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    model._build_marginal_quantiles(features)[..., 2].sum().backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0
    gap_grad = model.quantile_delta_proj.weight.grad
    assert gap_grad is None or not gap_grad.any()


def test_detached_median_changes_only_backward_not_quantile_values():
    median = torch.tensor([[[1.2]]], requires_grad=True)
    raw_gaps = torch.tensor([[[[-0.7, 0.4, 1.1, -0.3]]]], requires_grad=True)
    standard = build_monotonic_quantile_knots(median, raw_gaps, initial_gap=0.02)
    detached = build_monotonic_quantile_knots(median, raw_gaps, initial_gap=0.02, detach_median=True)
    torch.testing.assert_close(detached, standard, rtol=0, atol=0)

    # With an asymmetric target, the ordinary pinball loss moves the explicit median through
    # every quantile; the detached version must retain only the central pinball contribution.
    levels = torch.linspace(0.1, 0.9, 5)
    error = 3.0 - detached
    torch.maximum(levels * error, (levels - 1) * error).mean().backward()
    torch.testing.assert_close(median.grad, torch.full_like(median, -0.5 / 5))
    assert raw_gaps.grad.abs().sum() > 0


def test_independent_head_preserves_direct_unordered_values_and_explicit_median(tiny_head_model_factory):
    model = tiny_head_model_factory("independent")
    with torch.no_grad():
        model.action_out_proj.weight.zero_()
        model.action_out_proj.bias.fill_(3.0)
        model.quantile_delta_proj.weight.zero_()
        model.quantile_delta_proj.bias.copy_(torch.tensor([8.0, -2.0, 6.0, -4.0]).repeat(3))

    knots = model._build_marginal_quantiles(torch.zeros(1, 2, 4))
    expected = torch.tensor([8.0, -2.0, 3.0, 6.0, -4.0]).expand(1, 2, 3, 5)
    torch.testing.assert_close(knots, expected)
    torch.testing.assert_close(model.median_actions_from_quantiles(knots), torch.full((1, 2, 3), 3.0))


class _TrainableQuantileModel(QuantileHeadPytorch):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.prediction = torch.nn.Parameter(
            torch.tensor([[[1.0, -2.0, 1000.0], [3.0, -4.0, 1000.0], [5.0, -6.0, 1000.0]]])
        )
        self.raw_gaps = torch.nn.Parameter(torch.zeros(1, 3, 3, config.marginal_quantile_num_knots - 1))
        self.register_buffer(
            "quantile_levels", torch.linspace(0.1, 0.9, config.marginal_quantile_num_knots), persistent=False
        )

    def forward_regression(self, images, img_masks, tokens, masks, time, actions):
        assert actions.shape == (2, 3, 3)
        assert not time.any()
        return build_monotonic_quantile_knots(
            self.prediction.expand(actions.shape[0], -1, -1),
            self.raw_gaps.expand(actions.shape[0], -1, -1, -1),
            initial_gap=0.02,
        )


def _make_median_only_policy(num_knots=5, mask_ratio=0.0):
    policy = QuantileHeadPolicy.__new__(QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = QuantileHeadConfig(
        device="cpu",
        marginal_quantile_variant="median_only",
        marginal_quantile_num_knots=num_knots,
        max_action_dim=3,
        chunk_size=3,
        n_action_steps=2,
        action_loss_mask_ratio=mask_ratio,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    policy.model = _TrainableQuantileModel(policy.config)
    policy._preprocess_images = lambda batch: ([], [])
    return policy


def _padded_batch():
    return {
        ACTION: torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [float("nan"), float("inf")]], [[float("nan"), float("nan")]] * 3]
        ),
        f"{ACTION}_is_pad": torch.tensor([[False, False, True], [True, True, True]]),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }


@pytest.mark.parametrize("num_knots", [3, 7, 21])
def test_median_only_loss_is_half_l1_independent_of_knot_count_with_safe_padding(num_knots):
    policy = _make_median_only_policy(num_knots=num_knots)
    per_sample, metrics = policy(_padded_batch(), reduction="none")
    loss, _ = policy(_padded_batch())

    torch.testing.assert_close(per_sample, torch.tensor([1.25, 0.0]))
    torch.testing.assert_close(loss, per_sample.mean())
    assert metrics["loss_per_dim"] == pytest.approx([1.0, 1.5])
    assert metrics["loss_marginal_quantile"] == pytest.approx(0.625)
    assert metrics["marginal_quantile_supervised_knots"] == 1
    loss.backward()
    grad = policy.model.prediction.grad
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad[:, :2, :2], torch.tensor([[[0.0625, -0.0625], [0.0625, -0.0625]]]))
    assert not grad[:, 2].any()
    assert not grad[..., 2].any()
    # Keeping zero gradients, rather than unused parameters, permits DDP without changing its settings.
    assert policy.model.raw_gaps.grad is not None
    assert torch.isfinite(policy.model.raw_gaps.grad).all()
    assert not policy.model.raw_gaps.grad.any()


def test_median_only_training_mask_and_eval_reduction():
    policy = _make_median_only_policy(mask_ratio=0.5)
    batch = _padded_batch()
    torch.manual_seed(23)
    keep = policy.make_action_loss_keep_mask(batch[ACTION], batch[f"{ACTION}_is_pad"])
    expected_loss = policy.model.prediction[:, :, :2].expand(2, -1, -1)[keep].abs().mean() * 0.5
    torch.manual_seed(23)
    per_sample, _ = policy(batch, reduction="none")
    torch.testing.assert_close(per_sample, torch.stack([expected_loss, expected_loss.new_zeros(())]))
    per_sample.mean().backward()
    valid_grad = policy.model.prediction.grad[:, :, :2]
    assert torch.isfinite(valid_grad).all()
    assert not valid_grad[~keep[:1]].any()

    policy.eval()
    rng_state = torch.random.get_rng_state().clone()
    per_sample, _ = policy(batch, reduction="none")
    torch.testing.assert_close(per_sample, torch.tensor([1.25, 0.0]))
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_independent_training_keeps_all_pinball_terms_and_reports_crossings_without_padding():
    policy = _make_median_only_policy()
    policy.config.marginal_quantile_variant = "independent"
    knots = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0]).expand(2, 3, 3, 5).clone()
    knots[0, :2, :2] = torch.tensor([2.0, -1.0, 0.0, 3.0, 1.0])
    knots.requires_grad_()
    policy.model.forward_regression = lambda *args: knots

    per_sample, metrics = policy(_padded_batch(), reduction="none")

    torch.testing.assert_close(per_sample, torch.tensor([0.62, 0.0]))
    assert metrics["marginal_quantile_crossing_fraction"] == pytest.approx(0.5)
    assert metrics["marginal_quantile_supervised_knots"] == 5
    per_sample.sum().backward()
    assert torch.isfinite(knots.grad).all()
    assert not knots.grad[1].any()
    assert not knots.grad[:, 2].any()
    assert not knots.grad[..., 2, :].any()


class _TinyHeadPolicy(QuantileHeadPolicy):
    """Use real heads and checkpoint loading with the fixture's small backbone."""

    def __init__(self, config, **kwargs):
        del kwargs
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = QuantileHeadPytorch(config)


def _save_head_checkpoint(directory, policy, state=None):
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    policy.config.save_pretrained(directory)
    save_file(policy.state_dict() if state is None else state, directory / "model.safetensors")


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_checkpoint_restores_trained_heads_and_config(tmp_path, tiny_head_model_factory, variant):
    policy = _TinyHeadPolicy(tiny_head_model_factory(variant).config)
    with torch.no_grad():
        # A loaded independent head must not be silently reset to copies of the median head.
        policy.model.quantile_delta_proj.weight.fill_(1.5)
        policy.model.quantile_delta_proj.bias.fill_(-2.0)
    _save_head_checkpoint(tmp_path, policy)

    restored = _TinyHeadPolicy.from_pretrained(tmp_path, local_files_only=True)

    assert restored.config.marginal_quantile_variant == variant
    assert not restored.training
    for key, tensor in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)
    features = torch.ones(2, 3, 4)
    torch.testing.assert_close(
        restored.model._build_marginal_quantiles(features), policy.model._build_marginal_quantiles(features)
    )


@pytest.mark.parametrize(
    "source_variant,target_variant", [("standard", "independent"), ("independent", "standard")]
)
@pytest.mark.parametrize("allow_bootstrap", [False, True])
def test_independent_checkpoint_cannot_be_misread_as_gap_parameters(
    tmp_path, tiny_head_model_factory, source_variant, target_variant, allow_bootstrap
):
    policy = _TinyHeadPolicy(tiny_head_model_factory(source_variant).config)
    _save_head_checkpoint(tmp_path, policy)
    target_config = tiny_head_model_factory(target_variant).config

    with pytest.raises(RuntimeError, match="_independent_quantile_head"):
        _TinyHeadPolicy.from_pretrained(
            tmp_path,
            config=target_config,
            local_files_only=True,
            allow_missing_action_modules=allow_bootstrap,
        )


def test_bootstrap_initializes_independent_head_from_loaded_median_not_random_constructor(
    tmp_path, tiny_head_model_factory
):
    policy = _TinyHeadPolicy(tiny_head_model_factory().config)
    with torch.no_grad():
        policy.model.action_out_proj.weight.fill_(0.7)
        policy.model.action_out_proj.bias.fill_(1.3)
    base_state = {
        name: tensor for name, tensor in policy.state_dict().items() if "quantile_delta_proj" not in name
    }
    _save_head_checkpoint(tmp_path, policy, base_state)
    standard_config = tiny_head_model_factory().config
    independent_config = tiny_head_model_factory("independent").config

    torch.manual_seed(42)
    standard = _TinyHeadPolicy.from_pretrained(
        tmp_path, config=standard_config, local_files_only=True, allow_missing_action_modules=True
    )
    rng_state = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    independent = _TinyHeadPolicy.from_pretrained(
        tmp_path, config=independent_config, local_files_only=True, allow_missing_action_modules=True
    )

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.testing.assert_close(independent.model.action_out_proj.weight, policy.model.action_out_proj.weight)
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    torch.testing.assert_close(
        independent.model._build_marginal_quantiles(features),
        standard.model._build_marginal_quantiles(features),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_compiled_checkpointed_quantile_objective_preserves_eager_gradients(tiny_head_model_factory, variant):
    reference = _TinyHeadPolicy(tiny_head_model_factory(variant).config)
    with torch.no_grad():
        # Nonzero gap weights exercise auxiliary gradients into the shared features after initialization.
        reference.model.quantile_delta_proj.weight.add_(0.12)
    checkpointed = _TinyHeadPolicy(reference.config)
    checkpointed.load_state_dict(reference.state_dict(), strict=True)
    # Only the head is evaluated here, so enable its real checkpoint path without accessing
    # the mocked backbone's nonexistent vision/language checkpointing flags.
    checkpointed.model.gradient_checkpointing_enabled = True
    reference.train()
    checkpointed.train()

    features = torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4).requires_grad_()
    checkpointed_features = features.detach().clone().requires_grad_()
    targets = torch.linspace(-0.75, 0.75, 18).reshape(2, 3, 3)
    keep_mask = torch.arange(18).reshape(2, 3, 3).remainder(4) != 0
    targets = targets.masked_fill(~keep_mask, float("nan"))

    def objective(policy, shared_features, actions, mask):
        knots = policy.model._build_marginal_quantiles(shared_features)
        per_sample, masked_terms = policy.compute_marginal_quantile_loss(actions, knots, mask)
        return knots, per_sample, masked_terms

    def checkpointed_objective(shared_features, actions, mask):
        return objective(checkpointed, shared_features, actions, mask)

    compiled_objective = torch.compile(checkpointed_objective, backend="eager", fullgraph=True)
    expected = objective(reference, features, targets, keep_mask)
    actual = compiled_objective(checkpointed_features, targets, keep_mask)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    expected[1].mean().backward()
    actual[1].mean().backward()

    assert torch.isfinite(checkpointed_features.grad).all()
    assert checkpointed_features.grad.abs().sum() > 0
    torch.testing.assert_close(checkpointed_features.grad, features.grad)
    for name, parameter in reference.named_parameters():
        actual_grad = dict(checkpointed.named_parameters())[name].grad
        if parameter.grad is None:
            assert actual_grad is None
        else:
            assert actual_grad is not None
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, parameter.grad)

    assert checkpointed.model.action_out_proj.weight.grad.abs().sum() > 0
    for parameter in checkpointed.model.quantile_delta_proj.parameters():
        assert parameter.grad is not None
        if variant == "median_only":
            assert not parameter.grad.any()
        else:
            assert parameter.grad.abs().sum() > 0
