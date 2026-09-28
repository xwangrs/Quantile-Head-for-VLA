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

from lerobot.policies.quantile_head.modeling_quantile_head import QuantileHeadPolicy  # noqa: E402
from tests.quantile_head_test_utils import make_quantile_head_config_stub  # noqa: E402


class _TinyQuantileHeadPolicy(QuantileHeadPolicy):
    """Exercise QuantileHead checkpoint loading without constructing PaliGemma."""

    def __init__(self, config, **kwargs):
        del kwargs
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2, 2, bias=False)
        self.model.action_out_proj = torch.nn.Linear(2, 1, bias=False)
        self.model.quantile_delta_proj = None if config.use_pure_regression_head else torch.nn.Linear(2, 2)
        self.model.soft_prompt = (
            torch.nn.Parameter(torch.zeros(config.soft_prompt_len, 2)) if config.soft_prompt_len else None
        )

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        del model_config
        return state_dict


def _config(soft_prompt_len=0, use_pure_regression_head=False):
    return make_quantile_head_config_stub(
        action_generation_mode="regression",
        soft_prompt_len=soft_prompt_len,
        use_pure_regression_head=use_pure_regression_head,
    )


def _base_state_dict():
    return {
        "model.backbone.weight": torch.full((2, 2), 2.0),
        "model.action_out_proj.weight": torch.full((1, 2), 3.0),
    }


def _full_state_dict():
    return {
        **_base_state_dict(),
        "model.quantile_delta_proj.weight": torch.full((2, 2), 4.0),
        "model.quantile_delta_proj.bias": torch.full((2,), 4.5),
    }


def _legacy_copula_state_dict():
    return {
        **_full_state_dict(),
        "model.copula.factor_raw": torch.full((1,), 5.0),
        "model.copula.diagonal_raw": torch.full((1,), 6.0),
    }


def _mock_checkpoint(monkeypatch, state_dict):
    monkeypatch.setattr("transformers.utils.cached_file", lambda *args, **kwargs: "model.safetensors")
    monkeypatch.setattr("safetensors.torch.load_file", lambda path: state_dict)


def test_quantile_head_checkpoint_load_failure_is_fatal(monkeypatch):
    def fail_to_resolve(*args, **kwargs):
        raise FileNotFoundError("missing weights")

    monkeypatch.setattr("transformers.utils.cached_file", fail_to_resolve)

    with pytest.raises(RuntimeError, match="Failed to load QuantileHead checkpoint"):
        _TinyQuantileHeadPolicy.from_pretrained("missing-checkpoint", config=_config())


def test_quantile_head_checkpoint_resolver_options_are_forwarded(monkeypatch):
    observed = {}

    def resolve_checkpoint(path, filename, **kwargs):
        observed.update(path=path, filename=filename, **kwargs)
        return "model.safetensors"

    monkeypatch.setattr("transformers.utils.cached_file", resolve_checkpoint)
    monkeypatch.setattr("safetensors.torch.load_file", lambda path: _full_state_dict())

    _TinyQuantileHeadPolicy.from_pretrained(
        "trained-checkpoint",
        config=_config(),
        cache_dir="cache-dir",
        force_download=True,
        resume_download=True,
        proxies={"https": "proxy"},
        token="test-token",
        revision="test-revision",
        local_files_only=True,
    )

    assert observed == {
        "path": "trained-checkpoint",
        "filename": "model.safetensors",
        "cache_dir": "cache-dir",
        "force_download": True,
        "resume_download": True,
        "proxies": {"https": "proxy"},
        "token": "test-token",
        "revision": "test-revision",
        "local_files_only": True,
    }


def test_quantile_head_eval_requires_all_action_modules(monkeypatch):
    _mock_checkpoint(monkeypatch, _base_state_dict())

    with pytest.raises(RuntimeError, match="quantile_delta_proj"):
        _TinyQuantileHeadPolicy.from_pretrained("base-checkpoint", config=_config())


def test_quantile_head_training_bootstrap_allows_whole_missing_action_module_groups(monkeypatch):
    _mock_checkpoint(monkeypatch, _base_state_dict())

    policy = _TinyQuantileHeadPolicy.from_pretrained(
        "base-checkpoint",
        config=_config(),
        allow_missing_action_modules=True,
    )

    torch.testing.assert_close(policy.model.backbone.weight, torch.full((2, 2), 2.0))
    torch.testing.assert_close(policy.model.action_out_proj.weight, torch.full((1, 2), 3.0))


def test_quantile_head_training_bootstrap_rejects_partial_quantile_head(monkeypatch):
    partial_state_dict = {
        **_base_state_dict(),
        "model.quantile_delta_proj.weight": torch.full((2, 2), 4.0),
    }
    _mock_checkpoint(monkeypatch, partial_state_dict)

    with pytest.raises(RuntimeError, match="quantile_delta_proj.bias"):
        _TinyQuantileHeadPolicy.from_pretrained(
            "damaged-checkpoint",
            config=_config(),
            allow_missing_action_modules=True,
        )


def test_quantile_head_eval_rejects_unexpected_checkpoint_keys(monkeypatch):
    _mock_checkpoint(monkeypatch, {**_full_state_dict(), "model.unknown.weight": torch.ones(1)})

    with pytest.raises(RuntimeError, match="Unexpected key"):
        _TinyQuantileHeadPolicy.from_pretrained("damaged-checkpoint", config=_config())


def test_quantile_head_eval_strictly_loads_complete_checkpoint(monkeypatch):
    _mock_checkpoint(monkeypatch, _full_state_dict())

    policy = _TinyQuantileHeadPolicy.from_pretrained("trained-checkpoint", config=_config())

    torch.testing.assert_close(policy.model.quantile_delta_proj.weight, torch.full((2, 2), 4.0))
    torch.testing.assert_close(policy.model.quantile_delta_proj.bias, torch.full((2,), 4.5))
    assert not policy.training


def test_quantile_head_strict_load_rejects_removed_copula_checkpoint_keys(monkeypatch):
    _mock_checkpoint(monkeypatch, _legacy_copula_state_dict())

    with pytest.raises(RuntimeError, match="Unexpected key"):
        _TinyQuantileHeadPolicy.from_pretrained("legacy-checkpoint", config=_config())


@pytest.mark.parametrize("use_pure_regression_head", [False, True])
@pytest.mark.parametrize("bootstrap_flag", ["allow_missing_soft_prompt", "allow_missing_action_modules"])
def test_quantile_head_soft_prompt_bootstrap_requires_explicit_flag(
    monkeypatch, use_pure_regression_head, bootstrap_flag
):
    _mock_checkpoint(monkeypatch, _base_state_dict() if use_pure_regression_head else _full_state_dict())
    config = _config(soft_prompt_len=3, use_pure_regression_head=use_pure_regression_head)

    with pytest.raises(RuntimeError, match="model.soft_prompt"):
        _TinyQuantileHeadPolicy.from_pretrained("base-checkpoint", config=config)

    policy = _TinyQuantileHeadPolicy.from_pretrained(
        "base-checkpoint",
        config=config,
        **{bootstrap_flag: True},
    )

    assert policy.model.soft_prompt.shape == (3, 2)
    assert policy.model.soft_prompt.requires_grad
    torch.testing.assert_close(policy.model.backbone.weight, torch.full((2, 2), 2.0))


def test_quantile_head_soft_prompt_bootstrap_keeps_unrelated_keys_strict(monkeypatch):
    state_dict = _full_state_dict()
    del state_dict["model.backbone.weight"]
    _mock_checkpoint(monkeypatch, state_dict)

    with pytest.raises(RuntimeError, match="model.backbone.weight"):
        _TinyQuantileHeadPolicy.from_pretrained(
            "damaged-checkpoint",
            config=_config(soft_prompt_len=3),
            allow_missing_soft_prompt=True,
        )


def test_quantile_head_strictly_loads_trained_soft_prompt(monkeypatch):
    prompt_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    _mock_checkpoint(monkeypatch, {**_full_state_dict(), "model.soft_prompt": prompt_weight})

    policy = _TinyQuantileHeadPolicy.from_pretrained("trained-checkpoint", config=_config(soft_prompt_len=3))

    torch.testing.assert_close(policy.model.soft_prompt, prompt_weight)
    assert not policy.training


@pytest.mark.parametrize("soft_prompt_len", [0, 3])
def test_quantile_head_strict_load_rejects_removed_vlm_prompt_weights(monkeypatch, soft_prompt_len):
    _mock_checkpoint(monkeypatch, {**_full_state_dict(), "model.soft_prompt.weight": torch.ones(3, 4)})

    with pytest.raises(RuntimeError, match="Unexpected key"):
        _TinyQuantileHeadPolicy.from_pretrained(
            "vlm-prompt-checkpoint",
            config=_config(soft_prompt_len=soft_prompt_len),
            allow_missing_action_modules=True,
            allow_missing_soft_prompt=True,
        )


@pytest.mark.parametrize("allow_missing_soft_prompt", [False, True])
def test_quantile_head_rejects_incompatible_saved_soft_prompt_shape(monkeypatch, allow_missing_soft_prompt):
    _mock_checkpoint(monkeypatch, {**_full_state_dict(), "model.soft_prompt": torch.ones(2, 2)})

    with pytest.raises(RuntimeError, match="size mismatch for model.soft_prompt"):
        _TinyQuantileHeadPolicy.from_pretrained(
            "trained-checkpoint",
            config=_config(soft_prompt_len=3),
            allow_missing_soft_prompt=allow_missing_soft_prompt,
        )


@pytest.mark.parametrize("fresh_training", [False, True])
@pytest.mark.parametrize("use_pure_regression_head", [False, True])
def test_quantile_head_factory_bootstraps_soft_prompt_only_for_fresh_training(
    monkeypatch, fresh_training, use_pure_regression_head
):
    from lerobot.policies import factory
    from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig

    _mock_checkpoint(monkeypatch, _base_state_dict() if use_pure_regression_head else _full_state_dict())
    monkeypatch.setattr(factory, "get_policy_class", lambda name: _TinyQuantileHeadPolicy)
    monkeypatch.setattr(factory, "env_to_policy_features", lambda env: {})
    monkeypatch.setattr(factory, "validate_visual_features_consistency", lambda cfg, features: None)
    config = QuantileHeadConfig(
        device="cpu",
        pretrained_path="base-checkpoint",
        soft_prompt_len=3,
        action_generation_mode="regression",
        use_pure_regression_head=use_pure_regression_head,
    )
    kwargs = {
        "cfg": config,
        "env_cfg": SimpleNamespace(),
        "allow_missing_pretrained_action_modules": fresh_training,
    }

    if fresh_training:
        policy = factory.make_policy(**kwargs)
        assert policy.model.soft_prompt.shape == (3, 2)
    else:
        with pytest.raises(RuntimeError, match="model.soft_prompt"):
            factory.make_policy(**kwargs)


@pytest.mark.parametrize("soft_prompt_len", [0, 3])
def test_quantile_head_strictly_loads_pure_regression_checkpoint(monkeypatch, soft_prompt_len):
    state_dict = _base_state_dict()
    if soft_prompt_len:
        state_dict["model.soft_prompt"] = torch.full((soft_prompt_len, 2), 0.125)
    _mock_checkpoint(monkeypatch, state_dict)

    policy = _TinyQuantileHeadPolicy.from_pretrained(
        "pure-regression-checkpoint",
        config=_config(soft_prompt_len=soft_prompt_len, use_pure_regression_head=True),
    )

    assert policy.model.quantile_delta_proj is None
    assert not any("quantile_delta_proj" in key for key in policy.state_dict())
    torch.testing.assert_close(
        policy.model.action_out_proj.weight, state_dict["model.action_out_proj.weight"]
    )
    if soft_prompt_len:
        torch.testing.assert_close(policy.model.soft_prompt, state_dict["model.soft_prompt"])
    assert not policy.training


@pytest.mark.parametrize("allow_missing_action_modules", [False, True])
def test_quantile_head_pure_regression_rejects_quantile_checkpoint_even_during_bootstrap(
    monkeypatch, allow_missing_action_modules
):
    _mock_checkpoint(monkeypatch, _full_state_dict())

    with pytest.raises(RuntimeError, match="Unexpected key.*quantile_delta_proj"):
        _TinyQuantileHeadPolicy.from_pretrained(
            "quantile-trained-checkpoint",
            config=_config(use_pure_regression_head=True),
            allow_missing_action_modules=allow_missing_action_modules,
        )


def test_quantile_head_pure_regression_bootstrap_keeps_action_projection_strict(monkeypatch):
    state_dict = _base_state_dict()
    del state_dict["model.action_out_proj.weight"]
    _mock_checkpoint(monkeypatch, state_dict)

    with pytest.raises(RuntimeError, match="model.action_out_proj.weight"):
        _TinyQuantileHeadPolicy.from_pretrained(
            "damaged-pure-checkpoint",
            config=_config(use_pure_regression_head=True),
            allow_missing_action_modules=True,
        )
