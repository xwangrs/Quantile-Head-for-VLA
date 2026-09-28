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
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

pytest.importorskip("transformers")

from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig  # noqa: E402
from lerobot.policies.quantile_head.modeling_quantile_head import QuantileHeadPolicy  # noqa: E402


class _TinyQuantileHeadConfig(QuantileHeadConfig):
    def _save_pretrained(self, save_directory):
        del save_directory

    def to_dict(self):
        return vars(self).copy()


class _TinyQuantileHeadPolicy(QuantileHeadPolicy):
    """Exercise quantile-head checkpoint compatibility without constructing PaliGemma."""

    def __init__(self, config, **kwargs):
        del kwargs
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2, 2, bias=False)
        self.model.action_out_proj = torch.nn.Linear(2, 1, bias=False)
        self.model.quantile_delta_proj = torch.nn.Linear(2, 2, bias=False)
        self.model.soft_prompt = (
            torch.nn.Parameter(torch.zeros(config.soft_prompt_len, 2)) if config.soft_prompt_len else None
        )

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        del model_config
        return state_dict


def _quantile_head_config(soft_prompt_len=0):
    return _TinyQuantileHeadConfig(
        device="cpu",
        action_generation_mode="regression",
        pretrained_path="base-checkpoint",
        use_peft=False,
        soft_prompt_len=soft_prompt_len,
    )


@pytest.mark.parametrize("soft_prompt_len", [0, 3])
def test_plain_checkpoint_roundtrip_preserves_quantile_head(tmp_path, monkeypatch, soft_prompt_len):
    source = _TinyQuantileHeadPolicy(_quantile_head_config(soft_prompt_len))
    with torch.no_grad():
        source.model.backbone.weight.fill_(2.0)
        source.model.action_out_proj.weight.fill_(3.0)
        source.model.quantile_delta_proj.weight.fill_(4.0)
        if source.model.soft_prompt is not None:
            source.model.soft_prompt.fill_(5.0)
    source._save_pretrained(tmp_path)

    model_file = tmp_path / SAFETENSORS_SINGLE_FILE
    monkeypatch.setattr("transformers.utils.cached_file", lambda *args, **kwargs: str(model_file))
    restored = _TinyQuantileHeadPolicy.from_pretrained(
        tmp_path,
        config=_quantile_head_config(soft_prompt_len),
    )

    source_state = source.state_dict()
    restored_state = restored.state_dict()
    assert restored_state.keys() == source_state.keys()
    for key in source_state:
        torch.testing.assert_close(restored_state[key], source_state[key])


def test_quantile_head_default_peft_has_no_full_modules_to_save():
    policy = _TinyQuantileHeadPolicy(_quantile_head_config())

    defaults = policy._get_default_peft_targets()

    assert defaults["modules_to_save"] == []


def test_peft_adapter_state_contains_quantile_adapter_without_copula():
    peft = pytest.importorskip("peft")
    policy = _TinyQuantileHeadPolicy(_quantile_head_config())

    peft_model = policy.wrap_with_peft()
    adapter_state = peft.get_peft_model_state_dict(peft_model)

    assert any("quantile_delta_proj" in key for key in adapter_state)
    assert not any("copula" in key for key in adapter_state)


@pytest.mark.parametrize("modules_to_save", [None, ["soft_prompt"]])
def test_reference_action_prompt_rejects_peft_before_freezing_parameters(modules_to_save):
    peft = pytest.importorskip("peft")
    policy = _TinyQuantileHeadPolicy(_quantile_head_config(soft_prompt_len=3))

    with pytest.raises(ValueError, match="require non-PEFT"):
        policy.wrap_with_peft(
            peft_config=peft.LoraConfig(target_modules=["action_out_proj"], modules_to_save=modules_to_save)
        )

    assert policy.model.soft_prompt.requires_grad
    assert not policy.config.use_peft


def test_reference_action_prompt_rejects_peft_loading():
    policy = _TinyQuantileHeadPolicy(_quantile_head_config(soft_prompt_len=3))
    with pytest.raises(ValueError, match="require non-PEFT"):
        policy._prepare_peft_config_for_loading(SimpleNamespace(modules_to_save=["soft_prompt"]))


def test_reference_action_prompt_factory_rejects_adapter_loading(monkeypatch):
    from lerobot.policies import factory
    from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig

    monkeypatch.setattr(factory, "get_policy_class", lambda name: _TinyQuantileHeadPolicy)
    monkeypatch.setattr(factory, "env_to_policy_features", lambda env: {})
    monkeypatch.setattr(factory, "validate_visual_features_consistency", lambda cfg, features: None)
    config = QuantileHeadConfig(
        device="cpu", pretrained_path="adapter-checkpoint", use_peft=True, soft_prompt_len=3
    )
    with pytest.raises(ValueError, match="require non-PEFT"):
        factory.make_policy(config, env_cfg=SimpleNamespace())
