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

from types import MethodType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.policies.quantile_head.modeling_quantile_head import QuantileHeadPytorch  # noqa: E402
from tests.quantile_head_test_utils import make_quantile_head_config_stub  # noqa: E402


class _TinyVLMWithExpert(torch.nn.Module):
    def __init__(self, *, train_vlm: bool):
        super().__init__()
        self.paligemma = torch.nn.Module()
        self.paligemma.model = torch.nn.Module()
        self.paligemma.model.language_model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            layer.self_attn.q_proj.weight.copy_(torch.eye(4))
        self.paligemma.model.language_model.layers = torch.nn.ModuleList([layer])
        self.paligemma.requires_grad_(train_vlm)
        self.expert_scale = torch.nn.Parameter(torch.ones(()))

    def forward(
        self,
        *,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        adarms_cond,
        detach_prefix_from_suffix,
    ):
        del attention_mask, position_ids, past_key_values, use_cache, adarms_cond
        prefix_embs, suffix_embs = inputs_embeds
        vlm_weight = self.paligemma.model.language_model.layers[0].self_attn.q_proj.weight
        if detach_prefix_from_suffix:
            with torch.no_grad():
                vlm_output = torch.nn.functional.linear(prefix_embs.detach(), vlm_weight)
        else:
            vlm_output = torch.nn.functional.linear(prefix_embs, vlm_weight)
        action_context = vlm_output.mean(dim=1, keepdim=True)
        suffix_output = suffix_embs * self.expert_scale + action_context
        return [prefix_embs, suffix_output], None


def _make_context_model(*, detach_action_from_vlm: bool = False, train_expert_only: bool = True):
    model = QuantileHeadPytorch.__new__(QuantileHeadPytorch)
    torch.nn.Module.__init__(model)
    model.config = make_quantile_head_config_stub(
        detach_action_from_vlm=detach_action_from_vlm,
        train_expert_only=train_expert_only,
        chunk_size=1,
    )
    model.gradient_checkpointing_enabled = False
    model.soft_prompt = None
    model.paligemma_with_expert = _TinyVLMWithExpert(train_vlm=not train_expert_only)

    vlm_input_weight = torch.nn.Parameter(torch.ones(1, 2, 4))

    def embed_prefix(self, images, img_masks, tokens, masks):
        del self, images, img_masks, tokens, masks
        prefix_embs = vlm_input_weight * 2.0
        prefix_pad_masks = torch.ones(1, 2, dtype=torch.bool)
        prefix_att_masks = torch.zeros(1, 2)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def embed_suffix(self, action_queries, time):
        del self, time
        suffix_embs = torch.ones_like(action_queries)
        suffix_pad_masks = torch.ones(action_queries.shape[:2], dtype=torch.bool)
        suffix_att_masks = torch.ones(action_queries.shape[:2])
        adarms_cond = torch.zeros(action_queries.shape[0], action_queries.shape[-1])
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    model.embed_prefix = MethodType(embed_prefix, model)
    model.embed_suffix = MethodType(embed_suffix, model)
    return model, vlm_input_weight


def _action_loss(model, prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix):
    action_queries = torch.zeros(1, model.config.chunk_size, 4)
    time = torch.zeros(1)
    suffix_output = model._predict_action_hidden_from_context(
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        detach_prefix_from_suffix,
        action_queries,
        time,
    )
    return suffix_output.square().mean()


def test_train_expert_only_detaches_vlm_from_action_loss():
    model, vlm_input_weight = _make_context_model()

    prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix = (
        model._prepare_action_context(
            images=None,
            img_masks=None,
            tokens=None,
            masks=None,
        )
    )
    loss = _action_loss(
        model,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        detach_prefix_from_suffix,
    )
    loss.backward()

    assert detach_prefix_from_suffix
    assert vlm_input_weight.grad is None
    frozen_q_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
    assert not frozen_q_proj.weight.requires_grad
    assert frozen_q_proj.weight.grad is None
    assert model.paligemma_with_expert.expert_scale.grad is not None
    assert model.paligemma_with_expert.expert_scale.grad.abs().item() > 0


def test_full_finetuning_backpropagates_action_loss_into_vlm():
    model, vlm_input_weight = _make_context_model(train_expert_only=False)

    prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix = (
        model._prepare_action_context(
            images=None,
            img_masks=None,
            tokens=None,
            masks=None,
        )
    )
    loss = _action_loss(
        model,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        detach_prefix_from_suffix,
    )
    loss.backward()

    assert not detach_prefix_from_suffix
    assert vlm_input_weight.grad is not None
    assert vlm_input_weight.grad.abs().sum().item() > 0
    vlm_q_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
    assert vlm_q_proj.weight.requires_grad
    assert vlm_q_proj.weight.grad is not None
    assert vlm_q_proj.weight.grad.abs().sum().item() > 0
    assert model.paligemma_with_expert.expert_scale.grad is not None
    assert model.paligemma_with_expert.expert_scale.grad.abs().item() > 0


def test_explicit_detach_action_from_vlm_remains_a_hard_stop():
    model, vlm_input_weight = _make_context_model(
        detach_action_from_vlm=True,
        train_expert_only=False,
    )

    prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix = (
        model._prepare_action_context(
            images=None,
            img_masks=None,
            tokens=None,
            masks=None,
        )
    )
    loss = _action_loss(
        model,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        detach_prefix_from_suffix,
    )
    loss.backward()

    assert detach_prefix_from_suffix
    assert vlm_input_weight.grad is None
    vlm_q_proj = model.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj
    assert vlm_q_proj.weight.requires_grad
    assert vlm_q_proj.weight.grad is None
    assert model.paligemma_with_expert.expert_scale.grad is not None
    assert model.paligemma_with_expert.expert_scale.grad.abs().item() > 0
