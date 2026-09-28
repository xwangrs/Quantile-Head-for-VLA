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

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.quantile_head import modeling_quantile_head  # noqa: E402
from lerobot.policies.quantile_head.configuration_quantile_head import QuantileHeadConfig  # noqa: E402


@pytest.fixture
def tiny_model_factory(monkeypatch):
    """Keep real vision/attention, including different VLM and action-expert widths."""
    paligemma_class = modeling_quantile_head.PaliGemmaForConditionalGenerationWithPiGemma
    expert_class = modeling_quantile_head.PiGemmaForCausalLM

    def tiny_paligemma(config):
        config.text_config.vocab_size = 32
        config.image_token_index = 31
        config.projection_dim = 16
        config.vision_config.hidden_size = 16
        config.vision_config.intermediate_size = 32
        config.vision_config.projection_dim = 16
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        config.vision_config.patch_size = 2
        return paligemma_class(config=config)

    def tiny_expert(config):
        config.vocab_size = 32
        return expert_class(config=config)

    monkeypatch.setattr(
        modeling_quantile_head, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_paligemma
    )
    monkeypatch.setattr(modeling_quantile_head, "PiGemmaForCausalLM", tiny_expert)
    monkeypatch.setattr(
        modeling_quantile_head,
        "get_gemma_config",
        lambda variant: modeling_quantile_head.GemmaConfig(
            width=16 if variant == "gemma_2b" else 8,
            depth=2,
            mlp_dim=32,
            num_heads=8,
            num_kv_heads=1,
            head_dim=2,
        ),
    )

    def make_model(**kwargs):
        settings = {
            "device": "cpu",
            "dtype": "float32",
            "image_resolution": (4, 4),
            "chunk_size": 2,
            "n_action_steps": 2,
            "max_action_dim": 3,
            "marginal_quantile_num_knots": 3,
            "soft_prompt_len": 3,
            "train_expert_only": True,
        }
        settings.update(kwargs)
        return modeling_quantile_head.QuantileHeadPytorch(QuantileHeadConfig(**settings))

    return make_model


def _prefix_inputs(*, with_images=True):
    images = [torch.randn(2, 3, 4, 4)] if with_images else []
    image_masks = [torch.tensor([True, False])] if with_images else []
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    token_masks = torch.tensor([[True, True, True], [True, True, False]])
    return images, image_masks, tokens, token_masks


def test_soft_prompt_rejects_negative_token_counts():
    with pytest.raises(ValueError, match="soft_prompt_len"):
        QuantileHeadConfig(device="cpu", soft_prompt_len=-1)


@pytest.mark.parametrize(
    "use_pure_regression_head,pure_regression_loss", [(False, "l1"), (True, "l1"), (True, "l2")]
)
def test_soft_prompt_config_round_trip(tmp_path, use_pure_regression_head, pure_regression_loss):
    config = QuantileHeadConfig(
        device="cpu",
        soft_prompt_len=7,
        train_soft_prompt=False,
        soft_prompt_init_std=0.03,
        optimizer_soft_prompt_lr=1e-4,
        optimizer_soft_prompt_weight_decay=0.02,
        train_expert_only=True,
        detach_action_from_vlm=True,
        use_pure_regression_head=use_pure_regression_head,
        pure_regression_loss=pure_regression_loss,
    )
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)

    assert restored.soft_prompt_len == 7
    assert not restored.train_soft_prompt
    assert restored.soft_prompt_init_std == 0.03
    assert restored.optimizer_soft_prompt_lr == 1e-4
    assert restored.optimizer_soft_prompt_weight_decay == 0.02
    assert restored.train_expert_only
    assert restored.detach_action_from_vlm
    assert restored.use_pure_regression_head == use_pure_regression_head
    assert restored.pure_regression_loss == pure_regression_loss
    defaults = QuantileHeadConfig(device="cpu")
    assert defaults.soft_prompt_len == 16
    assert defaults.train_soft_prompt
    assert defaults.soft_prompt_init_std == 0.02
    assert defaults.optimizer_soft_prompt_lr == 1e-4
    assert defaults.optimizer_soft_prompt_weight_decay == 0.0


@pytest.mark.parametrize("soft_prompt_len", [0, 3])
@pytest.mark.parametrize("with_images", [False, True])
def test_prompt_is_expert_parameter_and_never_inserted_in_vlm_prefix(
    tiny_model_factory, soft_prompt_len, with_images
):
    model = tiny_model_factory(soft_prompt_len=soft_prompt_len)
    images, image_masks, tokens, token_masks = _prefix_inputs(with_images=with_images)
    embeddings, pad_masks, attention_masks = model.embed_prefix(images, image_masks, tokens, token_masks)

    expected_embeddings = []
    expected_masks = []
    if with_images:
        image_embeddings = model.paligemma_with_expert.embed_image(images[0])
        expected_embeddings.append(image_embeddings)
        expected_masks.append(image_masks[0][:, None].expand(2, image_embeddings.shape[1]))
    expected_embeddings.append(model.paligemma_with_expert.embed_language_tokens(tokens))
    expected_masks.append(token_masks)
    torch.testing.assert_close(embeddings, torch.cat(expected_embeddings, dim=1))
    torch.testing.assert_close(pad_masks, torch.cat(expected_masks, dim=1))
    assert embeddings.shape == (2, (4 if with_images else 0) + tokens.shape[1], 16)
    assert attention_masks.shape == pad_masks.shape
    assert not attention_masks.any()

    if soft_prompt_len:
        assert isinstance(model.soft_prompt, torch.nn.Parameter)
        assert model.soft_prompt.requires_grad
        assert model.soft_prompt.shape == (soft_prompt_len, 8)
        assert model.soft_prompt.dtype == torch.float32
        assert "soft_prompt" in model.state_dict()
        assert "soft_prompt.weight" not in model.state_dict()
    else:
        assert model.soft_prompt is None
        assert not any("soft_prompt" in key for key in model.state_dict())


@pytest.mark.parametrize("soft_prompt_len", [0, 1, 3])
def test_suffix_matches_reference_action_prompt_layout_and_attention(tiny_model_factory, soft_prompt_len):
    """Reference layout: [VLM block][expert prompt block][expert action block]."""
    model = tiny_model_factory(soft_prompt_len=soft_prompt_len)
    actions = torch.randn(2, 2, 3)
    times = torch.tensor([0.25, 0.75])
    suffix, suffix_pad, suffix_att, condition = model.embed_suffix(actions, times)

    # Assemble the reference suffix independently, without the expansion/helper under test.
    projected_actions = model.action_in_proj(actions)
    expected_suffix = projected_actions
    if soft_prompt_len:
        expected_suffix = torch.cat([model.soft_prompt[None].expand(2, -1, -1), projected_actions], dim=1)
    torch.testing.assert_close(suffix, expected_suffix)
    assert suffix.shape == (2, soft_prompt_len + 2, 8)
    assert suffix_pad.all()
    expected_blocks = ([1] + [0] * (soft_prompt_len - 1) if soft_prompt_len else []) + [1, 0]
    torch.testing.assert_close(suffix_att, suffix.new_tensor(expected_blocks)[None].expand(2, -1))
    assert condition.shape == (2, 8)

    prefix, prefix_pad, prefix_att = model.embed_prefix(*_prefix_inputs())
    prefix_len = prefix.shape[1]
    prompt_end = prefix_len + soft_prompt_len
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    blocks = torch.cat([prefix_att, suffix_att], dim=1)
    visibility = modeling_quantile_head.make_att_2d_masks(pad, blocks)
    assert not visibility[:, :prefix_len, prefix_len:].any()
    assert not visibility[:, prefix_len:prompt_end, prompt_end:].any()
    assert visibility[:, prefix_len:prompt_end, prefix_len:prompt_end].all()
    assert visibility[:, prompt_end:, prefix_len:].all()
    torch.testing.assert_close(
        visibility[:, prefix_len:, :prefix_len], prefix_pad[:, None].expand(-1, soft_prompt_len + 2, -1)
    )


def _joint_forward(model, inputs, actions, times):
    prefix, prefix_pad, prefix_att = model.embed_prefix(*inputs)
    suffix, suffix_pad, suffix_att, condition = model.embed_suffix(actions, times)
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    visibility = modeling_quantile_head.make_att_2d_masks(pad, torch.cat([prefix_att, suffix_att], dim=1))
    outputs, _ = model.paligemma_with_expert.forward(
        attention_mask=model._prepare_attention_masks_4d(visibility),
        position_ids=pad.cumsum(dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, condition],
    )
    return outputs


def test_real_attention_prompt_changes_actions_but_not_vlm_representations(tiny_model_factory):
    torch.manual_seed(12)
    model = tiny_model_factory()
    model.eval()
    inputs = _prefix_inputs()
    actions = torch.randn(2, 2, 3)
    times = torch.tensor([0.25, 0.75])
    with torch.no_grad():
        prefix_before, suffix_before = _joint_forward(model, inputs, actions, times)
        model.soft_prompt.add_(torch.randn_like(model.soft_prompt))
        prefix_after, suffix_after = _joint_forward(model, inputs, actions, times)
        _, suffix_new_actions = _joint_forward(model, inputs, actions + 1, times)

    # Fully masked padding queries have no defined representation; compare valid image/text tokens.
    prefix_pad = model.embed_prefix(*inputs)[1]
    torch.testing.assert_close(prefix_before[prefix_pad], prefix_after[prefix_pad], rtol=0, atol=0)
    assert not torch.allclose(suffix_before[:, -2:], suffix_after[:, -2:])
    # Prompt tokens also cannot read the later action block, even across multiple layers.
    torch.testing.assert_close(suffix_after[:, :3], suffix_new_actions[:, :3], rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("train_expert_only,detach_action_from_vlm", [(True, False), (False, True)])
@pytest.mark.parametrize(
    "action_generation_mode,use_pure_regression_head,pure_regression_loss",
    [
        ("regression", False, "l1"),
        ("flow_matching", False, "l1"),
        ("regression", True, "l1"),
        ("regression", True, "l2"),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_action_loss_trains_prompt_with_detached_vlm(
    tiny_model_factory,
    checkpointing,
    train_expert_only,
    detach_action_from_vlm,
    action_generation_mode,
    use_pure_regression_head,
    pure_regression_loss,
    dtype,
):
    torch.manual_seed(12)
    model = tiny_model_factory(
        action_generation_mode=action_generation_mode,
        use_pure_regression_head=use_pure_regression_head,
        pure_regression_loss=pure_regression_loss,
        dtype=dtype,
        train_expert_only=train_expert_only,
        detach_action_from_vlm=detach_action_from_vlm,
    )
    model.train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    inputs = _prefix_inputs()
    actions = torch.randn(2, 2, 3)
    times = torch.tensor([0.25, 0.75])
    initial_prompt = model.soft_prompt.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)

    if use_pure_regression_head:
        predictions = model.forward_pure_regression(*inputs, torch.zeros_like(times), actions)
        errors = predictions - actions
        loss = errors.abs().mean() if pure_regression_loss == "l1" else errors.square().mean()
    elif action_generation_mode == "regression":
        quantiles = model.forward_regression(*inputs, times, actions)
        errors = actions[..., None] - quantiles
        levels = model.quantile_levels
        loss = torch.maximum(levels * errors, (levels - 1) * errors).mean()
    else:
        loss = model(*inputs, actions, torch.randn_like(actions), times).mean()
    loss.backward()

    assert model._detach_action_from_vlm()
    assert model.soft_prompt.grad is not None
    assert torch.isfinite(model.soft_prompt.grad).all()
    assert model.soft_prompt.grad.abs().sum() > 0
    vlm = model.paligemma_with_expert.paligemma
    assert all(parameter.grad is None for parameter in vlm.parameters())
    if train_expert_only:
        assert all(not parameter.requires_grad for parameter in vlm.parameters())
    expert = model.paligemma_with_expert.gemma_expert
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in expert.parameters()
    )
    assert model.action_out_proj.weight.grad.abs().sum() > 0

    optimizer.step()
    assert not torch.equal(initial_prompt, model.soft_prompt)


@pytest.mark.parametrize("train_soft_prompt", [False, True])
@pytest.mark.parametrize("prompt_lr", [None, 1e-4])
def test_actual_optimizer_respects_prompt_trainability_lr_and_weight_decay(
    tiny_model_factory, train_soft_prompt, prompt_lr
):
    model = tiny_model_factory(
        train_soft_prompt=train_soft_prompt,
        optimizer_soft_prompt_lr=prompt_lr,
        optimizer_soft_prompt_weight_decay=0.0,
    )
    policy = modeling_quantile_head.QuantileHeadPolicy.__new__(modeling_quantile_head.QuantileHeadPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = model.config
    policy.model = model
    optimizer = policy.config.get_optimizer_preset().build(policy.get_optim_params())
    parameter_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(param) for param in policy.parameters() if param.requires_grad}
    prompt_groups = [
        group for group in optimizer.param_groups if any(p is model.soft_prompt for p in group["params"])
    ]
    assert len(prompt_groups) == int(train_soft_prompt)
    if train_soft_prompt:
        assert prompt_groups[0]["lr"] == (prompt_lr or model.config.optimizer_lr)
        assert prompt_groups[0]["weight_decay"] == 0.0
    other_group = next(
        group
        for group in optimizer.param_groups
        if any(p is model.action_out_proj.weight for p in group["params"])
    )
    assert other_group["lr"] == model.config.optimizer_lr
    assert other_group["weight_decay"] == model.config.optimizer_weight_decay

    before = model.soft_prompt.detach().clone()
    quantiles = model.forward_regression(*_prefix_inputs(), torch.zeros(2), torch.randn(2, 2, 3))
    quantiles.square().mean().backward()
    optimizer.step()
    assert (model.soft_prompt.grad is not None) == train_soft_prompt
    assert torch.equal(before, model.soft_prompt) != train_soft_prompt


def test_cached_flow_matches_full_forward_and_uses_action_prompt(tiny_model_factory, monkeypatch):
    torch.manual_seed(12)
    model = tiny_model_factory(action_generation_mode="flow_matching")
    model.eval()
    inputs = _prefix_inputs()
    noise = torch.zeros(2, 2, 3)
    calls = []
    original_forward = model.paligemma_with_expert.forward

    def record_forward(*args, **kwargs):
        calls.append(kwargs)
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.paligemma_with_expert, "forward", record_forward)
    original_actions = model.sample_actions(*inputs, noise=noise, num_steps=1)
    assert len(calls) == 2
    prefix_call, suffix_call = calls
    prefix_pad = model.embed_prefix(*inputs)[1]
    assert prefix_call["use_cache"]
    assert prefix_call["inputs_embeds"][0].shape == (2, 7, 16)
    assert prefix_call["inputs_embeds"][1] is None
    assert suffix_call["inputs_embeds"][0] is None
    assert suffix_call["inputs_embeds"][1].shape == (2, 5, 8)
    torch.testing.assert_close(
        suffix_call["position_ids"], prefix_pad.sum(dim=1)[:, None] + torch.arange(5)[None]
    )
    with torch.no_grad():
        # One integration step is x_0 = x_1 - v(x_1, t=1); cached and joint attention agree.
        full_velocity = model._predict_action_values(*inputs, noise, torch.ones(2))
        torch.testing.assert_close(original_actions, noise - full_velocity)
        model.soft_prompt.add_(torch.randn_like(model.soft_prompt))
    updated_actions = model.sample_actions(*inputs, noise=noise, num_steps=1)
    assert original_actions.shape == (2, 2, 3)
    assert torch.isfinite(original_actions).all()
    assert torch.isfinite(updated_actions).all()
    assert not torch.allclose(original_actions, updated_actions)


@pytest.mark.parametrize("use_pure_regression_head", [False, True])
def test_soft_prompt_weights_and_inference_survive_state_dict_round_trip(
    tiny_model_factory, use_pure_regression_head
):
    source = tiny_model_factory(use_pure_regression_head=use_pure_regression_head)
    with torch.no_grad():
        source.soft_prompt.fill_(0.123)
    restored = tiny_model_factory(use_pure_regression_head=use_pure_regression_head)
    restored.load_state_dict(source.state_dict(), strict=True)
    torch.testing.assert_close(restored.soft_prompt, source.soft_prompt)
    inputs = _prefix_inputs()
    source.eval()
    restored.eval()
    times = torch.zeros(2)
    if use_pure_regression_head:
        assert source.quantile_delta_proj is None
        assert source.quantile_levels is None
        expected_actions = source.regress_actions_from_zero_query(*inputs, times)
        restored_actions = restored.regress_actions_from_zero_query(*inputs, times)
        assert expected_actions.shape == (2, 2, 3)
    else:
        expected_actions = source.regress_quantiles_from_zero_query(*inputs, times)
        restored_actions = restored.regress_quantiles_from_zero_query(*inputs, times)
        assert expected_actions.shape == (2, 2, 3, 3)
    torch.testing.assert_close(restored_actions, expected_actions)
