#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    DynamicCache = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_quantile_head import DEFAULT_IMAGE_SIZE, QuantileHeadConfig


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def clone_past_key_values(past_key_values):
    """Clone the DynamicCache returned by prefix prefill for compiled denoising."""
    return DynamicCache(
        tuple(
            (keys.clone(), values.clone(), sliding_window) for keys, values, sliding_window in past_key_values
        )
    )


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    layers,
    rotary_emb,
    detach_prefix_from_suffix: bool = False,
):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        if detach_prefix_from_suffix and i == 0:
            with torch.no_grad():
                hidden_states = hidden_states.detach()
                hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            gates.append(gate.detach() if gate is not None else None)
            query_states.append(query_state.detach())
            key_states.append(key_state.detach())
            value_states.append(value_state.detach())
            continue

        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        att_slice = att_output[:, start_pos:end_pos]
        if detach_prefix_from_suffix and i == 0:
            with torch.no_grad():
                hidden_states = hidden_states.detach()
                if att_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_slice = att_slice.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(att_slice.detach())
                out_emb = _gated_residual(hidden_states, out_emb, gates[i])
                after_first_residual = out_emb.clone()
                out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)
                out_emb = _gated_residual(after_first_residual, out_emb, gate)
            outputs_embeds.append(out_emb.detach())
            start_pos = end_pos
            continue

        if att_slice.dtype != layer.self_attn.o_proj.weight.dtype:
            att_slice = att_slice.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_slice)
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for QuantileHead."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        detach_prefix_from_suffix: bool = False,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        detach_prefix_from_suffix=detach_prefix_from_suffix,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        detach_prefix_from_suffix=detach_prefix_from_suffix,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    if detach_prefix_from_suffix and i == 0:
                        with torch.no_grad():
                            out_emb, _ = layernorm_forward(
                                final_norms[i], hidden_states.detach(), adarms_cond[i]
                            )
                        outputs_embeds.append(out_emb.detach())
                        continue
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


def build_monotonic_quantile_knots(
    median: Tensor, raw_gaps: Tensor, initial_gap: float, *, detach_median: bool = False
) -> Tensor:
    """Construct ordered quantile knots around an explicitly represented median.

    ``initial_gap`` sets the gap parameterization scale, so a zero raw value maps
    to that gap. It is not a floor: pinball loss may shrink or expand every gap.
    ``detach_median`` blocks only the nonmedian knots' direct path to the median;
    gradients through the gaps and through the center knot remain intact.
    """
    gaps = F.softplus(raw_gaps) * (initial_gap / math.log(2.0))
    middle = (raw_gaps.shape[-1] + 1) // 2
    lower_offsets = gaps[..., :middle].cumsum(dim=-1)
    upper_offsets = gaps[..., middle:].cumsum(dim=-1)
    nonmedian_center = median.detach() if detach_median else median
    lower_quantiles = (nonmedian_center.unsqueeze(-1) - lower_offsets).flip(dims=(-1,))
    upper_quantiles = nonmedian_center.unsqueeze(-1) + upper_offsets
    return torch.cat([lower_quantiles, median.unsqueeze(-1), upper_quantiles], dim=-1)


def build_independent_quantile_knots(median: Tensor, values: Tensor) -> Tensor:
    """Insert the separately predicted median without coupling or sorting the other quantiles."""
    middle = values.shape[-1] // 2
    return torch.cat([values[..., :middle], median.unsqueeze(-1), values[..., middle:]], dim=-1)


def extend_quantile_knots_to_unit_interval(
    quantile_knots: Tensor,
    quantile_levels: Tensor,
) -> tuple[Tensor, Tensor]:
    """Linearly extend interior quantile knots to levels 0 and 1.

    The learned outer knots keep their configured quantile semantics. For example,
    with levels spanning [0.025, 0.975], the first learned knot remains q_0.025
    instead of becoming a support boundary with 2.5% point mass.
    """
    if quantile_levels.ndim != 1:
        raise ValueError(f"quantile_levels must be one-dimensional, got {quantile_levels.ndim}")
    if quantile_levels.numel() < 2:
        raise ValueError("At least two quantile levels are required for tail extrapolation")
    if quantile_knots.shape[-1] != quantile_levels.numel():
        raise ValueError(
            "quantile_knots and quantile_levels must have matching final dimensions, "
            f"got {quantile_knots.shape[-1]} and {quantile_levels.numel()}"
        )

    levels = quantile_levels.to(device=quantile_knots.device)
    lower_level_width = levels[1] - levels[0]
    upper_level_width = levels[-1] - levels[-2]
    lower_scale = (levels[0] / lower_level_width).to(dtype=quantile_knots.dtype)
    upper_scale = ((1.0 - levels[-1]) / upper_level_width).to(dtype=quantile_knots.dtype)

    lower_knot = quantile_knots[..., :1] - lower_scale * (quantile_knots[..., 1:2] - quantile_knots[..., :1])
    upper_knot = quantile_knots[..., -1:] + upper_scale * (
        quantile_knots[..., -1:] - quantile_knots[..., -2:-1]
    )
    extended_knots = torch.cat([lower_knot, quantile_knots, upper_knot], dim=-1)
    extended_levels = torch.cat([levels.new_zeros(1), levels, levels.new_ones(1)])
    return extended_knots, extended_levels


class QuantileHeadPytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core QuantileHead PyTorch model."""

    def __init__(self, config: QuantileHeadConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        if config.action_generation_mode == "regression" and not config.use_pure_regression_head:
            num_delta_knots = config.marginal_quantile_num_knots - 1
            self.quantile_delta_proj = nn.Linear(
                action_expert_config.width,
                config.max_action_dim * num_delta_knots,
            )
            nn.init.zeros_(self.quantile_delta_proj.weight)
            nn.init.zeros_(self.quantile_delta_proj.bias)
            if config.marginal_quantile_variant == "independent":
                # The same tensor shapes have different meanings in independent and gap heads.
                # A persistent tag makes strict loading reject accidental architecture switches.
                self.quantile_delta_proj.register_buffer("_independent_quantile_head", torch.tensor(True))
                self._initialize_independent_quantile_head()
            quantile_levels = torch.linspace(
                config.marginal_quantile_min_level,
                1.0 - config.marginal_quantile_min_level,
                config.marginal_quantile_num_knots,
                dtype=torch.float32,
            )
            self.register_buffer("quantile_levels", quantile_levels, persistent=False)
        else:
            self.quantile_delta_proj = None
            self.register_buffer("quantile_levels", None, persistent=False)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        if config.soft_prompt_len > 0:
            self.soft_prompt = nn.Parameter(
                torch.empty(config.soft_prompt_len, action_expert_config.width, dtype=torch.float32)
            )
            nn.init.normal_(self.soft_prompt, std=config.soft_prompt_init_std)
            self.soft_prompt.requires_grad_(config.train_soft_prompt)
        else:
            self.register_parameter("soft_prompt", None)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            compile_mode = config.compile_mode
            if config.gradient_checkpointing:
                if compile_mode == "max-autotune":
                    # CUDA Graph capture is incompatible with the FunctionalTensors produced by
                    # activation checkpointing under DDP. Keep Inductor autotuning, but disable graphs.
                    compile_mode = "max-autotune-no-cudagraphs"
                    logging.info(
                        "Using torch.compile mode 'max-autotune-no-cudagraphs' because QuantileHead "
                        "gradient checkpointing is enabled."
                    )

                # DDPOptimizer splits the compiled forward at gradient-bucket boundaries. With
                # non-reentrant activation checkpointing, those boundaries can receive
                # FunctionalTensors, which Inductor cannot inspect for input alignment. Disabling
                # this optional partitioning keeps DDP and Inductor enabled while avoiding that
                # invalid cross-graph data_ptr() access.
                torch._dynamo.config.optimize_ddp = False
                logging.info(
                    "Disabled TorchDynamo DDPOptimizer graph partitioning for QuantileHead gradient "
                    "checkpointing; DDP and torch.compile remain enabled."
                )

            self.sample_actions = torch.compile(self.sample_actions, mode=compile_mode)
            self.regress_quantiles_from_zero_query = torch.compile(
                self.regress_quantiles_from_zero_query, mode=compile_mode
            )
            if config.use_pure_regression_head:
                self.regress_actions_from_zero_query = torch.compile(
                    self.regress_actions_from_zero_query, mode=compile_mode
                )
                self.forward_pure_regression = torch.compile(self.forward_pure_regression, mode=compile_mode)
            # Also compile the main forward passes used during training
            self.forward = torch.compile(self.forward, mode=compile_mode)
            self.forward_regression = torch.compile(self.forward_regression, mode=compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for QuantileHeadPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for QuantileHeadPytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _detach_action_from_vlm(self) -> bool:
        return self.config.detach_action_from_vlm or self.config.train_expert_only

    def _expand_action_soft_prompt(
        self, bsize: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        soft_prompt = self.soft_prompt[None, :, :].expand(bsize, -1, -1)
        return soft_prompt.to(device=device, dtype=dtype)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb
        bsize = action_time_emb.shape[0]
        device = action_time_emb.device

        if self.soft_prompt is not None:
            soft_prompt = self._expand_action_soft_prompt(bsize, device, action_time_emb.dtype)
            embs.append(soft_prompt)
            pad_masks.append(torch.ones(bsize, self.config.soft_prompt_len, dtype=torch.bool, device=device))
            att_masks += [1] + ([0] * (self.config.soft_prompt_len - 1))

        embs.append(action_time_emb)
        _, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Start a new block so VLM prefix tokens do not attend to action-side tokens.
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def _prepare_action_context(self, images, img_masks, tokens, masks):
        """Prepare VLM/context tokens used by the action expert."""
        detach_action_from_vlm = self._detach_action_from_vlm()
        if detach_action_from_vlm:
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                    images, img_masks, tokens, masks
                )
            prefix_embs = prefix_embs.detach()
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks
            )
        return prefix_embs, prefix_pad_masks, prefix_att_masks, detach_action_from_vlm

    def _predict_action_hidden_from_context(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        detach_prefix_from_suffix,
        action_queries,
        time,
    ) -> Tensor:
        """Return final action-token hidden states for prepared context and query vectors."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(action_queries, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        ):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                detach_prefix_from_suffix=detach_prefix_from_suffix,
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        return suffix_out.to(dtype=torch.float32)

    def _predict_action_values_from_context(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        detach_prefix_from_suffix,
        action_queries,
        time,
    ) -> Tensor:
        suffix_out = self._predict_action_hidden_from_context(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            detach_prefix_from_suffix,
            action_queries,
            time,
        )

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        return self._apply_checkpoint(action_out_proj_func, suffix_out)

    @torch.no_grad()
    def _initialize_independent_quantile_head(self) -> None:
        """Match the standard head's initial predictions without consuming additional randomness.

        Run again after bootstrapping from a base checkpoint so the independent
        predictions use the loaded action projection rather than its temporary initialization.
        """
        if self.quantile_delta_proj is None:
            raise RuntimeError("Independent quantile initialization requires a quantile head")
        middle = self.config.marginal_quantile_num_knots // 2
        weight = self.action_out_proj.weight
        bias = self.action_out_proj.bias
        offsets = (
            torch.cat(
                [
                    torch.arange(-middle, 0, device=bias.device),
                    torch.arange(1, middle + 1, device=bias.device),
                ]
            ).to(bias.dtype)
            * self.config.marginal_quantile_initial_gap
        )
        self.quantile_delta_proj.weight.copy_(
            weight[:, None, :].expand(-1, 2 * middle, -1).reshape_as(self.quantile_delta_proj.weight)
        )
        self.quantile_delta_proj.bias.copy_(
            (bias[:, None] + offsets).reshape_as(self.quantile_delta_proj.bias)
        )

    def _build_marginal_quantiles(self, suffix_out: Tensor) -> Tensor:
        """Build the configured marginal quantiles, retaining the center knot's identity."""
        if self.quantile_delta_proj is None:
            raise RuntimeError(
                "Marginal quantile heads require regression mode with use_pure_regression_head=False"
            )

        def quantile_heads_func(suffix_out):
            median = self.action_out_proj(suffix_out)
            raw_gaps = self.quantile_delta_proj(suffix_out)
            raw_gaps = raw_gaps.reshape(
                *raw_gaps.shape[:2],
                self.config.max_action_dim,
                self.config.marginal_quantile_num_knots - 1,
            )
            variant = self.config.marginal_quantile_variant
            if variant == "independent":
                return build_independent_quantile_knots(median, raw_gaps)
            return build_monotonic_quantile_knots(
                median,
                raw_gaps,
                self.config.marginal_quantile_initial_gap,
                detach_median=variant == "stop_median_gradient",
            )

        return self._apply_checkpoint(quantile_heads_func, suffix_out)

    def interpolate_marginal_quantiles(self, quantile_knots: Tensor, marginal_levels: Tensor) -> Tensor:
        """Invert the tail-extended piecewise-linear quantile functions."""
        if self.quantile_levels is None:
            raise RuntimeError("Marginal quantile levels are only available in regression mode")

        knots, levels = extend_quantile_knots_to_unit_interval(quantile_knots, self.quantile_levels)
        requested_levels = marginal_levels.to(device=levels.device, dtype=levels.dtype)
        requested_levels = requested_levels.clamp(0.0, 1.0).contiguous()
        upper_indices = torch.searchsorted(levels.contiguous(), requested_levels)
        upper_indices = upper_indices.clamp(1, levels.numel() - 1)
        lower_indices = upper_indices - 1
        lower_levels = levels[lower_indices]
        upper_levels = levels[upper_indices]
        interpolation_weight = (requested_levels - lower_levels) / (upper_levels - lower_levels)
        lower_knots = knots.gather(-1, lower_indices.unsqueeze(-1)).squeeze(-1)
        upper_knots = knots.gather(-1, upper_indices.unsqueeze(-1)).squeeze(-1)
        return lower_knots + interpolation_weight * (upper_knots - lower_knots)

    def median_actions_from_quantiles(self, quantile_knots: Tensor) -> Tensor:
        """Return the explicitly represented marginal median for every action coordinate."""
        if self.quantile_levels is None:
            raise RuntimeError("Marginal quantile levels are only available in regression mode")
        if quantile_knots.shape[-1] != self.quantile_levels.numel():
            raise ValueError(
                "quantile_knots and quantile_levels must have matching final dimensions, "
                f"got {quantile_knots.shape[-1]} and {self.quantile_levels.numel()}"
            )
        return quantile_knots[..., quantile_knots.shape[-1] // 2]

    def _predict_action_values(
        self,
        images,
        img_masks,
        tokens,
        masks,
        action_queries,
        time,
    ) -> Tensor:
        """Predict action-valued outputs from action-side query vectors and a time condition."""
        prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix = (
            self._prepare_action_context(images, img_masks, tokens, masks)
        )
        return self._predict_action_values_from_context(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            detach_prefix_from_suffix,
            action_queries,
            time,
        )

    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> Tensor:
        """Do a flow-matching training forward pass and compute the velocity loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t = self._predict_action_values(images, img_masks, tokens, masks, x_t, time)

        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_regression(
        self,
        images,
        img_masks,
        tokens,
        masks,
        time,
        actions,
    ) -> Tensor:
        """Predict marginal action quantiles from an all-zero query."""
        prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix = (
            self._prepare_action_context(images, img_masks, tokens, masks)
        )

        action_queries = torch.zeros_like(actions)
        suffix_out = self._predict_action_hidden_from_context(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            detach_prefix_from_suffix,
            action_queries,
            time,
        )
        return self._build_marginal_quantiles(suffix_out)

    def forward_pure_regression(self, images, img_masks, tokens, masks, time, actions) -> Tensor:
        """Predict point actions with the shared action projection and an all-zero query."""
        return self._predict_action_values(images, img_masks, tokens, masks, torch.zeros_like(actions), time)

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    @torch.no_grad()
    def regress_actions_from_zero_query(self, images, img_masks, tokens, masks, time) -> Tensor:
        """Predict point actions in one pass, without quantile construction or sampling."""
        action_queries = torch.zeros(
            tokens.shape[0],
            self.config.chunk_size,
            self.config.max_action_dim,
            dtype=torch.float32,
            device=tokens.device,
        )
        return self._predict_action_values(images, img_masks, tokens, masks, action_queries, time)

    @torch.no_grad()
    def regress_quantiles_from_zero_query(self, images, img_masks, tokens, masks, time) -> Tensor:
        """Predict marginal quantile knots in one pass from an all-zero query."""
        prefix_embs, prefix_pad_masks, prefix_att_masks, detach_prefix_from_suffix = (
            self._prepare_action_context(images, img_masks, tokens, masks)
        )
        action_queries = torch.zeros(
            tokens.shape[0],
            self.config.chunk_size,
            self.config.max_action_dim,
            dtype=torch.float32,
            device=tokens.device,
        )
        suffix_out = self._predict_action_hidden_from_context(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            detach_prefix_from_suffix,
            action_queries,
            time,
        )

        return self._build_marginal_quantiles(suffix_out)

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class QuantileHeadPolicy(PreTrainedPolicy):
    """QuantileHead Policy for LeRobot."""

    config_class = QuantileHeadConfig
    name = "quantile_head"

    def __init__(
        self,
        config: QuantileHeadConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core QuantileHead model
        self.init_rtc_processor()
        self.model = QuantileHeadPytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self._marginal_quantile_sampling_generator = torch.Generator(device="cpu")
        self._marginal_quantile_sampling_generator.manual_seed(config.marginal_quantile_sampling_seed)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        allow_missing_action_modules: bool = False,
        allow_missing_soft_prompt: bool = False,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "QuantileHead extends the OpenPI PI0.5 backbone with a marginal quantile action head. \n"
            "The backbone retains the OpenPI parameter structure for checkpoint compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            from transformers.utils import cached_file

            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            if resolved_file is None:
                raise FileNotFoundError(f"model.safetensors was not found in {pretrained_name_or_path}")

            from safetensors.torch import load_file

            original_state_dict = load_file(resolved_file)
            print("✓ Loaded state dict from model.safetensors")

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0
            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load non-strictly first so we can distinguish an intentionally absent whole
            # action-side module (base-checkpoint bootstrap) from a damaged partial module.
            missing_keys, unexpected_keys = model.load_state_dict(
                remapped_state_dict,
                strict=False,
            )

            allowed_missing_keys: set[str] = set()
            if allow_missing_soft_prompt or allow_missing_action_modules:
                prompt_keys = {key for key in model.state_dict() if key == "model.soft_prompt"}
                if prompt_keys and prompt_keys.issubset(missing_keys):
                    allowed_missing_keys.update(prompt_keys)
                    print("Soft prompt parameters were initialized from scratch.")
            if allow_missing_action_modules:
                expected_keys = set(model.state_dict())
                bootstrap_groups: list[tuple[str, set[str]]] = []
                if model.config.action_generation_mode == "regression":
                    bootstrap_groups.append(
                        (
                            "marginal quantile head",
                            {key for key in expected_keys if key.startswith("model.quantile_delta_proj.")},
                        )
                    )

                missing_key_set = set(missing_keys)
                for group_name, group_keys in bootstrap_groups:
                    missing_group_keys = missing_key_set & group_keys
                    if group_keys and missing_group_keys == group_keys:
                        allowed_missing_keys.update(group_keys)
                        print(f"{group_name.capitalize()} parameters were initialized from scratch.")

            if strict:
                missing_blockers = [key for key in missing_keys if key not in allowed_missing_keys]
                if missing_blockers or unexpected_keys:
                    raise RuntimeError(
                        "Error(s) in loading state_dict for QuantileHeadPolicy:\n"
                        f"\tMissing key(s): {missing_blockers}\n"
                        f"\tUnexpected key(s): {unexpected_keys}"
                    )

            if model.config.marginal_quantile_variant == "independent":
                independent_keys = {
                    key for key in model.state_dict() if key.startswith("model.quantile_delta_proj.")
                }
                if independent_keys and independent_keys.issubset(allowed_missing_keys):
                    model.model._initialize_independent_quantile_head()

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load QuantileHead checkpoint from {pretrained_name_or_path}: {e}"
            ) from e

        # Match PreTrainedPolicy.from_pretrained: callers receive an inference-ready
        # module, while the training pipeline explicitly switches it back to train mode.
        model.eval()
        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for quantile_head
            # quantile_head model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in quantile_head
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in quantile_head mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> Any:
        if self.config.soft_prompt_len <= 0:
            return self.parameters()

        other_params = []
        soft_prompt_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "soft_prompt" in name:
                soft_prompt_params.append(param)
            else:
                other_params.append(param)

        if not soft_prompt_params:
            return other_params

        param_groups: list[dict[str, Any]] = []
        if other_params:
            param_groups.append({"params": other_params})
        param_groups.append(
            {
                "params": soft_prompt_params,
                "lr": self.config.optimizer_soft_prompt_lr or self.config.optimizer_lr,
                "weight_decay": self.config.optimizer_soft_prompt_weight_decay,
                "name": "soft_prompt",
            }
        )
        return param_groups

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # Sampling state is episode-local, while the dedicated generator keeps advancing so
        # consecutive episodes receive different (but reproducible) stochastic trajectories.
        self._marginal_quantile_sampling_latent: Tensor | None = None
        self._last_marginal_quantile_levels: Tensor | None = None

    def _get_marginal_quantile_sampling_generator(self) -> torch.Generator:
        """Return the sampler-local CPU RNG without touching PyTorch's global RNG."""
        generator = getattr(self, "_marginal_quantile_sampling_generator", None)
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.config.marginal_quantile_sampling_seed)
            self._marginal_quantile_sampling_generator = generator
        return generator

    def _marginal_quantile_sampling_exclusion_mask(self, action_dim: int) -> Tensor:
        """Identify action coordinates that should remain at their marginal medians."""
        exclude_joints = self.config.marginal_quantile_sampling_exclude_joints
        exclude_tokens = [str(name).lower() for name in exclude_joints if str(name)]
        mask = torch.zeros(action_dim, dtype=torch.bool)
        if not exclude_tokens:
            return mask

        action_names = self.config.action_feature_names
        # Some datasets provide a group label such as ["actions"], not one name
        # per coordinate. Only use names when they cover the decoded action vector.
        if action_names is not None and len(action_names) >= action_dim:
            for index, name in enumerate(action_names[:action_dim]):
                action_name = str(name).lower()
                mask[index] = any(token == action_name or token in action_name for token in exclude_tokens)
            return mask

        # LIBERO and most end-effector datasets place a single gripper command last,
        # even when the metadata does not supply individual coordinate names.
        if action_dim > 0 and any(token == "gripper" for token in exclude_tokens):
            mask[-1] = True
        return mask

    def _sample_temporally_coherent_quantile_levels(
        self,
        batch_size: int,
        action_dim: int,
    ) -> Tensor:
        """Sample one smoothly varying quantile level per batch item and action dimension."""
        generator = self._get_marginal_quantile_sampling_generator()
        latent_shape = (batch_size, action_dim)
        previous_latent = getattr(self, "_marginal_quantile_sampling_latent", None)
        if previous_latent is None or previous_latent.shape != latent_shape:
            latent = torch.randn(latent_shape, generator=generator, dtype=torch.float32)
        else:
            rho = float(self.config.marginal_quantile_sampling_ar_coefficient)
            if rho == 1.0:
                latent = previous_latent.clone()
            else:
                innovation = torch.randn(latent_shape, generator=generator, dtype=torch.float32)
                latent = rho * previous_latent + math.sqrt(1.0 - rho**2) * innovation
        self._marginal_quantile_sampling_latent = latent

        temperature = float(self.config.marginal_quantile_sampling_temperature)
        gaussian_cdf = 0.5 * (1.0 + torch.erf(temperature * latent / math.sqrt(2.0)))
        centered_rank = 2.0 * gaussian_cdf - 1.0
        min_level = float(self.config.marginal_quantile_sampling_min_level)
        max_level = float(self.config.marginal_quantile_sampling_max_level)
        levels = torch.where(
            centered_rank < 0,
            0.5 + centered_rank * (0.5 - min_level),
            0.5 + centered_rank * (max_level - 0.5),
        )

        exclusion_mask = self._marginal_quantile_sampling_exclusion_mask(action_dim)
        if exclusion_mask.any():
            levels[:, exclusion_mask] = 0.5
        self._last_marginal_quantile_levels = levels.clone()
        return levels

    def _decode_regression_quantiles(self, quantile_knots: Tensor) -> Tensor:
        """Decode learned marginal quantiles according to the configured inference strategy."""
        inference_mode = self.config.marginal_quantile_inference_mode
        if inference_mode not in {"median", "temporal_sample"}:
            raise ValueError(
                "marginal_quantile_inference_mode must be one of ['median', 'temporal_sample'], "
                f"got {inference_mode}"
            )
        if inference_mode == "median" or float(self.config.marginal_quantile_sampling_temperature) == 0.0:
            self._last_marginal_quantile_levels = torch.full(
                (quantile_knots.shape[0], quantile_knots.shape[2]),
                0.5,
                dtype=torch.float32,
            )
            return self.model.median_actions_from_quantiles(quantile_knots)

        levels = self._sample_temporally_coherent_quantile_levels(
            quantile_knots.shape[0], quantile_knots.shape[2]
        )
        levels = levels.to(device=quantile_knots.device, dtype=quantile_knots.dtype)
        levels = levels[:, None, :].expand(*quantile_knots.shape[:-1])
        return self.model.interpolate_marginal_quantiles(quantile_knots, levels)

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def make_action_padding_keep_mask(
        self,
        values: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Return a coordinate mask that excludes action steps padded past an episode boundary."""
        keep_mask = torch.ones_like(values, dtype=torch.bool)
        if action_is_pad is None:
            return keep_mask

        action_is_pad = action_is_pad.to(device=values.device, dtype=torch.bool)
        if action_is_pad.shape != values.shape[:2]:
            raise ValueError(
                "action_is_pad must match the batch and action-horizon dimensions, "
                f"got {tuple(action_is_pad.shape)} for values {tuple(values.shape)}"
            )
        return keep_mask & ~action_is_pad.unsqueeze(-1)

    def make_action_loss_keep_mask(
        self,
        values: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        """Exclude padding and randomly mask an exact fraction of each sample's valid losses."""
        keep_mask = self.make_action_padding_keep_mask(values, action_is_pad)
        mask_ratio = self.config.action_loss_mask_ratio
        if mask_ratio <= 0 or not self.training or values.numel() == 0:
            return keep_mask

        batch_size = values.shape[0]
        values_per_sample = values[0].numel()
        flat_keep_mask = keep_mask.reshape(batch_size, values_per_sample)
        valid_counts = flat_keep_mask.sum(dim=1)
        num_masked_values = torch.ceil(valid_counts.float() * mask_ratio).long()
        num_masked_values = torch.minimum(
            num_masked_values,
            (valid_counts - 1).clamp_min(0),
        )
        if not torch.any(num_masked_values > 0):
            return keep_mask

        random_scores = torch.rand(batch_size, values_per_sample, device=values.device)
        random_scores = random_scores.masked_fill(~flat_keep_mask, float("-inf"))
        random_order = random_scores.argsort(dim=1, descending=True)
        random_ranks = torch.empty_like(random_order)
        ordered_ranks = torch.arange(values_per_sample, device=values.device).expand_as(random_order)
        random_ranks.scatter_(dim=1, index=random_order, src=ordered_ranks)
        randomly_masked = flat_keep_mask & (random_ranks < num_masked_values.unsqueeze(1))
        flat_keep_mask &= ~randomly_masked
        return keep_mask

    def compute_marginal_quantile_loss(
        self,
        clean_actions: Tensor,
        quantile_knots: Tensor,
        keep_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the masked marginal pinball loss."""
        quantile_levels = self.model.quantile_levels
        if quantile_levels is None:
            raise RuntimeError("Marginal quantile levels are unavailable outside regression mode")

        # Clean masked targets before arithmetic, including non-finite padded values.
        # Keep every head in the graph even for median-only training (zero nonmedian gradients).
        expanded_mask = keep_mask.unsqueeze(-1)
        errors = torch.where(expanded_mask, clean_actions.unsqueeze(-1), 0.0) - torch.where(
            expanded_mask, quantile_knots, 0.0
        )
        levels = quantile_levels.to(device=errors.device, dtype=errors.dtype)
        levels = levels.view(1, 1, 1, -1)
        pinball_losses = torch.maximum(levels * errors, (levels - 1.0) * errors)
        supervised_knots = self.config.marginal_quantile_num_knots
        if self.config.marginal_quantile_variant == "median_only":
            center_mask = (
                torch.arange(supervised_knots, device=quantile_knots.device) == supervised_knots // 2
            )
            expanded_mask = expanded_mask & center_mask
            supervised_knots = 1
        masked_pinball_losses = torch.where(
            expanded_mask,
            pinball_losses,
            torch.zeros_like(pinball_losses),
        )
        kept_values = keep_mask.sum(dim=(1, 2)).clamp_min(1)
        per_sample_loss = masked_pinball_losses.sum(dim=(1, 2, 3))
        per_sample_loss = per_sample_loss / (kept_values * supervised_knots)
        return per_sample_loss, masked_pinball_losses

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        original_action_dim = self.config.output_features[ACTION].shape[0]

        if self.config.action_generation_mode == "regression":
            time = torch.zeros(tokens.shape[0], dtype=torch.float32, device=tokens.device)
            if self.config.use_pure_regression_head:
                actions = self.model.regress_actions_from_zero_query(images, img_masks, tokens, masks, time)
            else:
                quantile_knots = self.model.regress_quantiles_from_zero_query(
                    images,
                    img_masks,
                    tokens,
                    masks,
                    time,
                )
                quantile_knots = quantile_knots[:, :, :original_action_dim]
                actions = self._decode_regression_quantiles(quantile_knots)
        else:
            # Sample actions using the model (pass through RTC kwargs, no separate state needed for QuantileHead)
            actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        actions = actions[:, :, :original_action_dim].clone()

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)
        original_action_dim = self.config.output_features[ACTION].shape[0]

        if self.config.use_pure_regression_head:
            time = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)
            predicted_actions = self.model.forward_pure_regression(
                images, img_masks, tokens, masks, time, actions
            )
            predicted_actions = predicted_actions[:, :, :original_action_dim]
            clean_actions = actions[:, :, :original_action_dim]
            action_is_pad = batch.get(f"{ACTION}_is_pad")
            padding_keep_mask = self.make_action_padding_keep_mask(clean_actions, action_is_pad)
            loss_keep_mask = self.make_action_loss_keep_mask(clean_actions, action_is_pad)
            # Mask before computing errors so padded targets cannot introduce NaN gradients.
            errors = torch.where(loss_keep_mask, predicted_actions, 0.0) - torch.where(
                loss_keep_mask, clean_actions, 0.0
            )
            losses = errors.abs() if self.config.pure_regression_loss == "l1" else errors.square()
            kept_values = loss_keep_mask.sum(dim=(1, 2)).clamp_min(1)
            per_sample_loss = losses.sum(dim=(1, 2)) / kept_values
            loss = per_sample_loss.mean()
            per_dim_kept_values = loss_keep_mask.sum(dim=(0, 1)).clamp_min(1)
            loss_per_dim = losses.sum(dim=(0, 1)) / per_dim_kept_values
            loss_dict = {
                "action_loss_mask_ratio": self.config.action_loss_mask_ratio,
                "action_loss_masked_fraction": (~loss_keep_mask).float().mean().item(),
                "action_padding_fraction": (~padding_keep_mask).float().mean().item(),
                f"loss_pure_regression_{self.config.pure_regression_loss}": loss.item(),
                "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
                "loss": loss.item(),
            }
            if reduction == "none":
                return per_sample_loss, loss_dict
            return loss, loss_dict

        if self.config.action_generation_mode == "regression":
            time = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)
            quantile_knots = self.model.forward_regression(
                images,
                img_masks,
                tokens,
                masks,
                time,
                actions,
            )
            clean_actions = actions[:, :, :original_action_dim]
            quantile_knots = quantile_knots[:, :, :original_action_dim]
            action_is_pad = batch.get(f"{ACTION}_is_pad")
            padding_keep_mask = self.make_action_padding_keep_mask(clean_actions, action_is_pad)
            loss_keep_mask = self.make_action_loss_keep_mask(clean_actions, action_is_pad)
            outside_outer_quantiles = (clean_actions < quantile_knots[..., 0]) | (
                clean_actions > quantile_knots[..., -1]
            )
            marginal_pinball_loss_per_sample, masked_pinball_losses = self.compute_marginal_quantile_loss(
                clean_actions, quantile_knots, loss_keep_mask
            )
            per_sample_loss = marginal_pinball_loss_per_sample
            loss = per_sample_loss.mean()

            middle = self.config.marginal_quantile_num_knots // 2
            median_actions = quantile_knots[..., middle]
            padding_kept_values = padding_keep_mask.sum().clamp_min(1)
            marginal_quantile_tail_fraction = (
                outside_outer_quantiles & padding_keep_mask
            ).float().sum() / padding_kept_values
            squared_median_errors = (median_actions - clean_actions).square()
            squared_median_errors = torch.where(
                padding_keep_mask,
                squared_median_errors,
                torch.zeros_like(squared_median_errors),
            )
            median_mse = squared_median_errors.sum() / padding_kept_values
            quantile_widths = quantile_knots[..., -1] - quantile_knots[..., 0]
            quantile_widths = torch.where(
                padding_keep_mask,
                quantile_widths,
                torch.zeros_like(quantile_widths),
            )
            quantile_width = quantile_widths.sum() / padding_kept_values
            adjacent_gaps = quantile_knots[..., 1:] - quantile_knots[..., :-1]
            actual_log_gaps = adjacent_gaps.float().clamp_min(torch.finfo(torch.float32).tiny).log()
            valid_log_gaps = actual_log_gaps.masked_select(padding_keep_mask.unsqueeze(-1))
            if valid_log_gaps.numel() == 0:
                mean_log_gap = actual_log_gaps.new_zeros(())
                min_log_gap = actual_log_gaps.new_zeros(())
            else:
                mean_log_gap = valid_log_gaps.mean()
                min_log_gap = valid_log_gaps.min()
            per_dim_kept_values = loss_keep_mask.sum(dim=[0, 1]).clamp_min(1)
            loss_per_dim = masked_pinball_losses.sum(dim=[0, 1, 3])
            supervised_knots = (
                1
                if self.config.marginal_quantile_variant == "median_only"
                else self.config.marginal_quantile_num_knots
            )
            loss_per_dim = loss_per_dim / (per_dim_kept_values * supervised_knots)
            crossing_fraction = ((adjacent_gaps < 0) & padding_keep_mask.unsqueeze(-1)).sum() / (
                padding_kept_values * (self.config.marginal_quantile_num_knots - 1)
            )

            loss_dict = {
                "action_loss_mask_ratio": self.config.action_loss_mask_ratio,
                "action_loss_masked_fraction": (~loss_keep_mask).float().mean().item(),
                "action_padding_fraction": (~padding_keep_mask).float().mean().item(),
                "loss_marginal_quantile": marginal_pinball_loss_per_sample.mean().item(),
                "marginal_quantile_tail_fraction": marginal_quantile_tail_fraction.item(),
                "marginal_quantile_outer_coverage": (1.0 - marginal_quantile_tail_fraction).item(),
                "loss_median_mse": median_mse.item(),
                "marginal_quantile_mean_width": quantile_width.item(),
                "marginal_quantile_mean_log_gap": mean_log_gap.item(),
                "marginal_quantile_min_log_gap": min_log_gap.item(),
                "marginal_quantile_crossing_fraction": crossing_fraction.item(),
                "marginal_quantile_supervised_knots": supervised_knots,
                "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
                "loss": loss.item(),
            }
            if reduction == "none":
                return per_sample_loss, loss_dict
            return loss, loss_dict

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)

        losses = losses[:, :, :original_action_dim]
        action_is_pad = batch.get(f"{ACTION}_is_pad")
        padding_keep_mask = self.make_action_padding_keep_mask(losses, action_is_pad)
        loss_keep_mask = self.make_action_loss_keep_mask(losses, action_is_pad)
        masked_losses = torch.where(loss_keep_mask, losses, torch.zeros_like(losses))
        per_sample_kept_values = loss_keep_mask.sum(dim=(1, 2)).clamp_min(1)
        per_sample_loss = masked_losses.sum(dim=(1, 2)) / per_sample_kept_values
        masked_loss = per_sample_loss.mean()
        per_dim_kept_values = loss_keep_mask.sum(dim=[0, 1]).clamp_min(1)
        loss_per_dim = masked_losses.sum(dim=[0, 1]) / per_dim_kept_values
        loss_dict = {
            "action_loss_mask_ratio": self.config.action_loss_mask_ratio,
            "action_loss_masked_fraction": (~loss_keep_mask).float().mean().item(),
            "action_padding_fraction": (~padding_keep_mask).float().mean().item(),
            "loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist(),
        }
        if reduction == "none":
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict

        loss_dict["loss"] = masked_loss.item()
        return masked_loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for QuantileHead fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|quantile_delta_proj|time_mlp_in|time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        super()._validate_peft_config(peft_config)
        self._validate_soft_prompt_peft_config(peft_config)

    def _validate_soft_prompt_peft_config(self, peft_config) -> None:
        if self.config.soft_prompt_len:
            raise ValueError(
                "QuantileHead reference action-side soft prompts require non-PEFT training and checkpoints. "
                "The standalone model.soft_prompt Parameter cannot be saved via PEFT modules_to_save. "
                "Disable PEFT to preserve the reference prompt training and checkpoint behavior."
            )

    def _prepare_peft_config_for_loading(self, peft_config):
        """Validate that the current action-side prompt can be loaded through PEFT."""
        self._validate_soft_prompt_peft_config(peft_config)
        return peft_config
