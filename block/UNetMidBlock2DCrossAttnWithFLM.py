import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.transformers import Transformer2DModel
from typing import Any, Dict, Optional, Tuple, Union

from block.FuzzyLearningModule import FuzzyLearningModule


class UNetMidBlock2DCrossAttnWithFLM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            num_attention_heads: int = 8,
            cross_attention_dim: int = 1280,
            resnet_eps: float = 1e-6,
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.attentions = nn.ModuleList()
        self.resnets = nn.ModuleList()
        self.flms = nn.ModuleList()

        # Add initial Resnet block
        self.resnets.append(ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            non_linearity=resnet_act_fn,
        ))

        for _ in range(num_layers):
            # Add attention block
            self.attentions.append(Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=out_channels // num_attention_heads,
                in_channels=out_channels,
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=True,
                attention_type="default",
            ))

            # Add FLM block after each attention layer
            self.flms.append(FuzzyLearningModule(out_channels))

            # Add Resnet block
            self.resnets.append(ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                non_linearity=resnet_act_fn,
            ))

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        # Iterate over attentions, FLMs, and resnets
        for attn, flm, resnet in zip(self.attentions, self.flms, self.resnets[1:]):
        # for attn, resnet in zip(self.attentions,  self.resnets[1:]):
            # Cross attention
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # Fuzzy learning module
            hidden_states = flm(hidden_states)

            # Residual block
            hidden_states = resnet(hidden_states, temb)

        return hidden_states
