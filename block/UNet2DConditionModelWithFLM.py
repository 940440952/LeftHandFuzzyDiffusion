import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from torch import nn
from block.FuzzyLearningModule import FuzzyLearningModule
from diffusers.models.transformers import Transformer2DModel
from typing import Any, Dict, List, Optional, Tuple, Union

from block.UNetMidBlock2DCrossAttnWithFLM import UNetMidBlock2DCrossAttnWithFLM


class UNet2DConditionModelWithFLM(UNet2DConditionModel):
    def __init__(self):
        super().__init__(cross_attention_dim=768)

        # Replace mid_block with the custom block that includes FLM
        self.mid_block = UNetMidBlock2DCrossAttnWithFLM(
            in_channels=1280,
            temb_channels=1280,
            out_channels=1280,
            num_layers=1,  # Adjust the number of layers if needed
            num_attention_heads=8,
            cross_attention_dim=768,
            resnet_groups=32,
        )

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
            **kwargs
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # Center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # Get timestep embeddings
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        # Initial convolution layer
        sample = self.conv_in(sample)

        # Downsample with cross-attention in down blocks
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # Mid block with cross-attention and FLM
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        # Upsample with cross-attention
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )

        # Final post-process layers
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return UNet2DConditionOutput(sample=sample)


# # 定义输入参数
# batch_size = 1  # 批量大小
# in_channels = 4  # 输入通道数
# height = 512  # 输入高度
# width = 512  # 输入宽度
# timestep = 0  # 示例时间步
#
# # 创建模型实例
# model = UNet2DConditionModelWithFLM().to("cuda")
# # model_name = "benjamin-paine/stable-diffusion-v1-5"
# # pipe = StableDiffusionPipeline.from_pretrained(model_name)
# # model = pipe.unet  # 获取 UNet 模型
#
# # 准备输入数据
# sample_input = torch.randn(batch_size, in_channels, height, width).to("cuda", dtype=torch.float16)  # 随机生成输入张量
# t_emb = torch.tensor([timestep]).to("cuda", dtype=torch.float16)  # 将时间步转换为张量
#
# # 假设 sequence_length 为 10，特征维度为 1280
# sequence_length = 77
# encoder_hidden_states = torch.randn(batch_size, sequence_length, 768).to("cuda", dtype=torch.float16)  # 随机生成编码器隐藏状态
#
# # 前向传播
# output = model(sample_input, t_emb, encoder_hidden_states)
#
# # 输出结果
# print("Output shape:", output.sample.shape)
