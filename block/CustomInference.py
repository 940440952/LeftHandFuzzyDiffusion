from typing import Any, List, Optional, Union
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL


def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    设置调度器的时间步长，并返回时间步长。

    Args:
        scheduler: 调度器对象（如 DDIMScheduler 或其他）。
        num_inference_steps (int): 采样的步数。
        device (torch.device): 设备。

    Returns:
        torch.Tensor: 时间步长张量。
    """
    scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler.timesteps


def custom_inference(
        pipeline: StableDiffusionPipeline,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: Optional[torch.Generator] = None,
        return_latents: bool = False,
        return_tensors: bool = False,
) -> Union[List[Image.Image], List[torch.Tensor], torch.Tensor]:
    """
    自定义 Stable Diffusion 推理过程，支持用户自定义 VAE，并对生成过程进行了优化。

    Args:
        pipeline (StableDiffusionPipeline): 已初始化的 Stable Diffusion Pipeline。
        prompt (str or List[str]): 提示词，用于指导生成图像。
        vae (AutoencoderKL): 自定义的 VAE 模型。
        negative_prompt (str or List[str], optional): 负面提示词，用于抑制生成特定内容。
        num_inference_steps (int): 采样的步数。
        guidance_scale (float): 文本引导的强度。
        height (int): 生成图像的高度。
        width (int): 生成图像的宽度。
        generator (torch.Generator, optional): 随机数生成器，用于控制噪声。
        return_latents (bool): 是否返回潜在空间表示。
        return_tensors (bool): 是否返回图像张量。

    Returns:
        List[PIL.Image] or List[torch.Tensor] or torch.Tensor: 生成的图像。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 编码文本提示
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=(guidance_scale > 1),
    )


    # 2. 初始化潜在变量
    latents = pipeline.prepare_latents(
        batch_size=len(prompt) if isinstance(prompt, list) else 1,
        num_channels_latents=pipeline.unet.config.in_channels,
        height=height,
        width=width,
        dtype=pipeline.unet.dtype,
        device=device,
        generator=generator,
    )

    # 3. 设置时间步长
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 4. 推理步骤
    with torch.no_grad():
        for t in timesteps:
            latent_model_input = pipeline.scheduler.scale_model_input(latents, t).to(device)

            # 合并正负引导
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # 使用指导比例调整噪声预测
            # if guidance_scale > 1:
            #     noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]


    # 如果需要返回潜在空间
    if return_latents:
        return latents

    # 5. 使用自定义 VAE 解码生成图像
    with torch.no_grad():
        # 对解码后的图像进行缩放和转换
        reconstructed_images = pipeline.vae.decode(latents).sample
        if return_tensors:
            return reconstructed_images

        # 将图像从 [-1, 1] 转换到 [0, 1] 范围
        reconstructed_images = (reconstructed_images / 2 + 0.5).clamp(0, 1)
        reconstructed_images = reconstructed_images.cpu().permute(0, 2, 3, 1).numpy()

    # 返回 PIL 图像列表
    return [Image.fromarray((image * 255).astype("uint8")) for image in reconstructed_images]
