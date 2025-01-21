import os
import warnings
from PIL import Image
import cv2
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from torchvision import transforms
from ultralytics import YOLO

import wandb
from tqdm import tqdm

from block.UNet2DConditionModelWithFLM import UNet2DConditionModelWithFLM
from utils.BoneClassProcessor import BoneClassProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "local_model/stable-diffusion-v1-5"
vae_checkpoint_path = "history_checkPoint/VAE/vae_kl_recg_percep_ssim_-11_epoch10/vae_last.pth"
sd_checkpoint_path = "history_checkPoint/SD/sd-model-finetuned-step2-withFLM/last_model.pth"
df = pd.read_excel("unique_combinations.xlsx")
# 随机挑选出df中的一条
row = df.sample(1).iloc[0]
# 使用模型生成图片
# 将输入张量转换为 BFloat16
sex = row["性别"]
boneAge = row["CHN骨龄"]
rao_score = row["rao_score"]
zhang1_score = row["zhang1_score"]
zhang3_score = row["zhang3_score"]
zhang5_score = row["zhang5_score"]
jin1_score = row["jin1_score"]
jin3_score = row["jin3_score"]
jin5_score = row["jin5_score"]
zhong3_score = row["zhong3_score"]
zhong5_score = row["zhong5_score"]
yuan1_score = row["yuan1_score"]
yuan3_score = row["yuan3_score"]
yuan5_score = row["yuan5_score"]
tou_score = row["tou_score"]
gou_score = row["gou_score"]
prompt = f"s{sex};b{boneAge};r{rao_score};z{zhang1_score},{zhang3_score},{zhang5_score};j{jin1_score},{jin3_score},{jin5_score};zh{zhong3_score},{zhong5_score};y{yuan1_score},{yuan3_score},{yuan5_score};t{tou_score};g{gou_score}"

pipe = StableDiffusionPipeline.from_pretrained(model_name, safety_checker=None,
                                               requires_safety_checker=False).to(device)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
state_dict = torch.load(vae_checkpoint_path, map_location=device)
vae.load_state_dict(state_dict)
pipe.vae = vae

unet = UNet2DConditionModelWithFLM().to(device)
saved_weights = torch.load(sd_checkpoint_path, weights_only=True)
# 加载权重到模型
unet.load_state_dict(saved_weights)
pipe.unet = unet
pipe.unet.config.sample_size = 64


generated_image = pipe(prompt, num_inference_steps=50, progress_bar=False).images[0]
# 图片resize成1626*2032
generated_image = generated_image.resize((1626, 2032), resample=Image.BILINEAR)
generated_image.save("sample.png")