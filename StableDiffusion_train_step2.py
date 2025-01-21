import numpy as np
from diffusers.schedulers import KarrasDiffusionSchedulers
from torchmetrics.image import FrechetInceptionDistance, InceptionScore

from transformers import CLIPTextModel, CLIPTokenizer
from ultralytics import YOLO

import wandb
from accelerate.utils import ProjectConfiguration
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, AutoencoderKL, LMSDiscreteScheduler, \
    DPMSolverMultistepScheduler
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm  # 引入tqdm库以显示进度条
import warnings
from huggingface_hub import login
from block.UNet2DConditionModelWithFLM import UNet2DConditionModelWithFLM
from loss.BoneClassLoss import bone_age_loss
from loss.CountGradeConsistencyLoss import count_grade_consistency_loss
from utils.BoneClassProcessor import BoneClassProcessor

# 设置环境变量以允许多个OpenMP运行时库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["WANDB_MODE"] = "offline"

# login(token="*************")

os.environ["WANDB_API_KEY"] = "******************"
os.environ['YOLO_VERBOSE'] = 'False'
device = "cuda" if torch.cuda.is_available() else "cpu"

# 忽略 FutureWarning 和 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ScoreDataset(Dataset):
    def __init__(self, data_frame_path):
        self.annotations = pd.read_excel(data_frame_path)


    def __len__(self):
        """Return the total number of rows."""
        return len(self.annotations)

    def __getitem__(self, idx):
        rao_score = self.annotations.iloc[idx]["rao_score"]  # 获取文本报告
        zhang1_score = self.annotations.iloc[idx]["zhang1_score"]
        zhang3_score = self.annotations.iloc[idx]["zhang3_score"]
        zhang5_score = self.annotations.iloc[idx]["zhang5_score"]
        jin1_score = self.annotations.iloc[idx]["jin1_score"]
        jin3_score = self.annotations.iloc[idx]["jin3_score"]
        jin5_score = self.annotations.iloc[idx]["jin5_score"]
        zhong3_score = self.annotations.iloc[idx]["zhong3_score"]
        zhong5_score = self.annotations.iloc[idx]["zhong5_score"]
        yuan1_score = self.annotations.iloc[idx]["yuan1_score"]
        yuan3_score = self.annotations.iloc[idx]["yuan3_score"]
        yuan5_score = self.annotations.iloc[idx]["yuan5_score"]
        tou_score = self.annotations.iloc[idx]["tou_score"]
        gou_score = self.annotations.iloc[idx]["gou_score"]
        sex = self.annotations.iloc[idx]["性别"]
        boneAge = self.annotations.iloc[idx]["CHN骨龄"]

        report = (
            f"s{sex};b{boneAge};r{rao_score};"
            f"z{zhang1_score},{zhang3_score},{zhang5_score};"
            f"j{jin1_score},{jin3_score},{jin5_score};"
            f"zh{zhong3_score},{zhong5_score};"
            f"y{yuan1_score},{yuan3_score},{yuan5_score};"
            f"t{tou_score};g{gou_score}"
        )
        return report

if __name__ == '__main__':

    batch_size = 1
    learning_rate = 1e-5
    num_train_epochs = 10
    output_dir = "history_checkPoint/SD/sd-model-finetuned-step2-withFLM-BCE"
    weight_dtype = torch.float32
    model_name = "./local_model/stable-diffusion-v1-5"
    vae_checkpoint_path = "history_checkPoint/VAE/vae_kl_recg_percep_ssim_-11_epoch10/vae_last.pth"
    unet_checkpoint_path = "history_checkPoint/SD/sdWithFLMstep1/last_model.pth"
    file_path = 'unique_combinations.xlsx'

    wandb.init(
        project="StableDiffusion-Finetune",  # Your project name
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "model_name": "benjamin-paine/stable-diffusion-v1-5"
        }
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # Hardcoded value for gradient accumulation steps
        mixed_precision="no",  # Set to use fp16 precision bf16 no
        project_config=ProjectConfiguration(project_dir="output_dir", logging_dir="logs"),  # Example dirs
    )


    dataset = ScoreDataset(file_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=weight_dtype, safety_checker=None,
                                                   requires_safety_checker=False).to(device)

    # 加载自定义 VAE
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    state_dict = torch.load(vae_checkpoint_path, map_location=device)
    vae.load_state_dict(state_dict)
    pipe.vae = vae
    # pipe.vae.config.scaling_factor = 1.0
    # vae = pipe.vae

    # 初始化 U-Net 模型
    unet = UNet2DConditionModelWithFLM()
    saved_weights = torch.load(unet_checkpoint_path, weights_only=True)
    # 加载权重到模型
    unet.load_state_dict(saved_weights)
    pipe.unet = unet
    pipe.unet.config.sample_size = 64
    # unet = pipe.unet

    # 获取 pipeline 的文本编码器、tokenizer 和噪声调度器
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    # 优化器设置
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-3
    )
    # optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

    # 加速器准备
    tokenizer, text_encoder, vae, unet, dataloader, scheduler = accelerator.prepare(tokenizer,
                                                                                    text_encoder, vae,
                                                                                    unet,
                                                                                    dataloader,
                                                                                    scheduler)


    # 选择一个固定的随机种子以确保可重复性
    torch.manual_seed(37)
    np.random.seed(37)
    best_loss = float("inf")
    save_path_best = os.path.join(output_dir, "best_model.pth")  # 最优模型保存路径
    save_path_last = os.path.join(output_dir, "last_model.pth")  # 最后模型保存路径
    os.makedirs(output_dir, exist_ok=True)

    yolo_detect = YOLO("识别/runs/detect/train4/weights/best.pt")

    processor = BoneClassProcessor()

    bce_loss = torch.nn.BCELoss()

    total_loss = 0
    count=1
    # 读取execl
    df = pd.read_excel("骨骼等级不同组合提取.xlsx")
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs}", leave=False)
        for step, batch in enumerate(progress_bar):
            # 从dataloader中获取数据
            prompt = batch[0]
            with accelerator.autocast():
                # 生成图像
                image = pipe(prompt, num_inference_steps=50, progress_bar=False).images[0]
                result = yolo_detect(image, verbose=False)

                cls_indices = result[0].boxes.cls  # 类别索引
                confidences = result[0].boxes.conf  # 置信度

                # 创建一个128维的数组，遍历cls_indices与confidences，把confidence填入到cls_indice位置
                y_pred = torch.zeros(128, dtype=torch.float32, requires_grad=True).to(device)

                # 将confidences填入cls_indices对应的位置
                for cls, conf in zip(cls_indices, confidences):
                    y_pred[int(cls)] = conf

                y_true = processor(prompt).to(device)
                perceptual_loss = bce_loss(y_pred, y_true)


            accelerator.backward(perceptual_loss)
            optimizer.step()
            optimizer.zero_grad()
            # 进度条打印loss
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_train_epochs}, Loss: {perceptual_loss.item():.4f}")
            # 上传loss到wandb
            wandb.log({"perceptual_loss": perceptual_loss.item()})

            # 更新tqdm进度条
            progress_bar.set_postfix(loss=perceptual_loss.item())

            # 每100轮保存一次生成的图像
            if step % 100 == 0:
                image.save(os.path.join(output_dir, f"generated_image_{epoch + 1}_{step}.png"))
                wandb.log({f"Generated Image - {prompt}": wandb.Image(image)})

            # 保存最优模型
            if perceptual_loss.item() < best_loss:
                best_loss = perceptual_loss
                torch.save(unet.state_dict(), save_path_best)
        torch.save(unet.state_dict(), save_path_last)
    print("Last model saved!")
