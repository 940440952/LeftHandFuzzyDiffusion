import numpy as np

import wandb
from accelerate.utils import ProjectConfiguration
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

from block.UNet2DConditionModelWithFLM import UNet2DConditionModelWithFLM


# 设置环境变量以允许多个OpenMP运行时库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["WANDB_MODE"] = "offline"

# login(token="********************")

os.environ["WANDB_API_KEY"] = "**********************"
os.environ['YOLO_VERBOSE'] = 'False'
device = "cuda" if torch.cuda.is_available() else "cpu"


class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, annotations_file, transform=None):
        """
        Args:
            image_dir (str): 存放医学图像的文件夹路径

            
            annotations_file (str): 包含图像路径和对应文本报告的CSV文件路径
            transform (callable, optional): 预处理操作，默认是None
        """
        self.image_dir = image_dir
        self.annotations = pd.read_excel(annotations_file)
        self.transform = transform

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """根据索引获取图像及其对应的文本描述"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径和对应的文本描述
        img_name = os.path.join(self.image_dir, str(self.annotations.iloc[idx]["序号"]) + ".jpg")  # 图像文件路径
        image = Image.open(img_name).convert("RGB")  # 打开图像并转换为RGB格式

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
        age = self.annotations.iloc[idx]["年龄"]
        weight = self.annotations.iloc[idx]["体重"]
        sex = self.annotations.iloc[idx]["性别"]
        height = self.annotations.iloc[idx]["身高"]
        boneAge = self.annotations.iloc[idx]["CHN骨龄"]

        # report = f"age_{age} sex_{sex} weight_{weight} height_{height} rao_{rao_score} zhang_{zhang1_score}_{zhang3_score}_{zhang5_score} jin_{jin1_score}_{jin3_score}_{jin5_score} zhong_{zhong3_score}_{zhong5_score} yuan_{yuan1_score}_{yuan3_score}_{yuan5_score} tou_{tou_score} gou_{gou_score}"
        report = (
            f"s{sex};b{boneAge};r{rao_score};"
            f"z{zhang1_score},{zhang3_score},{zhang5_score};"
            f"j{jin1_score},{jin3_score},{jin5_score};"
            f"zh{zhong3_score},{zhong5_score};"
            f"y{yuan1_score},{yuan3_score},{yuan5_score};"
            f"t{tou_score};g{gou_score}"
        )
        # 如果定义了transform，则对图像进行预处理
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'text': report,
                  'score': [rao_score, zhang1_score, zhang3_score, zhang5_score, jin1_score, jin3_score, jin5_score,
                            zhong3_score, zhong5_score, yuan1_score, yuan3_score, yuan5_score, tou_score, gou_score]}

        return sample


# 忽略 FutureWarning 和 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    batch_size = 1
    learning_rate = 1e-5
    num_train_epochs = 20
    output_dir = r"history_checkPoint/SD/sd-model-finetuned"
    weight_dtype = torch.float32
    model_name = "./local_model/stable-diffusion-v1-5"
    vae_checkpoint_path = "history_checkPoint/VAE/fp32_kl_recg_epoch10/vae_last.pth"

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
        mixed_precision="fp16",  # Set to use fp16 precision bf16 no
        project_config=ProjectConfiguration(project_dir="output_dir", logging_dir="logs"),  # Example dirs
    )

    # 设置图像的预处理操作
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 创建数据集和数据加载器
    dataset = MedicalImageDataset(image_dir='blackBackgroundData', annotations_file='20241207整理.xlsx',
                                  transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 使用 benjamin-paine/stable-diffusion-v1-5 加载 Stable Diffusion 管道
    # 配置参数

    # 加载 Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=weight_dtype, safety_checker=None,
                                                   requires_safety_checker=False).to(device)

    # 加载自定义 VAE
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    state_dict = torch.load(vae_checkpoint_path, map_location=device)
    vae.load_state_dict(state_dict)
    pipe.vae = vae

    # vae = pipe.vae

    # 初始化 U-Net 模型
    unet = UNet2DConditionModelWithFLM()
    saved_weights = torch.load("stable_diffusion_model_without_flm.pt", weights_only=True)
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
    tokenizer, text_encoder, vae, unet, train_dataloader, scheduler,optimizer = accelerator.prepare(tokenizer,
                                                                                          text_encoder, vae,
                                                                                          unet,
                                                                                          train_dataloader,
                                                                                          scheduler,optimizer)

    unique_combinations = pd.read_excel('unique_combinations.xlsx')



    torch.manual_seed(37)
    np.random.seed(37)
    best_loss = float("inf")
    save_path_best = os.path.join(output_dir, "best_model.pth")  # 最优模型保存路径
    save_path_last = os.path.join(output_dir, "last_model.pth")  # 最后模型保存路径
    save_path_generate = os.path.join(output_dir, "generate_model")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_path_generate, exist_ok=True)

    # df = pd.read_excel("骨骼等级不同组合提取.xlsx")

    for epoch in range(num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs}", leave=False)
        unet.train()
        epoch_loss = 0.0
        for step, batch in enumerate(progress_bar):
            images = batch["image"]
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],)).long()
            timesteps = timesteps.to(accelerator.device)

            noisy_latents = scheduler.add_noise(latents, noise,
                                                timesteps)

            text_inputs = tokenizer(
                batch["text"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                return_tensors="pt"
            ).to(accelerator.device)

            encoder_hidden_states = text_encoder(text_inputs.input_ids, return_dict=False)[0]
            with accelerator.autocast():
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                mse_loss = F.mse_loss(model_pred, noise, reduction="mean")

            wandb.log({"loss": mse_loss.item()})

            accelerator.backward(mse_loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += mse_loss.item()

            progress_bar.set_postfix(loss=mse_loss.item())

        epoch_loss /= len(train_dataloader)  # 计算平均训练损失

        # 检查并保存最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(accelerator.unwrap_model(unet).state_dict(), save_path_best)

        # 切换到评估模式
        unet.eval()

        # 随机选取 10 种不同的组合
        random_combinations = unique_combinations.sample(10)
        fixed_prompts = [
        ]

        for _, row in random_combinations.iterrows():
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

            prompt = report = (
                f"s{sex};b{boneAge};r{rao_score};"
                f"z{zhang1_score},{zhang3_score},{zhang5_score};"
                f"j{jin1_score},{jin3_score},{jin5_score};"
                f"zh{zhong3_score},{zhong5_score};"
                f"y{yuan1_score},{yuan3_score},{yuan5_score};"
                f"t{tou_score};g{gou_score}"
            )
            fixed_prompts.append(prompt)

        with torch.no_grad():
            for prompt in fixed_prompts:
                generated_image = pipe(prompt, num_inference_steps=50).images[0]
                # 保存生成的图像
                generated_image.save(os.path.join(save_path_generate, f"epoch_{epoch + 1}_{prompt}.png"))
                wandb.log({prompt: wandb.Image(generated_image)})
        # 切换回训练模式
        unet.train()
        # 重置评估器以便

    accelerator.end_training()

    # 保存微调后的模型
    torch.save(accelerator.unwrap_model(unet).state_dict(), save_path_last)

    wandb.finish()
