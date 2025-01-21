import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from transformers import get_scheduler
import wandb
from accelerate import Accelerator
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
import os
import pandas as pd
from PIL import Image
from torchvision.utils import make_grid
from loss.MultiScalePerceptualLoss import MultiScalePerceptualLoss
import os
from huggingface_hub import login
from skimage.metrics import structural_similarity as ssim

# 使用生成的 Access Token 登录
login(token="******************")

os.environ["WANDB_API_KEY"] = "***********************"


# os.environ["WANDB_MODE"] = "offline"

# Dataset definition
class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, annotations_file, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_excel(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, str(self.annotations.iloc[idx]["序号"]) + ".jpg")
        image = Image.open(img_name).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return {"image": image}


def ssim_loss(pred, target):
    # 将张量转换为NumPy数组
    pred = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    target = target.cpu().detach().numpy().transpose(0, 2, 3, 1)

    # 计算每对图像的SSIM
    ssim_values = []
    for i in range(pred.shape[0]):
        ssim_value = ssim(pred[i], target[i], multichannel=True, channel_axis=2, data_range=1)
        ssim_values.append(ssim_value)

    # 计算SSIM损失：1 - ssim，确保损失越小表示两张图像越相似
    return torch.tensor(1.0 - torch.tensor(ssim_values).mean())


if __name__ == '__main__':

    # Initialize W&B for logging
    wandb.init(project="VAE_Training")

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5

    # Initialize Accelerator for mixed-precision training
    accelerator = Accelerator()

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Compatible image size for VAE
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset and DataLoader
    dataset = MedicalImageDataset(image_dir="blackBackgroundData", annotations_file="20241207整理.xlsx",
                                  transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Load Stable Diffusion and extract VAE
    model_name = "local_model/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")

    # Optimizer and learning rate scheduler
    optimizer = AdamW(vae.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval()  # 使用VGG-19的前16层
    for param in vgg.parameters():
        param.requires_grad = False  # 冻结VGG网络的参数
    vgg = vgg

    # 定义多尺度感知损失
    perceptual_loss_fn = MultiScalePerceptualLoss(vgg)

    # Prepare model and data for Accelerator
    vae, optimizer, train_dataloader, lr_scheduler, vgg, perceptual_loss_fn = accelerator.prepare(vae, optimizer,
                                                                                                  train_dataloader,
                                                                                                  lr_scheduler, vgg,
                                                                                                  perceptual_loss_fn)

    # 保存模型的目录
    save_dir = "vae_checkpoints"
    pic_dir = os.path.join(save_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    best_loss = float('inf')  # 用于跟踪最优模型

    last_reconstructed_images = None
    last_images = None
    # Training loop
    for epoch in range(EPOCHS):
        vae.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}",
                            disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            images = batch["image"]

            with accelerator.autocast():
                # Encode and decode using the VAE
                latents = vae.encode(images).latent_dist  # Latent space sampling
                output = vae.decode(latents.sample())
                reconstructed_images = output.sample

                # 保存最后一个 batch 的图像
                last_reconstructed_images = reconstructed_images
                last_images = images

                # Reconstruction Loss
                recon_loss = torch.nn.functional.mse_loss(reconstructed_images, images, reduction="mean")

                # KL Divergence Loss
                kl_loss = -0.5 * torch.mean(1 + latents.logvar - latents.mean.pow(2) - latents.logvar.exp())

                perceptual_loss = perceptual_loss_fn(reconstructed_images, images)

                # ssim损失
                ssim_loss_value = ssim_loss(reconstructed_images, images)

                # Total loss
                loss = recon_loss + 0.001 * kl_loss + 0.1 * perceptual_loss + 0.1 * ssim_loss_value
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            # Log the batch loss to WandB
            wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})

            # Progress bar update
            progress_bar.set_postfix({"loss": loss.item()})

        # 平均损失用于选择最优模型
        avg_loss = total_loss / len(train_dataloader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            accelerator.wait_for_everyone()
            unwrapped_vae = accelerator.unwrap_model(vae)
            torch.save(unwrapped_vae.state_dict(), os.path.join(save_dir, "vae_best.pth"))

        # 上传训练和生成的图像
        accelerator.wait_for_everyone()
        unwrapped_vae = accelerator.unwrap_model(vae)
        with torch.no_grad():
            recon_grid = make_grid(last_reconstructed_images[:8].cpu(), nrow=4, normalize=True, value_range=(-1, 1))
            input_grid = make_grid(last_images[:8].cpu(), nrow=4, normalize=True, value_range=(-1, 1))

            # 将图像保存到磁盘
            recon_path = os.path.join(pic_dir, f"reconstructed_epoch_{epoch + 1}.png")
            input_path = os.path.join(pic_dir, f"input_epoch_{epoch + 1}.png")
            recon_grid = recon_grid.float()
            input_grid = input_grid.float()
            transforms.ToPILImage()(recon_grid).save(recon_path)
            transforms.ToPILImage()(input_grid).save(input_path)

            # 上传到 W&B
            wandb.log({
                "Reconstructed Images": wandb.Image(recon_path, caption=f"Reconstructed Epoch {epoch + 1}"),
                "Input Images": wandb.Image(input_path, caption=f"Input Epoch {epoch + 1}")
            })
    unwrapped_vae = accelerator.unwrap_model(vae)
    # 保存最后一个模型
    torch.save(unwrapped_vae.state_dict(), os.path.join(save_dir, "vae_last.pth"))

    # 结束W&B记录
    wandb.finish()
