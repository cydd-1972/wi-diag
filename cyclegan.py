"""
CycleGAN环境适配模块
实现无配对图像转换，消除环境依赖
基于论文[44]架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN生成器
    论文使用的架构：3卷积 + 3残差块 + 2反卷积
    """

    def __init__(self, in_channels=3, out_channels=3, n_residual=3):
        super(Generator, self).__init__()

        # 下采样
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 残差块
        self.residual = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_residual)]
        )

        # 上采样
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.residual(x)
        x = self.up(x)
        return x


class Discriminator(nn.Module):
    """
    判别器 - PatchGAN结构
    70×70 PatchGANs [50]
    """

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class CycleGAN:
    """
    CycleGAN完整实现
    论文公式(11)-(14)
    """

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 初始化生成器和判别器
        self.G = Generator().to(self.device)  # S → T
        self.F = Generator().to(self.device)  # T → S
        self.D_S = Discriminator().to(self.device)  # 源域判别器
        self.D_T = Discriminator().to(self.device)  # 目标域判别器

        # 优化器
        self.optimizer_G = optim.Adam(
            list(self.G.parameters()) + list(self.F.parameters()),
            lr=config.CYCLEGAN_LR, betas=(0.5, 0.999)
        )
        self.optimizer_D_S = optim.Adam(
            self.D_S.parameters(), lr=config.CYCLEGAN_LR, betas=(0.5, 0.999)
        )
        self.optimizer_D_T = optim.Adam(
            self.D_T.parameters(), lr=config.CYCLEGAN_LR, betas=(0.5, 0.999)
        )

        # 损失函数
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # 记录损失
        self.losses = {'G': [], 'D_S': [], 'D_T': [], 'cycle': []}

    def train(self, dataloader_S, dataloader_T, epochs=None):
        """
        训练CycleGAN

        参数:
            dataloader_S: 源域数据加载器（多人分离频谱图）
            dataloader_T: 目标域数据加载器（单目标频谱图）
        """
        if epochs is None:
            epochs = self.config.CYCLEGAN_EPOCHS

        lambda_cycle = self.config.CYCLEGAN_LAMBDA
        lambda_identity = 0.5 * lambda_cycle

        for epoch in range(epochs):
            epoch_loss_G = 0
            epoch_loss_D_S = 0
            epoch_loss_D_T = 0
            epoch_loss_cycle = 0
            n_batches = 0

            pbar = tqdm(zip(dataloader_S, dataloader_T),
                        desc=f'Epoch {epoch + 1}/{epochs}')

            for real_S, real_T in pbar:
                real_S = real_S.to(self.device)
                real_T = real_T.to(self.device)

                batch_size = real_S.size(0)

                # 创建标签
                valid = torch.ones((batch_size, 1, 30, 30),
                                   device=self.device, requires_grad=False)
                fake = torch.zeros((batch_size, 1, 30, 30),
                                   device=self.device, requires_grad=False)

                # ------------------
                # 训练生成器
                # ------------------
                self.optimizer_G.zero_grad()

                # 恒等映射损失 (可选)
                loss_id_S = self.criterion_identity(self.G(real_S), real_S)
                loss_id_T = self.criterion_identity(self.F(real_T), real_T)
                loss_identity = (loss_id_S + loss_id_T) / 2

                # 对抗损失
                fake_T = self.G(real_S)
                loss_GAN_S2T = self.criterion_gan(self.D_T(fake_T), valid)

                fake_S = self.F(real_T)
                loss_GAN_T2S = self.criterion_gan(self.D_S(fake_S), valid)

                loss_GAN = (loss_GAN_S2T + loss_GAN_T2S) / 2

                # 循环一致性损失
                recovered_S = self.F(fake_T)
                loss_cycle_S = self.criterion_cycle(recovered_S, real_S)

                recovered_T = self.G(fake_S)
                loss_cycle_T = self.criterion_cycle(recovered_T, real_T)

                loss_cycle = (loss_cycle_S + loss_cycle_T) / 2

                # 总生成器损失
                loss_G = loss_GAN + lambda_cycle * loss_cycle + lambda_identity * loss_identity

                loss_G.backward()
                self.optimizer_G.step()

                # ------------------
                # 训练判别器 D_S
                # ------------------
                self.optimizer_D_S.zero_grad()

                loss_real = self.criterion_gan(self.D_S(real_S), valid)
                loss_fake = self.criterion_gan(self.D_S(fake_S.detach()), fake)
                loss_D_S = (loss_real + loss_fake) / 2

                loss_D_S.backward()
                self.optimizer_D_S.step()

                # ------------------
                # 训练判别器 D_T
                # ------------------
                self.optimizer_D_T.zero_grad()

                loss_real = self.criterion_gan(self.D_T(real_T), valid)
                loss_fake = self.criterion_gan(self.D_T(fake_T.detach()), fake)
                loss_D_T = (loss_real + loss_fake) / 2

                loss_D_T.backward()
                self.optimizer_D_T.step()

                # 记录损失
                epoch_loss_G += loss_G.item()
                epoch_loss_D_S += loss_D_S.item()
                epoch_loss_D_T += loss_D_T.item()
                epoch_loss_cycle += loss_cycle.item()
                n_batches += 1

                pbar.set_postfix({
                    'G': f'{loss_G.item():.4f}',
                    'D_S': f'{loss_D_S.item():.4f}',
                    'D_T': f'{loss_D_T.item():.4f}'
                })

            # 记录epoch平均损失
            self.losses['G'].append(epoch_loss_G / n_batches)
            self.losses['D_S'].append(epoch_loss_D_S / n_batches)
            self.losses['D_T'].append(epoch_loss_D_T / n_batches)
            self.losses['cycle'].append(epoch_loss_cycle / n_batches)

            # 每10个epoch保存模型
            if (epoch + 1) % 10 == 0:
                self.save_model(f'cyclegan_epoch_{epoch + 1}.pth')

    def transform(self, source_images):
        """
        将源域图像转换为目标域
        """
        self.G.eval()
        with torch.no_grad():
            if isinstance(source_images, np.ndarray):
                source_images = torch.FloatTensor(source_images)
            source_images = source_images.to(self.device)
            target_images = self.G(source_images)
        return target_images.cpu().numpy()

    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'G': self.G.state_dict(),
            'F': self.F.state_dict(),
            'D_S': self.D_S.state_dict(),
            'D_T': self.D_T.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_S': self.optimizer_D_S.state_dict(),
            'optimizer_D_T': self.optimizer_D_T.state_dict(),
            'losses': self.losses
        }, os.path.join(self.config.MODEL_DIR, filename))

    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(os.path.join(self.config.MODEL_DIR, filename))
        self.G.load_state_dict(checkpoint['G'])
        self.F.load_state_dict(checkpoint['F'])
        self.D_S.load_state_dict(checkpoint['D_S'])
        self.D_T.load_state_dict(checkpoint['D_T'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D_S.load_state_dict(checkpoint['optimizer_D_S'])
        self.optimizer_D_T.load_state_dict(checkpoint['optimizer_D_T'])
        self.losses = checkpoint['losses']


class SpectrogramDataset(Dataset):
    """频谱图数据集"""

    def __init__(self, spectrograms, transform=None):
        self.spectrograms = spectrograms
        self.transform = transform

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        img = self.spectrograms[idx]

        # 确保是3通道图像
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = img.transpose(2, 0, 1)

        # 归一化到[-1, 1] (Tanh输出范围)
        img = (img / 255.0) * 2 - 1

        return torch.FloatTensor(img)