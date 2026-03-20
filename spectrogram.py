"""
频谱图生成模块
STFT变换 + 频谱增强
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import cv2
from typing import Tuple, List


class SpectrogramGenerator:
    """频谱图生成器"""

    def __init__(self, config):
        self.config = config

    def stft_transform(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        短时傅里叶变换
        论文使用窗口大小1024，步长1
        """
        f, t, Zxx = signal.stft(signal_data,
                                fs=self.config.FS,
                                window='hann',
                                nperseg=self.config.STFT_WINDOW,
                                noverlap=self.config.STFT_WINDOW - self.config.STFT_HOP)
        return f, t, Zxx

    def energy_normalization(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        能量归一化
        计算前100个样本的总能量并归一化
        """
        # 计算前100个样本的能量
        energy = np.sum(np.abs(spectrogram[:, :100]))

        if energy > 0:
            normalized = np.abs(spectrogram) / energy
        else:
            normalized = np.abs(spectrogram)

        return normalized

    def remove_silence(self, spectrograms: List[np.ndarray],
                       threshold: float = 0.01) -> List[np.ndarray]:
        """
        删除静音区间
        能量低于阈值的频谱图被丢弃
        """
        valid_spectrograms = []

        for spec in spectrograms:
            energy = np.sum(np.abs(spec))
            if energy > threshold:
                valid_spectrograms.append(spec)

        print(f"Removed {len(spectrograms) - len(valid_spectrograms)} silent segments")
        return valid_spectrograms

    def noise_reduction(self, spectrogram: np.ndarray,
                        noise_estimate_time: int = 50) -> np.ndarray:
        """
        频域降噪
        减去短期平均噪声估计
        """
        # 估计噪声基底（取前noise_estimate_time个时间片的平均）
        noise_floor = np.mean(np.abs(spectrogram[:, :noise_estimate_time]),
                              axis=1, keepdims=True)

        # 减去噪声
        denoised = np.abs(spectrogram) - noise_floor
        denoised[denoised < 0] = 0

        return denoised

    def gaussian_smooth(self, spectrogram: np.ndarray,
                        sigma: float = 0.8, size: int = 5) -> np.ndarray:
        """
        高斯低通滤波
        论文使用size=5, σ=0.8
        """
        smoothed = gaussian_filter(spectrogram, sigma=sigma)
        return smoothed

    def resize_spectrogram(self, spectrogram: np.ndarray,
                           target_size: Tuple[int, int]) -> np.ndarray:
        """
        调整频谱图到目标尺寸
        论文使用128×256
        """
        # 使用OpenCV进行缩放
        if spectrogram.ndim == 2:
            # 单通道
            resized = cv2.resize(spectrogram, target_size[::-1],
                                 interpolation=cv2.INTER_LINEAR)
        else:
            # 多通道
            resized = np.zeros((target_size[0], target_size[1], spectrogram.shape[2]))
            for i in range(spectrogram.shape[2]):
                resized[:, :, i] = cv2.resize(spectrogram[:, :, i],
                                              target_size[::-1],
                                              interpolation=cv2.INTER_LINEAR)

        return resized

    def generate_spectrogram_pipeline(self, separated_signal: np.ndarray) -> np.ndarray:
        """
        完整的频谱图生成流程

        步骤:
        1. STFT变换
        2. 能量归一化
        3. 降噪
        4. 高斯平滑
        5. 尺寸调整
        """
        # 1. STFT
        f, t, Zxx = self.stft_transform(separated_signal)
        spectrogram = Zxx

        # 2. 能量归一化
        normalized = self.energy_normalization(spectrogram)

        # 3. 降噪
        denoised = self.noise_reduction(normalized)

        # 4. 高斯平滑
        smoothed = self.gaussian_smooth(denoised)

        # 5. 调整尺寸
        resized = self.resize_spectrogram(smoothed, self.config.SPECTROGRAM_SIZE)

        return resized

    def generate_multi_subject_spectrograms(self, separated_signals: np.ndarray) -> List[np.ndarray]:
        """
        为多个分离出的信号生成频谱图
        """
        spectrograms = []

        for i in range(separated_signals.shape[1]):
            spec = self.generate_spectrogram_pipeline(separated_signals[:, i])
            spectrograms.append(spec)

        return spectrograms

    def spectrogram_to_rgb(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        将灰度频谱图转换为RGB图像
        用于CycleGAN输入
        """
        # 归一化到0-255
        normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)

        # 转换为3通道
        rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        return rgb