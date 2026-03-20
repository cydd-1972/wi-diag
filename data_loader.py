"""
数据加载模块
处理CSI原始数据加载和存储
"""

import numpy as np
import h5py
import os
from typing import List, Tuple, Dict
import pickle
from tqdm import tqdm


class CSIDataLoader:
    """CSI数据加载器"""

    def __init__(self, config):
        self.config = config
        self.data = {}
        self.labels = {}

    def load_csi_from_h5(self, file_path: str) -> np.ndarray:
        """
        从H5文件加载CSI数据
        论文使用Intel 5300 NIC，输出格式为复数CSI
        """
        with h5py.File(file_path, 'r') as f:
            # 假设数据结构：csi_data shape = [时间, 天线对, 子载波]
            csi_data = f['csi'][:]

            # Intel 5300输出为30个子载波的复数形式
            if csi_data.dtype == np.complex64 or csi_data.dtype == np.complex128:
                return csi_data
            else:
                # 如果是实数格式，需要转换为复数
                # 假设数据格式：[实部, 虚部, 实部, 虚部, ...]
                csi_complex = csi_data[..., 0] + 1j * csi_data[..., 1]
                return csi_complex

    def generate_synthetic_csi(self, n_samples: int = 1000,
                               n_subjects: int = 1) -> Tuple[np.ndarray, List[int]]:
        """
        生成合成CSI数据用于测试
        基于论文的数学模型：x(t) = A·s(t) + n
        """
        # 生成独立源信号（每个目标的步态信号）
        t = np.arange(n_samples) / self.config.FS
        sources = []

        for i in range(n_subjects):
            # 步态频率在20-40Hz之间
            gait_freq = np.random.uniform(20, 40)
            # 添加谐波分量
            signal = (np.sin(2 * np.pi * gait_freq * t) +
                      0.3 * np.sin(4 * np.pi * gait_freq * t) +
                      0.1 * np.sin(6 * np.pi * gait_freq * t))
            # 添加随机相位偏移
            phase = np.random.uniform(0, 2 * np.pi)
            signal = np.roll(signal, int(phase * n_samples / (2 * np.pi)))
            sources.append(signal)

        sources = np.array(sources).T  # shape: [时间, 源信号]

        # 生成混合矩阵 A (随机但保持非奇异)
        # 每个天线对看到的是不同线性组合
        n_antennas = self.config.N_ANTENNAS_TX * self.config.N_ANTENNAS_RX
        A = np.random.randn(n_antennas, n_subjects) + \
            0.1j * np.random.randn(n_antennas, n_subjects)

        # 线性混合
        mixed = sources @ A.T  # shape: [时间, 天线对]

        # 添加噪声（符合论文公式4）
        noise_level = 0.05
        noise = noise_level * (np.random.randn(*mixed.shape) +
                               1j * np.random.randn(*mixed.shape))
        mixed += noise

        # 扩展子载波维度（每个天线对30个子载波）
        csi_data = np.zeros((n_samples, n_antennas,
                             self.config.N_SUBCARRIERS), dtype=complex)
        for i in range(self.config.N_SUBCARRIERS):
            # 子载波间有轻微频率选择性衰落
            fading = np.exp(1j * np.random.uniform(0, 2 * np.pi))
            csi_data[:, :, i] = mixed * fading

        # 生成标签（正常=0，异常=1）
        labels = np.random.randint(0, 2, n_subjects).tolist()

        return csi_data, labels

    def save_csi_data(self, csi_data: np.ndarray, labels: List[int],
                      file_path: str, metadata: Dict = None):
        """保存CSI数据"""
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('csi', data=csi_data, compression='gzip')
            f.create_dataset('labels', data=np.array(labels))
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value

    def load_dataset(self, data_dir: str) -> Dict:
        """加载整个数据集"""
        dataset = {'train': [], 'test': [], 'labels': []}

        for file_name in os.listdir(data_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(data_dir, file_name)
                with h5py.File(file_path, 'r') as f:
                    csi_data = f['csi'][:]
                    labels = f['labels'][:]

                if 'train' in file_name:
                    dataset['train'].append(csi_data)
                    dataset['labels'].extend(labels)
                elif 'test' in file_name:
                    dataset['test'].append(csi_data)

        return dataset