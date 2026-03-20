"""
数据预处理模块
包含滤波、PCA降维、行走检测
"""

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


class CSIPreprocessor:
    """CSI数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.pca_models = {}  # 每个天线对的PCA模型
        self.noise_level = 0

    def butterworth_filter(self, csi_data: np.ndarray) -> np.ndarray:
        """
        巴特沃斯带通滤波器
        保留10-70Hz的步态频率成分
        """
        nyquist = self.config.FS / 2
        low = self.config.BUTTERWORTH_LOW / nyquist
        high = self.config.BUTTERWORTH_HIGH / nyquist

        # 设计带通滤波器
        b, a = signal.butter(self.config.BUTTERWORTH_ORDER,
                             [low, high],
                             btype='band')

        # 对每个天线对和子载波进行滤波
        filtered_data = np.zeros_like(csi_data, dtype=np.float64)

        # 处理实数部分和虚数部分
        for i in range(csi_data.shape[1]):  # 天线对
            for j in range(csi_data.shape[2]):  # 子载波
                # 取幅值（论文中使用幅值而非相位）
                magnitude = np.abs(csi_data[:, i, j])

                # 前向-后向滤波以消除相位延迟
                filtered = signal.filtfilt(b, a, magnitude)
                filtered_data[:, i, j] = filtered

        return filtered_data

    def pca_denoise(self, csi_data: np.ndarray,
                    antenna_pair: int, fit: bool = True) -> np.ndarray:
        """
        PCA降噪和降维
        利用子载波间的相关性去除带内噪声
        """
        # 重塑数据：[时间, 子载波]
        n_samples = csi_data.shape[0]
        data_reshaped = csi_data[:, antenna_pair, :]

        if fit:
            # 训练新的PCA模型
            pca = PCA(n_components=self.config.PCA_COMPONENTS)
            transformed = pca.fit_transform(data_reshaped)
            self.pca_models[antenna_pair] = pca
        else:
            # 使用已有的PCA模型
            pca = self.pca_models.get(antenna_pair)
            if pca is None:
                raise ValueError(f"No PCA model for antenna pair {antenna_pair}")
            transformed = pca.transform(data_reshaped)

        # 重构数据（可选）
        # reconstructed = pca.inverse_transform(transformed)

        return transformed.flatten()  # 返回一维主成分

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """
        归一化到[0,1]区间
        论文公式(1): y_j_i = (x_j_i - min(x_i)) / (max(x_i) - min(x_i))
        """
        normalized = np.zeros_like(data)
        for i in range(data.shape[1]):  # 对每个特征维度
            min_val = np.min(data[:, i])
            max_val = np.max(data[:, i])
            if max_val > min_val:
                normalized[:, i] = (data[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = data[:, i]
        return normalized

    def walking_detection(self, csi_pca: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        基于方差的行走检测
        论文公式(2): Lt = (1-β)Lt-1 + β × vart

        返回:
            - 检测标志数组
            - 行走区间列表 [(start1, end1), (start2, end2), ...]
        """
        n_samples = len(csi_pca)
        detection_flags = np.zeros(n_samples, dtype=bool)
        walking_segments = []

        # 计算滑动窗口方差
        window_size = self.config.WINDOW_SIZE
        beta = self.config.BETA
        threshold_multiplier = self.config.DETECTION_THRESHOLD

        # 初始化噪声水平
        noise_level = np.var(csi_pca[:window_size])

        in_walking = False
        start_idx = 0

        for i in range(window_size, n_samples):
            # 计算当前窗口方差
            window_data = csi_pca[i - window_size:i]
            var_current = np.var(window_data)

            # 更新噪声水平（指数移动平均）
            noise_level = (1 - beta) * noise_level + beta * var_current

            # 检测阈值
            threshold = threshold_multiplier * noise_level

            # 判断是否在行走
            if var_current > threshold:
                detection_flags[i] = True
                if not in_walking:
                    in_walking = True
                    start_idx = i - window_size // 2
            else:
                if in_walking:
                    in_walking = False
                    end_idx = i - window_size // 2
                    walking_segments.append((start_idx, end_idx))

        # 处理最后一个行走段
        if in_walking:
            walking_segments.append((start_idx, n_samples))

        return detection_flags, walking_segments

    def preprocess_pipeline(self, csi_data: np.ndarray,
                            fit_pca: bool = True) -> Tuple[np.ndarray, List]:
        """
        完整的预处理流程
        """
        # 1. 带通滤波
        print("Applying Butterworth filter...")
        filtered = self.butterworth_filter(csi_data)

        # 2. PCA降维
        print("Applying PCA denoising...")
        pca_results = []
        for ant in range(csi_data.shape[1]):
            pca_component = self.pca_denoise(filtered, ant, fit=fit_pca)
            pca_results.append(pca_component)

        pca_matrix = np.array(pca_results).T  # [时间, 天线对]

        # 3. 归一化
        print("Normalizing...")
        normalized = self.normalize(pca_matrix)

        # 4. 行走检测
        print("Detecting walking segments...")
        # 使用第一个天线对的PCA分量进行检测
        _, walking_segments = self.walking_detection(normalized[:, 0])

        return normalized, walking_segments