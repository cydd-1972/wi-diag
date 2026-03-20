"""
ICA盲源分离模块
实现RobustICA算法分离多人步态信号
"""

import numpy as np
from scipy import linalg
from sklearn.decomposition import FastICA
from typing import Optional, Tuple


class RobustICA:
    """
    RobustICA算法实现
    基于论文[42]：通过峭度最大化进行复数信号分离
    """

    def __init__(self, n_components: int, max_iter: int = 1000,
                 tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None  # 解混矩阵

    def _kurtosis(self, y: np.ndarray) -> float:
        """计算峭度"""
        y = y - np.mean(y)
        return np.mean(np.abs(y) ** 4) / (np.mean(np.abs(y) ** 2) ** 2) - 2

    def _complex_kurtosis(self, y: np.ndarray) -> float:
        """复数信号的峭度"""
        y = y - np.mean(y)
        return np.mean(np.abs(y) ** 4) / (np.mean(np.abs(y) ** 2) ** 2) - 2

    def _gsd(self, w: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        广义峭度梯度下降步骤
        """
        y = w.conj() @ X
        # 计算梯度
        grad = np.mean(np.abs(y) ** 2 * y.conj() * X, axis=1) - \
               2 * w * np.mean(np.abs(y) ** 2)
        return grad

    def fit(self, X: np.ndarray):
        """
        拟合RobustICA模型

        参数:
            X: 混合信号矩阵 [特征数, 样本数] 或 [样本数, 特征数]
        """
        # 确保X是 [特征数, 样本数] 格式
        if X.shape[0] > X.shape[1]:
            X = X.T

        n_features, n_samples = X.shape

        # 白化预处理
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        cov = X_centered @ X_centered.conj().T / n_samples
        D, V = linalg.eigh(cov)

        # 白化矩阵
        whitening = np.diag(1.0 / np.sqrt(D + 1e-6)) @ V.conj().T
        X_white = whitening @ X_centered

        # 初始化解混矩阵
        W = np.eye(self.n_components, n_features, dtype=complex)

        # 迭代优化
        for iteration in range(self.max_iter):
            W_old = W.copy()

            # 对每个分量进行更新
            for i in range(self.n_components):
                w = W[i:i + 1, :].T

                # 计算梯度
                grad = self._gsd(w.flatten(), X_white)
                grad = grad.reshape(-1, 1)

                # 更新权重
                mu = 0.1  # 步长
                w_new = w + mu * grad

                # 正交化
                if i > 0:
                    w_new = w_new - W[:i, :].T @ (W[:i, :] @ w_new)
                w_new = w_new / np.linalg.norm(w_new)

                W[i:i + 1, :] = w_new.T

            # 检查收敛
            change = np.max(np.abs(np.abs(W) - np.abs(W_old)))
            if change < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break

        self.W = W @ whitening
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用解混矩阵"""
        if X.shape[0] > X.shape[1]:
            X = X.T
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        S = self.W @ X_centered
        return S.T  # 返回 [样本数, 源信号数]


class ConjugateMultiplier:
    """
    共轭相乘模块
    消除CSI相位误差，保持线性特性
    基于论文公式(10)
    """

    def __init__(self, config):
        self.config = config

    def conjugate_multiplication(self, antenna1: np.ndarray,
                                 antenna2: np.ndarray) -> np.ndarray:
        """
        两天线信号的共轭相乘
        论文公式(10): x_cm = x1 * conj(x2)

        参数:
            antenna1: 第一天线信号 [时间]
            antenna2: 第二天线信号 [时间]

        返回:
            处理后的线性组合信号
        """
        # 共轭相乘
        cm_result = antenna1 * np.conj(antenna2)

        # 减去静态分量（取均值）
        static_part = np.mean(cm_result)
        dynamic_part = cm_result - static_part

        return dynamic_part

    def process_all_pairs(self, csi_matrix: np.ndarray) -> np.ndarray:
        """
        处理所有天线对

        参数:
            csi_matrix: [时间, 天线对]

        返回:
            处理后的信号矩阵 [时间, 天线对组合数]
        """
        n_antennas = csi_matrix.shape[1]
        n_combinations = n_antennas * (n_antennas - 1) // 2
        processed = np.zeros((csi_matrix.shape[0], n_combinations),
                             dtype=complex)

        idx = 0
        for i in range(n_antennas):
            for j in range(i + 1, n_antennas):
                processed[:, idx] = self.conjugate_multiplication(
                    csi_matrix[:, i], csi_matrix[:, j]
                )
                idx += 1

        return processed


class GaitSeparator:
    """步态信号分离器"""

    def __init__(self, config):
        self.config = config
        self.conj_mult = ConjugateMultiplier(config)
        self.ica = None

    def separate_gaits(self, csi_data: np.ndarray,
                       n_subjects: int) -> np.ndarray:
        """
        分离多人步态信号

        流程:
        1. 共轭相乘消除相位误差
        2. ICA分离源信号
        """
        # 1. 共轭相乘处理
        print("Applying conjugate multiplication...")
        linear_signals = self.conj_mult.process_all_pairs(csi_data)

        # 取实部用于ICA（论文中使用幅值信息）
        linear_signals_real = np.real(linear_signals)

        # 2. ICA分离
        print(f"Separating {n_subjects} subjects using ICA...")

        # 尝试使用FastICA（如果可用）
        try:
            self.ica = FastICA(n_components=n_subjects,
                               max_iter=self.config.ICA_MAX_ITER,
                               tol=self.config.ICA_TOL)
            separated = self.ica.fit_transform(linear_signals_real)
            print("FastICA separation completed")

        except Exception as e:
            print(f"FastICA failed: {e}, falling back to RobustICA")
            # 使用自定义RobustICA
            self.ica = RobustICA(n_components=n_subjects,
                                 max_iter=self.config.ICA_MAX_ITER,
                                 tol=self.config.ICA_TOL)
            separated = self.ica.fit_transform(linear_signals_real)

        return separated

    def get_mixing_matrix(self) -> Optional[np.ndarray]:
        """获取混合矩阵A的估计"""
        if hasattr(self.ica, 'mixing_'):
            return self.ica.mixing_
        elif hasattr(self.ica, 'W'):
            return np.linalg.pinv(self.ica.W)
        return None