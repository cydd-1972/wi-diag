"""
Wi-Diag主程序
整合所有模块，完成多目标异常步态诊断
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import argparse

from config import Config
from data_loader import CSIDataLoader
from preprocessing import CSIPreprocessor
from separation import GaitSeparator
from spectrogram import SpectrogramGenerator
from cyclegan import CycleGAN, SpectrogramDataset
from cnn_classifier import GaitDiagnosisModel
from utils import *


class WiDiag:
    """Wi-Diag系统主类"""

    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化各模块
        self.data_loader = CSIDataLoader(config)
        self.preprocessor = CSIPreprocessor(config)
        self.separator = GaitSeparator(config)
        self.spectrogram_gen = SpectrogramGenerator(config)
        self.cyclegan = None
        self.classifier = None

        # 创建模型保存目录
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def train_single_subject_model(self, csi_data_list, labels):
        """
        训练单目标诊断模型
        用于目标域数据
        """
        print("\n=== Training Single-Subject Diagnosis Model ===")

        # 预处理所有单目标数据
        all_spectrograms = []

        for i, csi_data in enumerate(csi_data_list):
            # 预处理
            processed, segments = self.preprocessor.preprocess_pipeline(
                csi_data, fit_pca=(i == 0)
            )

            # 只取行走段
            for start, end in segments:
                segment = processed[start:end, :]

                # 生成频谱图
                # 使用第一个天线对的数据
                spectrogram = self.spectrogram_gen.generate_spectrogram_pipeline(
                    segment[:, 0]
                )

                # 转换为RGB
                rgb_spec = self.spectrogram_gen.spectrogram_to_rgb(spectrogram)
                all_spectrograms.append(rgb_spec)

        # 转换为numpy数组
        X = np.array(all_spectrograms)
        y = np.array(labels)

        # 归一化
        X = X / 255.0

        # 调整维度 [N, H, W, C] -> [N, C, H, W]
        X = X.transpose(0, 3, 1, 2)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练分类器
        self.classifier = GaitDiagnosisModel(self.config, self.device)
        self.classifier.train(X_train, y_train, X_val, y_val)

        # 保存模型
        self.classifier.save_model(
            os.path.join(self.config.MODEL_DIR, 'single_subject_cnn.pth')
        )

        print("Single-subject model training completed!")

    def train_cyclegan(self, source_spectrograms, target_spectrograms):
        """
        训练CycleGAN进行域适配

        参数:
            source_spectrograms: 源域频谱图（多人分离后）
            target_spectrograms: 目标域频谱图（单目标）
        """
        print("\n=== Training CycleGAN for Domain Adaptation ===")

        # 创建数据集
        dataset_S = SpectrogramDataset(source_spectrograms)
        dataset_T = SpectrogramDataset(target_spectrograms)

        dataloader_S = torch.utils.data.DataLoader(
            dataset_S,
            batch_size=self.config.CYCLEGAN_BATCH_SIZE,
            shuffle=True
        )
        dataloader_T = torch.utils.data.DataLoader(
            dataset_T,
            batch_size=self.config.CYCLEGAN_BATCH_SIZE,
            shuffle=True
        )

        # 初始化CycleGAN
        self.cyclegan = CycleGAN(self.config, self.device)

        # 训练
        self.cyclegan.train(dataloader_S, dataloader_T)

        # 保存模型
        self.cyclegan.save_model('cyclegan_final.pth')

        print("CycleGAN training completed!")

    def process_multi_subject(self, csi_data, n_subjects):
        """
        处理多人数据

        流程:
        1. 预处理
        2. ICA分离
        3. 生成频谱图
        4. CycleGAN转换（如果有）
        """
        print(f"\n=== Processing {n_subjects}-Subject Data ===")

        # 1. 预处理
        processed, segments = self.preprocessor.preprocess_pipeline(
            csi_data, fit_pca=False
        )

        # 2. ICA分离
        separated = self.separator.separate_gaits(processed, n_subjects)

        # 3. 为每个分离信号生成频谱图
        separated_spectrograms = []
        for i in range(n_subjects):
            spec = self.spectrogram_gen.generate_spectrogram_pipeline(
                separated[:, i]
            )
            rgb_spec = self.spectrogram_gen.spectrogram_to_rgb(spec)
            separated_spectrograms.append(rgb_spec)

        # 4. 如果有CycleGAN，进行域转换
        if self.cyclegan is not None:
            print("Applying CycleGAN domain adaptation...")
            # 准备输入 [N, H, W, C]
            input_specs = np.array(separated_spectrograms)
            input_specs = input_specs / 255.0  # 归一化到[0,1]
            input_specs = input_specs.transpose(0, 3, 1, 2)  # [N, C, H, W]
            input_specs = (input_specs * 2) - 1  # 归一化到[-1,1]

            # 转换
            transformed = self.cyclegan.transform(input_specs)

            # 转回[0,1]范围
            transformed = (transformed + 1) / 2
            transformed = transformed.transpose(0, 2, 3, 1)  # [N, H, W, C]
            transformed = (transformed * 255).astype(np.uint8)

            return transformed
        else:
            return np.array(separated_spectrograms)

    def diagnose(self, spectrograms):
        """
        诊断步态

        返回:
            predictions: 预测类别 (0=正常, 1=异常)
            probabilities: 类别概率
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")

        # 预处理
        if isinstance(spectrograms, np.ndarray):
            # 归一化
            X = spectrograms / 255.0
            # 调整维度 [N, H, W, C] -> [N, C, H, W]
            if X.ndim == 3:
                X = X[np.newaxis, ...]
            X = X.transpose(0, 3, 1, 2)

        return self.classifier.predict(X)

    def run_experiment(self, experiment_type='single', **kwargs):
        """
        运行实验

        参数:
            experiment_type: 'single', 'multi', 'ablation'
        """
        if experiment_type == 'single':
            # 单目标实验
            csi_data_list = kwargs.get('csi_data_list', [])
            labels = kwargs.get('labels', [])
            self.train_single_subject_model(csi_data_list, labels)

        elif experiment_type == 'multi':
            # 多目标实验
            csi_data = kwargs.get('csi_data')
            n_subjects = kwargs.get('n_subjects', 2)
            ground_truth = kwargs.get('ground_truth', [])

            # 处理多人数据
            spectrograms = self.process_multi_subject(csi_data, n_subjects)

            # 诊断
            predictions, probabilities = self.diagnose(spectrograms)

            # 计算指标
            if ground_truth:
                metrics = self.classifier.calculate_metrics(
                    np.array(ground_truth), predictions
                )
                print(f"Diagnosis Results:")
                print(f"  Accuracy: {metrics['ACC'] * 100:.2f}%")
                print(f"  TPR: {metrics['TPR']:.3f}")
                print(f"  FPR: {metrics['FPR']:.3f}")
                return metrics

            return predictions, probabilities

        elif experiment_type == 'ablation':
            # 消融实验（评估ICA和CycleGAN的效果）
            results = {}

            # 1. Without ICA
            print("\n=== Ablation: Without ICA ===")
            # 直接使用混合信号
            mixed_specs = []
            # ... 处理代码

            # 2. Without CycleGAN
            print("\n=== Ablation: Without CycleGAN ===")
            # 使用分离但未转换的频谱图
            # ... 处理代码

            # 3. Full system
            print("\n=== Ablation: Full System ===")
            # 使用完整流程
            # ... 处理代码

            return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Wi-Diag: Multi-subject Abnormal Gait Diagnosis')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'train_single', 'train_cyclegan',
                                 'test_multi', 'ablation'],
                        help='运行模式')
    parser.add_argument('--n_subjects', type=int, default=2,
                        help='同时行走的人数')
    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 初始化系统
    system = WiDiag(config)

    if args.mode == 'demo':
        print("=" * 50)
        print("Wi-Diag: Multi-subject Abnormal Gait Diagnosis")
        print("=" * 50)

        # 生成模拟数据
        print("\nGenerating synthetic CSI data...")

        # 单目标数据（用于训练诊断模型）
        single_subject_data = []
        single_subject_labels = []
        for i in range(20):  # 20个样本
            csi, label = system.data_loader.generate_synthetic_csi(
                n_samples=5000, n_subjects=1
            )
            single_subject_data.append(csi)
            single_subject_labels.append(label[0])

        # 训练单目标模型
        system.train_single_subject_model(
            single_subject_data, single_subject_labels
        )

        # 生成多人数据用于测试
        print("\nGenerating multi-subject test data...")
        multi_csi, ground_truth = system.data_loader.generate_synthetic_csi(
            n_samples=5000, n_subjects=args.n_subjects
        )

        # 处理多人数据并诊断
        metrics = system.run_experiment(
            experiment_type='multi',
            csi_data=multi_csi,
            n_subjects=args.n_subjects,
            ground_truth=ground_truth
        )

        print("\n" + "=" * 50)
        print(f"Multi-subject ({args.n_subjects} subjects) diagnosis completed!")
        print("=" * 50)

    elif args.mode == 'train_single':
        print("Training single-subject model...")
        # 加载实际数据并训练
        # TODO: 加载真实CSI数据

    elif args.mode == 'train_cyclegan':
        print("Training CycleGAN for domain adaptation...")
        # 加载源域和目标域数据并训练CycleGAN
        # TODO: 加载真实频谱图数据

    elif args.mode == 'test_multi':
        print("Testing multi-subject diagnosis...")
        # 加载测试数据并评估
        # TODO: 加载真实多人数据

    elif args.mode == 'ablation':
        print("Running ablation experiments...")
        # 运行消融实验
        # TODO: 评估各组件效果


if __name__ == '__main__':
    main()