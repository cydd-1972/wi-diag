"""
工具函数模块
包含评估指标、可视化等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Tuple
import os


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """计算信噪比"""
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)


def plot_walking_detection(csi_data: np.ndarray,
                           detection_flags: np.ndarray,
                           walking_segments: List[Tuple],
                           save_path: str = None):
    """
    绘制行走检测结果
    论文图2样式
    """
    plt.figure(figsize=(12, 6))

    # 绘制CSI幅值
    time = np.arange(len(csi_data)) / 1000  # 转换为秒
    plt.plot(time, csi_data, 'b-', alpha=0.7, label='CSI Amplitude')

    # 标记检测到的行走区域
    for start, end in walking_segments:
        plt.axvspan(start / 1000, end / 1000, alpha=0.3, color='green',
                    label='Walking' if start == walking_segments[0][0] else '')

    # 标记检测标志
    detection_time = time[detection_flags]
    detection_values = csi_data[detection_flags]
    plt.scatter(detection_time, detection_values,
                c='red', s=10, label='Detection', zorder=5)

    plt.xlabel('Time (s)')
    plt.ylabel('CSI Amplitude Variance')
    plt.title('Walking Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ica_separation(original_mixed: np.ndarray,
                        separated_signals: np.ndarray,
                        save_path: str = None):
    """
    绘制ICA分离结果
    论文图5样式
    """
    n_subjects = separated_signals.shape[1]

    fig, axes = plt.subplots(2, max(2, n_subjects),
                             figsize=(4 * n_subjects, 8))

    # 绘制原始混合信号
    axes[0, 0].plot(original_mixed[:, 0], label='Antenna Pair 1')
    axes[0, 0].set_title('Mixed Signal - Antenna 1')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)

    axes[0, 1].plot(original_mixed[:, 1], label='Antenna Pair 2', color='orange')
    axes[0, 1].set_title('Mixed Signal - Antenna 2')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True)

    # 绘制分离后的信号
    for i in range(n_subjects):
        axes[1, i].plot(separated_signals[:, i])
        axes[1, i].set_title(f'Separated Subject {i + 1}')
        axes[1, i].set_xlabel('Time')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_spectrogram(spectrogram: np.ndarray,
                     title: str = 'Spectrogram',
                     save_path: str = None):
    """
    绘制频谱图
    论文图6样式
    """
    plt.figure(figsize=(10, 6))

    plt.imshow(spectrogram, aspect='auto', origin='lower',
               cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: List[str] = ['Normal', 'Abnormal'],
                          save_path: str = None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(train_losses: List[float],
                          val_accuracies: List[float] = None,
                          save_path: str = None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()

    # 准确率曲线
    if val_accuracies:
        ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True)
        ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison_with_baseline(results: dict,
                                  baseline_names: List[str],
                                  save_path: str = None):
    """
    绘制与基线系统的对比
    论文图11样式
    """
    plt.figure(figsize=(10, 6))

    x = np.arange(len(baseline_names))
    width = 0.35

    # 提取结果
    accuracies = [results[name] for name in baseline_names]

    bars = plt.bar(x, accuracies, width, label='Accuracy (%)')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.xlabel('System')
    plt.ylabel('Accuracy (%)')
    plt.title('Performance Comparison with Baseline Systems')
    plt.xticks(x, baseline_names)
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def export_results_to_csv(results: dict, filename: str):
    """导出结果到CSV"""
    import pandas as pd

    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")