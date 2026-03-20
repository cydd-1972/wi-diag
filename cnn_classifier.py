"""
CNN分类器模块
用于步态正常/异常二分类
基于论文图9的网络结构
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm


class GaitCNN(nn.Module):
    """
    步态诊断CNN
    论文图9的网络结构
    """

    def __init__(self, input_size=(128, 256), num_classes=2, dropout=0.5):
        super(GaitCNN, self).__init__()

        # 卷积层
        self.conv_layers = nn.Sequential(
            # Conv1: 3×128×256 → 32×64×128
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv2: 32×64×128 → 64×32×64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv3: 64×32×64 → 128×16×32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv4: 128×16×32 → 256×8×16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 计算展平后的特征维度
        self._calculate_flatten_size(input_size)

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def _calculate_flatten_size(self, input_size):
        """计算展平后的特征维度"""
        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            x = self.conv_layers(x)
            self.flatten_size = x.numel()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class GaitDiagnosisModel:
    """步态诊断模型封装类"""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model = GaitCNN(
            input_size=config.SPECTROGRAM_SIZE,
            num_classes=2,
            dropout=config.CNN_DROPOUT
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.CNN_LR
        )

        self.train_losses = []
        self.val_accuracies = []

    def train(self, train_data, train_labels, val_data=None, val_labels=None):
        """
        训练CNN分类器

        参数:
            train_data: 训练数据 [N, C, H, W]
            train_labels: 训练标签 [N]
            val_data: 验证数据
            val_labels: 验证标签
        """
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data),
            torch.LongTensor(train_labels)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.CNN_BATCH_SIZE,
            shuffle=True
        )

        if val_data is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(val_data),
                torch.LongTensor(val_labels)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.CNN_BATCH_SIZE,
                shuffle=False
            )

        for epoch in range(self.config.CNN_EPOCHS):
            # 训练阶段
            self.model.train()
            epoch_loss = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.CNN_EPOCHS}')
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)

            # 验证阶段
            if val_data is not None:
                accuracy = self.evaluate(val_loader)
                self.val_accuracies.append(accuracy)
                print(f'Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, '
                      f'Val Accuracy = {accuracy:.2f}%')
            else:
                print(f'Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}')

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)

                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def predict(self, spectrograms):
        """
        预测步态类别

        返回:
            predictions: 预测类别 (0=正常, 1=异常)
            probabilities: 类别概率
        """
        self.model.eval()

        if isinstance(spectrograms, np.ndarray):
            spectrograms = torch.FloatTensor(spectrograms)

        spectrograms = spectrograms.to(self.device)

        with torch.no_grad():
            outputs = self.model(spectrograms)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def calculate_metrics(self, y_true, y_pred):
        """
        计算TPR, FPR, ACC
        论文公式(15)-(17)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # 真正率
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假正率
        acc = (tp + tn) / (tp + tn + fp + fn)  # 准确率

        return {
            'TPR': tpr,
            'FPR': fpr,
            'ACC': acc,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }

    def plot_roc_curve(self, y_true, y_scores):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        return roc_auc

    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }, filename)

    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_accuracies = checkpoint['val_accuracies']