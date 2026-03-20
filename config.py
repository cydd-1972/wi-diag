"""
Wi-Diag系统配置文件
包含所有可调节参数
"""


class Config:
    # 数据采集参数
    FS = 1000  # 采样频率 1000Hz
    CARRIER_FREQ = 5.825e9  # 5.825 GHz
    N_SUBCARRIERS = 30  # 每个天线对的子载波数
    N_ANTENNAS_TX = 1  # 发射天线数
    N_ANTENNAS_RX = 3  # 接收天线数

    # 预处理参数
    BUTTERWORTH_LOW = 10  # 带通滤波器下限 (Hz)
    BUTTERWORTH_HIGH = 70  # 带通滤波器上限 (Hz)
    BUTTERWORTH_ORDER = 4  # 滤波器阶数

    # PCA参数
    PCA_COMPONENTS = 1  # 保留的主成分数

    # 行走检测参数
    WINDOW_SIZE = 200  # 滑动窗口大小
    BETA = 0.1  # 指数移动平均系数
    DETECTION_THRESHOLD = 4.0  # 检测阈值倍数

    # ICA分离参数
    ICA_MAX_ITER = 1000
    ICA_TOL = 1e-4

    # STFT参数
    STFT_WINDOW = 1024  # FFT窗口大小
    STFT_HOP = 1  # 滑动步长
    SPECTROGRAM_SIZE = (128, 256)  # 频谱图目标尺寸

    # CycleGAN参数
    CYCLEGAN_LAMBDA = 10  # 循环一致性损失权重
    CYCLEGAN_LR = 0.001  # 学习率
    CYCLEGAN_EPOCHS = 200
    CYCLEGAN_BATCH_SIZE = 1  # CycleGAN通常使用batch size=1

    # CNN分类器参数
    CNN_LR = 0.001
    CNN_EPOCHS = 50
    CNN_BATCH_SIZE = 32
    CNN_DROPOUT = 0.5

    # 数据路径
    DATA_DIR = './data/'
    TRAIN_DIR = './data/train/'
    TEST_DIR = './data/test/'
    MODEL_DIR = './models/'

    # 实验参数
    N_SUBJECTS_MAX = 4  # 最大同时测试人数
    WALKING_DISTANCE = 5  # 行走距离 (m)
    WALKING_TIME = 5  # 最佳行走时间 (s)

    # 路径损耗模型参数
    PATH_LOSS_EXPONENT = 2.0
    REFERENCE_DISTANCE = 1.0

    # 异常步态类型
    ABNORMAL_GAIT_TYPES = {
        0: 'normal',
        1: 'spastic',  # 痉挛步态
        2: 'scissors',  # 剪刀步态
        3: 'steppage',  # 跨阈步态
        4: 'waddling',  # 摇摆步态
        5: 'propulsive',  # 前冲步态
        6: 'parkinsonian'  # 帕金森步态
    }