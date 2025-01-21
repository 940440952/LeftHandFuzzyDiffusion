import torch
import torch.nn as nn


class FuzzyLearningModule(nn.Module):
    def __init__(self, num_channels, num_fuzzy_functions=5):
        super(FuzzyLearningModule, self).__init__()

        # 定义高斯模糊函数的均值和标准差参数
        self.means = nn.Parameter(torch.randn(num_fuzzy_functions, num_channels))
        self.stds = nn.Parameter(torch.abs(torch.randn(num_fuzzy_functions, num_channels)))

        # 定义批归一化层和残差连接
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # x 的形状为 (batch_size, num_channels, height, width)
        batch_size, num_channels, height, width = x.size()

        # 展开 x 以计算模糊特征
        x_flat = x.view(batch_size, num_channels, -1)  # 形状: (batch_size, num_channels, H*W)

        # 计算高斯模糊函数
        fuzzy_features = []
        for i in range(self.means.size(0)):
            mean = self.means[i].view(1, num_channels, 1)
            std = self.stds[i].view(1, num_channels, 1)
            fuzzy_feature = torch.exp(-((x_flat - mean) ** 2) / (2 * std ** 2))
            fuzzy_features.append(fuzzy_feature)

        # 将所有模糊特征应用 "AND" 规则（取乘积）
        fuzzy_features = torch.stack(fuzzy_features, dim=0).prod(dim=0)  # 形状: (batch_size, num_channels, H*W)
        fuzzy_features = fuzzy_features.view(batch_size, num_channels, height, width)

        # 残差连接并通过批归一化
        fuzzy_features = self.bn1(fuzzy_features)
        x = self.bn2(x)
        output = fuzzy_features + x

        return output
