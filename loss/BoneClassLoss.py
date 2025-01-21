import torch
import torch.nn.functional as F

def bone_age_loss(y_true, y_pred):
    """
    骨龄种类损失函数 (PyTorch 实现)

    参数:
    y_true: torch.Tensor
        真实标签 (形状: [128])，值为 0 或 1，表示对应位置是否为目标类别
    y_pred: torch.Tensor
        预测值 (形状: [128])，值为置信度（0 到 1）

    返回:
    loss: torch.Tensor
        损失值，计算方式为 y_true 为 1 的位置对应的 (1 - y_pred) 的总和
    """
    # 确保 y_true 和 y_pred 的形状一致
    assert y_true.shape == y_pred.shape, "y_true 和 y_pred 的形状必须一致"

    # 提取 y_true 为 1 的位置对应的 y_pred
    conf_values = y_pred[y_true == 1]

    # 计算损失为 1 - conf_values，并求和
    loss = torch.sum(1.0 - conf_values)

    return loss