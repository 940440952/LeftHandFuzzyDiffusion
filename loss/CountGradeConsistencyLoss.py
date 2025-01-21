import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


def count_grade_consistency_loss(true_bone_classes, true_bone_grades, pred_bone_classes, pred_bone_grades,
                                 pred_bone_confidences,
                                 lambda_1=0.1, lambda_2=0.1):
    """
    Count-Grade Consistency Loss (CGCL) 实现，结合数量一致性损失和等级一致性损失。

    参数：
    - true_bone_classes: 真实的骨骼种类，形状为 [num_bones]
    - true_bone_grades: 真实的骨骼成熟度等级，形状为 [num_bones]
    - pred_bone_classes: 模型预测的骨骼种类，形状为 [num_bones]
    - pred_bone_grades: 模型预测的骨骼成熟度等级，形状为 [num_bones]
    - pred_bone_confidences: 模型预测的骨骼成熟度等级置信度，形状为 [num_bones]
    - lambda_1: 数量一致性损失的权重
    - lambda_2: 等级一致性损失的权重

    返回：
    - total_loss: 总损失
    """
    # pred_bone_classes, pred_bone_grades, pred_bone_confidences长度要相等，不相等就报错
    if len(pred_bone_classes) != len(pred_bone_grades) or len(pred_bone_grades) != len(pred_bone_confidences):
        raise ValueError("pred_bone_classes, pred_bone_grades, pred_bone_confidences must have the same length.")
    num_bone_classes = true_bone_classes.shape[0]

    # 1. 数量一致性损失
    count_loss = 0.0

    for j in range(num_bone_classes):
        is_pred_class_match = pred_bone_classes == true_bone_classes[j]
        is_pred_class_match = torch.tensor(is_pred_class_match, dtype=torch.float32)
        matching_pred_count = torch.sum(is_pred_class_match)
        count_loss += torch.abs(matching_pred_count - 1)

    # 2. 等级一致性损失
    grade_loss = 0.0

    for j in range(num_bone_classes):  # 遍历每个真实骨头类别
        true_bone_class = true_bone_classes[j]  # 当前真实骨骼种类
        true_bone_grade = true_bone_grades[j]  # 当前真实骨骼等级

        # 找到预测中与 true_bone_class 匹配的骨头
        is_pred_class_match = pred_bone_classes == true_bone_class
        is_pred_class_match = torch.tensor(is_pred_class_match, dtype=torch.float32)
        matching_pred_count = torch.sum(is_pred_class_match)

        if matching_pred_count == 0:  # 没有骨头
            grade_loss += 1.0
        elif matching_pred_count == 1.0:  # 有一个骨头
            # 找下标
            matching_indices = pred_bone_classes == true_bone_class
            matching_indices = torch.tensor(matching_indices, dtype=torch.float32)
            matching_pred_indices = torch.where(matching_indices)[0]
            pred_bone_grade = pred_bone_grades[matching_pred_indices]
            pred_bone_confidence = pred_bone_confidences[matching_pred_indices]

            if pred_bone_grade != true_bone_grade:  # 等级不一致
                grade_loss += torch.abs(pred_bone_grade - true_bone_grade)
            else:  # 等级一致
                grade_loss += 1.0 - pred_bone_confidence
        else:  # 有多个骨头
            # 找最大置信度的骨头等级
            matching_indices = pred_bone_classes == true_bone_class
            filtered_confidences = [confidence if matching else 0 for confidence, matching in
                                    zip(pred_bone_confidences, matching_indices)]
            max_index = np.argmax(filtered_confidences)
            pred_bone_confidence = pred_bone_confidences[max_index]
            pred_bone_grade = pred_bone_grades[max_index]
            if pred_bone_grade != true_bone_grade:  # 等级不一致
                grade_loss += torch.abs(pred_bone_grade - true_bone_grade)
            else:  # 等级一致
                grade_loss += 1.0 - pred_bone_confidence

            # 找到

    # 3. 总损失
    total_loss = lambda_1 * count_loss + lambda_2 * grade_loss
    return total_loss


# model = YOLO(r"D:\WorkSpace\boneAge\实验20241022\识别\runs\detect\train\weights\best.pt")
#
# results = model(r"D:\WorkSpace\boneAge\实验20241022\whole hand\7606.jpg")
#
# # for box in results[0].boxes:
# #     print(box.cls, box.conf, box.xyxy)
#
# # 示例
# batch_size = 2
# num_bones = 14
# num_classes = 5
#
# # 模拟真实数据
# true_class = np.array(['5DP', '5MP', '5PP', '5MC', '3DP', '3MP', '3PP', '3MC', '1DP', '1MC', '1PP', 'Ham', 'Cap',
#                        'Rad'])  # 真实骨头类别，形状 [num_bones]
# true_class_grade = torch.from_numpy(np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]))  # 真实骨头成熟度等级，形状 [num_bones]
# y_pred_class = np.array(
#     ['5MP', '5MP', '5DP', '5MC', '3DP', '3MP', '3PP', '3MC', '1DP', '1MC', '5PP', '1PP', 'Ham', 'Cap',
#      'Rad'])  # 预测骨头种类，形状 [batch_size, num_bones]
# y_pred_class_grade = torch.from_numpy(
#     np.array([4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]))  # 预测骨头成熟度等级，形状 [batch_size, num_bones]
# y_pred_conf = torch.from_numpy(np.array([0.5, 0.1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
#                                          0.4]))  # 预测的置信度，形状 [batch_size, num_bones]
#
# # 计算损失
# loss = count_grade_consistency_loss(true_class, true_class_grade, y_pred_class, y_pred_class_grade, y_pred_conf,
#                                     lambda_1=1.0, lambda_2=1.0)
# print("Total Loss:", loss)
