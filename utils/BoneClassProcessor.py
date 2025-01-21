import re

import torch


class BoneClassProcessor:
    def __init__(self):
        # 初始化骨骼字典和标签
        self.dictionary = {
            '0': ["yuan5_score", "第五远节指骨骺", "5th distal phalanx", "5DP"],
            '1': ["zhong5_score", "第五中节指骨骺", "5th middle phalanx", "5MP"],
            '2': ["jin5_score", "第五近节指骨骺", "5th proximal phalanx", "5PP"],
            '3': ["zhang5_score", "第五掌骨骺", "5th metacarpal", "5MC"],
            '4': ["yuan3_score", "第三远节指骨骺", "3rd distal phalanx", "3DP"],
            '5': ["zhong3_score", "第三中节指骨骺", "3rd middle phalanx", "3MP"],
            '6': ["jin3_score", "第三近节指骨骺", "3rd proximal phalanx", "3PP"],
            '7': ["zhang3_score", "第三掌骨骺", "3rd metacarpal", "3MC"],
            '8': ["yuan1_score", "第一远节指骨骺", "1st distal phalanx", "1DP"],
            '9': ["zhang1_score", "第一掌骨骺", "1st metacarpal", "1MC"],
            '10': ["jin1_score", "第一近节指骨骺", "1st proximal phalanx", "1PP"],
            '11': ["gou_score", "钩骨", "hamate", "Ham"],
            '12': ["tou_score", "头状骨", "capitate", "Cap"],
            '13': ["rao_score", "桡骨骺", "radial", "Rad"],
        }
        self.labels = [
            "5DP_0", "5DP_1", "5DP_2", "5DP_3", "5DP_4", "5DP_5", "5DP_6", "5DP_7", "5DP_8",
            "5MP_0", "5MP_1", "5MP_2", "5MP_3", "5MP_4", "5MP_5", "5MP_6", "5MP_7", "5MP_8",
            "5PP_0", "5PP_1", "5PP_2", "5PP_3", "5PP_4", "5PP_5", "5PP_6", "5PP_7", "5PP_8",
            "5MC_0", "5MC_1", "5MC_2", "5MC_3", "5MC_4", "5MC_5", "5MC_6", "5MC_7", "5MC_8",
            "3DP_0", "3DP_1", "3DP_2", "3DP_3", "3DP_4", "3DP_5", "3DP_6", "3DP_7", "3DP_8",
            "3MP_0", "3MP_1", "3MP_2", "3MP_3", "3MP_4", "3MP_5", "3MP_6", "3MP_7", "3MP_8",
            "3PP_0", "3PP_1", "3PP_2", "3PP_3", "3PP_4", "3PP_5", "3PP_6", "3PP_7", "3PP_8",
            "3MC_0", "3MC_1", "3MC_2", "3MC_3", "3MC_4", "3MC_5", "3MC_6", "3MC_7", "3MC_8",
            "1DP_0", "1DP_1", "1DP_2", "1DP_3", "1DP_4", "1DP_5", "1DP_6", "1DP_7", "1DP_8",
            "1MC_0", "1MC_1", "1MC_2", "1MC_3", "1MC_4", "1MC_5", "1MC_6", "1MC_7", "1MC_8",
            "1PP_0", "1PP_1", "1PP_2", "1PP_3", "1PP_4", "1PP_5", "1PP_6", "1PP_7", "1PP_8",
            "Ham_0", "Ham_1", "Ham_2", "Ham_3", "Ham_4", "Ham_5", "Ham_6", "Ham_7", "Ham_8",
            "Cap_0", "Cap_1", "Cap_2", "Cap_3", "Cap_4", "Cap_5", "Cap_6", "Cap_7", "Cap_8",
            "Rad_0", "Rad_1", "Rad_2", "Rad_3", "Rad_4", "Rad_5", "Rad_6", "Rad_7", "Rad_8", "Rad_9", "Rad_10"
        ]
    # s0.0;b2.3;r1.0;z1.0,2.0,1.0;j1.0,3.0,2.0;zh2.0,1.0;y1.0,1.0,1.0;t3.0;g2.0
    def extract_key_value_pairs_from_prompt(self, prompt):
        bone_mapping = {
            "r": ["rao_score"],
            "z": ["zhang1_score", "zhang3_score", "zhang5_score"],
            "j": ["jin1_score", "jin3_score", "jin5_score"],
            "zh": ["zhong3_score", "zhong5_score"],
            "y": ["yuan1_score", "yuan3_score", "yuan5_score"],
            "t": ["tou_score"],
            "g": ["gou_score"]
        }
        result = {}
        # 按分号分割
        groups = prompt.split(";")
        for group in groups:
            # 如果不是s和b开头
            if not group.startswith("s") and not group.startswith("b"):
                match = re.match(r'^([a-zA-Z]*)(.*)', group)
                bone_name = match.group(1)
                rest = match.group(2)
                parts = rest.split(",")
                scores = list(map(float, parts[:]))
                if bone_name in bone_mapping:
                    keys = bone_mapping[bone_name]
                    for i, score in enumerate(scores):
                        if i < len(keys):
                            result[keys[i]] = str(int(score))
        return result

    def convert_to_abbreviation_format(self, key_value_pairs):
        result = []
        for key, value in key_value_pairs.items():
            for idx, details in self.dictionary.items():
                if details[0] == key:
                    abbreviation = details[3]
                    formatted = f"{abbreviation}_{value}"
                    result.append(formatted)
                    break
        return result

    def fill_into_128_positions(self, abbreviation_list):
        positions = torch.zeros(128, dtype=torch.float32)
        for abbreviation in abbreviation_list:
            if abbreviation in self.labels:
                index = self.labels.index(abbreviation)
                positions[index] = 1
        return positions

    def __call__(self, prompt):
        """
        使用 __call__ 方法封装整个流程，便于直接调用类实例。

        参数:
        prompt (str): 输入的 prompt 字符串。

        返回:
        torch.Tensor: 128 维的张量。
        """
        key_value_pairs = self.extract_key_value_pairs_from_prompt(prompt)
        abbreviation_list = self.convert_to_abbreviation_format(key_value_pairs)
        positions = self.fill_into_128_positions(abbreviation_list)
        return positions
