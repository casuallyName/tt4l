# @Time     : 2024/8/5 18:42
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ['ComputeMetricsForUniversalInformationExtraction']

from typing import Dict

import numpy as np
from tt4l.metrics.base import BaseComputeMetric, MetricInput


class ComputeMetricsForUniversalInformationExtraction(BaseComputeMetric):
    def __init__(self, **kwargs):
        super().__init__()

    def _compute(self, p: MetricInput, **kwargs) -> Dict:
        all_start_probs, all_end_probs = p.predictions
        all_start_ids, all_end_ids = p.label_ids
        tp = 0  # 数据集中实际为命名实体，模型也预测为命名实体的数量
        fp = 0  # 数据集中实际为非命名实体，模型却预测为命名实体的数量
        fn = 0  # 数据集中实际为命名实体，模型却预测为非命名实体的数量
        for start_ids, end_ids, start_probs, end_probs in (
                zip(all_start_ids, all_end_ids, all_start_probs, all_end_probs)):
            true_result = set()
            predict_result = set()
            for start_id, end_id in zip(np.argwhere(start_ids > .5).reshape(-1).tolist(),
                                        np.argwhere(end_ids > .5).reshape(-1).tolist()):
                true_result.add((start_id, end_id))
            for start_id, end_id in zip(np.argwhere(start_probs > .5).reshape(-1).tolist(),
                                        np.argwhere(end_probs > .5).reshape(-1).tolist()):
                predict_result.add((start_id, end_id))
            tp += len(set(true_result) & set(predict_result))
            fp += len(set(predict_result) - set(true_result))
            fn += len(set(true_result) - set(predict_result))

        return {
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fn + fp),
        }
