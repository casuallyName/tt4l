# @Time     : 2024/7/22 10:12
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ['ComputeMetricsWithSeqeval']

from typing import Dict

import numpy as np
from ..base import BaseComputeMetric, MetricInput
from ..evaluate.seqeval import Seqeval


class ComputeMetricsWithSeqeval(BaseComputeMetric):
    def __init__(self, label_list, return_entity_level_metrics: bool = False, ignore_token_id: int = -100, **kwargs):
        super().__init__()
        self.metric = Seqeval(**kwargs)
        self.label_list = label_list
        self.return_entity_level_metrics = return_entity_level_metrics
        self.ignore_token_id = ignore_token_id

    def _compute(self, p: MetricInput, **kwargs) -> Dict:
        if isinstance(p.predictions, tuple):
            predictions = p.predictions[1]
        else:
            predictions = np.argmax(p.predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != self.ignore_token_id]
            for prediction, label in zip(predictions, p.label_ids)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != self.ignore_token_id]
            for prediction, label in zip(predictions, p.label_ids)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        if self.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
