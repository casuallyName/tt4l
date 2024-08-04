# @Time     : 2024/7/22 10:48
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["ComputeMetricsForMse", "ComputeMetricsForAccuracy", "ComputeMetricsForF1"]

from typing import Dict, Literal

import numpy as np
from tt4l.metrics.base import BaseComputeMetric, MetricInput
from tt4l.metrics.evaluate.accuracy import Accuracy
from tt4l.metrics.evaluate.f1 import F1
from tt4l.metrics.evaluate.mse import Mse


class ComputeMetricsForMse(BaseComputeMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.metric = Mse(**kwargs)

    def _compute(self, p: MetricInput, **kwargs) -> Dict:
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.squeeze(predictions)
        return self.metric.compute(predictions=predictions, references=p.label_ids)


class ComputeMetricsForAccuracy(BaseComputeMetric):
    def __init__(self, **kwargs):
        super().__init__()
        self.metric = Accuracy(**kwargs)

    def _compute(self, p: MetricInput, **kwargs) -> Dict:
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=p.label_ids)


class ComputeMetricsForF1(BaseComputeMetric):
    def __init__(self, average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = "micro", **kwargs):
        super().__init__()
        self.metric = F1(**kwargs)
        self.average = average

    def _compute(self, p: MetricInput, **kwargs) -> Dict:
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # convert predictions to multi-hot encoding
        predictions = np.array([np.where(p > 0, 1, 0) for p in predictions], dtype=p.label_ids.dtype).ravel()
        # Micro F1 is commonly used in multi-label classification
        references = p.label_ids.ravel()
        return self.metric.compute(predictions=predictions, references=references, average=self.average)
