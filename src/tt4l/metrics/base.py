# @Time     : 2024/7/10 17:07
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ['MetricInput', 'BaseComputeMetric']

from typing import TypeVar, Dict

from transformers.trainer_utils import EvalPrediction, PredictionOutput

MetricInput = TypeVar('MetricInput', EvalPrediction, PredictionOutput)


class BaseComputeMetric:
    def __call__(self, p: MetricInput, **kwargs) -> Dict:
        return self._compute(p=p, **kwargs)

    def _compute(self, p: MetricInput, **kwargs) -> Dict:
        raise NotImplementedError
