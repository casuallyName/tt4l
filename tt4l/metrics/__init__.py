# @Time     : 2024/7/10 16:59
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "compute": [],
    "compute.text_classification": ["ComputeMseMetrics", "ComputeAccuracyMetrics", "ComputeF1Metrics"],
    "compute.token_classification": ['ComputeSeqevalMetrics'],

    "evaluate": [],
    "evaluate.accuracy": ["Accuracy"],
    "evaluate.f1": ["F1"],
    "evaluate.mse": ["Mse"],
    "evaluate.seqeval": ["Seqeval"],
}

if TYPE_CHECKING:
    from .compute import *
    from .evaluate import *

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
