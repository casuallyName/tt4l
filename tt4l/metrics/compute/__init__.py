# @Time     : 2024/7/22 10:08
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "text_classification": ["ComputeMseMetrics", "ComputeAccuracyMetrics", "ComputeF1Metrics"],
    "token_classification": ['ComputeSeqevalMetrics'],
}

if TYPE_CHECKING:
    from tt4l.metrics.compute.text_classification import (
        ComputeMseMetrics,
        ComputeF1Metrics,
        ComputeAccuracyMetrics
    )

    from tt4l.metrics.compute.token_classification import (
        ComputeSeqevalMetrics,
    )


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
