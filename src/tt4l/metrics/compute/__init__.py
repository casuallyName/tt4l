# @Time     : 2024/7/22 10:08
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "text_classification": [
        "ComputeMetricsForMse",
        "ComputeMetricsForAccuracy",
        "ComputeMetricsForF1",
    ],
    "token_classification": [
        'ComputeMetricsWithSeqeval',
    ],
    "universal_information_extraction": [
        "ComputeMetricsForUniversalInformationExtraction",
    ],
}

if TYPE_CHECKING:
    from .text_classification import (
        ComputeMetricsForMse,
        ComputeMetricsForAccuracy,
        ComputeMetricsForF1
    )

    from .token_classification import ComputeMetricsWithSeqeval

    from .universal_information_extraction import ComputeMetricsForUniversalInformationExtraction


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
