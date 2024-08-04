# @Time     : 2024/7/22 10:08
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "accuracy": ["Accuracy"],
    "f1": ["F1"],
    "mse": ["Mse"],
    "seqeval": ["Seqeval"],
}

if TYPE_CHECKING:
    from tt4l.metrics.evaluate.accuracy import Accuracy
    from tt4l.metrics.evaluate.f1 import F1
    from tt4l.metrics.evaluate.mse import Mse
    from tt4l.metrics.evaluate.seqeval import Seqeval


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
