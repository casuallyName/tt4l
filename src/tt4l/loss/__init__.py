# @Time     : 2024/8/2 18:05
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

from typing import TYPE_CHECKING

from ..utils.import_utils import LazyModule

_import_structure = {
    "multi_label_categorical_cross_entropy": [
        "multi_label_categorical_cross_entropy_loss",
        "MultiLabelCategoricalCrossEntropy",
    ]
}

if TYPE_CHECKING:
    from .multi_label_categorical_cross_entropy import *

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
