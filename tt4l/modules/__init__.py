# @Time     : 2024/7/4 16:47
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "encoder": [],
    "encoder.position": [
        "SinusoidalPositionEmbedding",
    ],
    "loss": [],
    "loss.multi_label_categorical_cross_entropy": [
        "multi_label_categorical_cross_entropy_loss",
        "MultiLabelCategoricalCrossEntropyLoss"
    ],
    "pointer": [],
    "pointer.global_pointer": [
        "GlobalPointer",
        "EfficientGlobalPointer",
    ],
    "pointer.span": [
        'Pointer',
        'EntitySpan',
        'EntityPointers',
    ],
    "pooling": [],
    "pooling.last_hidden_states_pooling": [
        "HiddenStatesPooling",
        "PoolingStrategy",
    ],
    "sequence": [],
    "sequence.crf": ["CRF"]
}

if TYPE_CHECKING:
    from . import *


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
