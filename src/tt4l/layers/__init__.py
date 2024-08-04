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
    from .encoder import *
    from .pointer import *
    from .pooling import *
    from .sequence import *


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
