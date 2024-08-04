# @Time     : 2024/7/4 17:00
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "last_hidden_states_pooling": [
        "HiddenStatesPooling",
        "PoolingStrategy"
    ]
}

if TYPE_CHECKING:
    from .last_hidden_states_pooling import *
else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
