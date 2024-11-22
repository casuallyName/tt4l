# @Time     : 2024/10/9 17:25
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from ..utils.import_utils import LazyModule

_import_structure = {
    "numpy": [
        "softmax",
        "sigmoid",
    ]
}

if TYPE_CHECKING:
    from .numpy import *

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)