# @Time     : 2024/7/4 18:06
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "global_pointer": [
        "GlobalPointer",
        "EfficientGlobalPointer"
    ],
    "span": [
        'Pointer',
        'EntitySpan',
        'EntityPointers'
    ],
}

if TYPE_CHECKING:
    from .global_pointer import *
    from .span import *
else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
