# -*- coding: utf-8 -*-
# @Time     : 2024/7/23 22:41
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from ..utils.import_utils import LazyModule

_import_structure = {
    "sequence_parser": ["SequenceParser"],
}

if TYPE_CHECKING:
    from .base import *
    from .sequence_parser import *

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
