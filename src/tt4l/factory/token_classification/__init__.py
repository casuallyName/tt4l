# @Time     : 2024/7/19 14:11
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from ...utils.import_utils import LazyModule

_import_structure = {
    "arguments": [
        "TokenClassificationPredictArguments",
        "TokenClassificationTaskArguments"
    ],
    "data": [
        "DataPreProcessForTokenClassification"
    ],
    "factory": ["TokenClassificationFactory"]
}

if TYPE_CHECKING:
    from .arguments import (
        TokenClassificationPredictArguments,
        TokenClassificationTaskArguments
    )
    from .data import DataPreProcessForTokenClassification
    from .factory import TokenClassificationFactory

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
