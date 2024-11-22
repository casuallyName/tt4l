# @Time     : 2024/7/9 16:21
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

from typing import TYPE_CHECKING

from ...utils.import_utils import LazyModule

_import_structure = {
    "arguments": [
        "TextClassificationTaskArguments",
        "TextClassificationPredictArguments",
    ],
    "data": [
        "DataPreProcessForSequenceClassification"
    ],
    "factory": ["TextClassificationFactory"]
}

if TYPE_CHECKING:
    from .arguments import *
    from .data import DataPreProcessForSequenceClassification
    from .factory import TextClassificationFactory

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
