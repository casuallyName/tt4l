# @Time     : 2024/7/4 16:39
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

from typing import TYPE_CHECKING

from ..utils.import_utils import LazyModule

_import_structure = {
    "modeling_bert": [
        'BertForMultiLabelSequenceClassification',
        'BertWithHiddenPoolingForSequenceClassification',
        'BertWithCrfForTokenClassification',
        'BertForEntitySpanClassification',
    ],
    "modeling_ernie": [
        "ErnieForUniversalInformationExtraction",
        "ErnieForMultiLabelSequenceClassification",
        "ErnieWithHiddenPoolingForSequenceClassification",
        "ErnieWithCrfForTokenClassification",
        "ErnieForEntitySpanClassification",
    ]
}

if TYPE_CHECKING:
    from .modeling_bert import *
    from .modeling_ernie import *

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
