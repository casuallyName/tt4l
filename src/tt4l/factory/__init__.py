# @Time     : 2024/7/9 16:19
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from ..utils.import_utils import LazyModule

_import_structure = {
    "base": [],
    "auto": ["AutoFactory", "FACTORY_MAP", "TaskFactoryNotFound"],
    "text_classification": [],
    "text_classification.arguments": [
        "TextClassificationTaskArguments",
        "TextClassificationPredictArguments",
    ],
    "text_classification.data": ["DataPreProcessForSequenceClassification", ],
    "text_classification.factory": ["TextClassificationFactory"],

    "token_classification": [],
    "token_classification.arguments": ["TokenClassificationPredictArguments", "TokenClassificationTaskArguments", ],
    "token_classification.data": ["DataPreProcessForTokenClassification", ],
    "token_classification.factory": ["TokenClassificationFactory"],

    "universal_information_extraction": [],
    "universal_information_extraction.arguments": [
        "UniversalInformationExtractionTaskArguments",
        "UniversalInformationExtractionPredictArguments"
    ],
    "universal_information_extraction.data": [
        "DataPreProcessForUniversalInformationExtraction"
    ],
    "universal_information_extraction.factory": ["UniversalInformationExtractionFactory"]
}

if TYPE_CHECKING:
    from .auto import AutoFactory, FACTORY_MAP, TaskFactoryNotFound
    from .text_classification import *
    from .token_classification import *
    from .universal_information_extraction import *


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
