# @Time     : 2024/8/2 11:40
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "arguments": [
        "UniversalInformationExtractionTaskArguments",
        "UniversalInformationExtractionPredictArguments"
    ],
    "data": [
        "DataPreProcessForUniversalInformationExtraction"
    ],
    "factory": ["UniversalInformationExtractionFactory"]
}

if TYPE_CHECKING:
    from .arguments import UniversalInformationExtractionPredictArguments, UniversalInformationExtractionTaskArguments
    from .data import DataPreProcessForUniversalInformationExtraction
    from .factory import UniversalInformationExtractionFactory

else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
