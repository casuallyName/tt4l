# @Time     : 2024/7/9 16:20
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

from typing import TYPE_CHECKING

from tt4l.utils.import_utils import LazyModule

_import_structure = {
    "arguments": [
        "BaseTaskArguments",
        "BasePredictArguments",
    ],
    "data_process": [
        "DataPreProcess",
    ],
    "factory": ["DatasetType", "BaseTaskFactory"]
}

if TYPE_CHECKING:
    from .arguments import BaseTaskArguments, BasePredictArguments
    from .data_process import DataPreProcess
    from .factory import BaseTaskFactory, DatasetType


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
