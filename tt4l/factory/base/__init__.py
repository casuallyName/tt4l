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
        "BasePipelineArguments",
    ],
    "data_process": [
        "DataPreProcess",
    ],
    "factory": ["DatasetType", "BaseTaskFactory", "BasePipeline"]
}

if TYPE_CHECKING:
    from .arguments import BaseTaskArguments, BasePredictArguments, BasePipelineArguments
    from .data_process import DataPreProcess
    from .factory import BaseTaskFactory, DatasetType, BasePipeline


else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
