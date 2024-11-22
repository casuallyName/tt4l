# @Time     : 2024/7/4 16:05
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import TYPE_CHECKING

from .utils.env import VERSION
from .utils.import_utils import LazyModule

_import_structure = {
    "cli": ['main'],
    "factory": [],
    "functional": [],
    "layers": [],
    "loss": [],
    "metrics": [],
    "models": [],
    "parser": [],
    "modeling_outputs": [
        "TokenDecodeClassifierOutput",
        "UniversalInformationExtractionModelOutput"
    ],
    "optimizer_builder": [
        "OptimizerBuilderForCrf"
    ],
}

_factory_import_structure = {
    "factory.base": [],
    "factory.auto": [
        "AutoFactory",
        "FACTORY_MAP",
        "TaskFactoryNotFound",
    ],

    "factory.text_classification": [],
    "factory.text_classification.arguments": [
        "TextClassificationTaskArguments",
        "TextClassificationPredictArguments",
    ],
    "factory.text_classification.data": [
        "DataPreProcessForSequenceClassification",
    ],
    "factory.text_classification.factory": [
        "TextClassificationFactory",
    ],

    "factory.token_classification": [],
    "factory.token_classification.arguments": [
        "TokenClassificationPredictArguments",
        "TokenClassificationTaskArguments",
    ],
    "factory.token_classification.data": [
        "DataPreProcessForTokenClassification",
    ],
    "factory.token_classification.factory": [
        "TokenClassificationFactory"
    ],

    "factory.universal_information_extraction": [],
    "factory.universal_information_extraction.arguments": [
        "UniversalInformationExtractionTaskArguments",
        "UniversalInformationExtractionPredictArguments"
    ],
    "factory.universal_information_extraction.data": [
        "DataPreProcessForUniversalInformationExtraction"
    ],
    "factory.universal_information_extraction.factory": [
        "UniversalInformationExtractionFactory"
    ]
}

_functional_import_structure = {
    "functional.numpy": [
        "softmax",
        "sigmoid",
    ]
}

_layers_import_structure = {
    "layers.encoder": [],
    "layers.encoder.position": [
        "SinusoidalPositionEmbedding",
    ],
    "layers.pointer.global_pointer": [
        "GlobalPointer",
        "EfficientGlobalPointer",
    ],
    "layers.pointer.span": [
        'Pointer',
        'EntitySpan',
        'EntityPointers',
    ],
    "layers.pooling.last_hidden_states_pooling": [
        "HiddenStatesPooling",
        "PoolingStrategy",
    ],
    "layers.sequence.crf": ["CRF"],
}

_loss_import_structure = {
    "loss.multi_label_categorical_cross_entropy": [
        "multi_label_categorical_cross_entropy_loss",
        "MultiLabelCategoricalCrossEntropy",
    ]
}

_metrics_import_structure = {
    "metrics.compute": [],

    "metrics.compute.text_classification": [
        "ComputeMetricsForMse",
        "ComputeMetricsForAccuracy",
        "ComputeMetricsForF1",
    ],
    "metrics.compute.token_classification": [
        'ComputeMetricsWithSeqeval',
    ],
    "metrics.compute.universal_information_extraction": [
        "ComputeMetricsForUniversalInformationExtraction",
    ],

    "metrics.evaluate": [],
    "metrics.evaluate.accuracy": ["Accuracy", ],
    "metrics.evaluate.f1": ["F1", ],
    "metrics.evaluate.mse": ["Mse", ],
    "metrics.evaluate.seqeval": ["Seqeval", ],
}

_models_import_structure = {
    "models.modeling_bert": [
        'BertForMultiLabelSequenceClassification',
        'BertWithHiddenPoolingForSequenceClassification',
        'BertWithCrfForTokenClassification',
        'BertForEntitySpanClassification',
    ],

    "models.modeling_ernie": [
        "ErnieForUniversalInformationExtraction",
        "ErnieForMultiLabelSequenceClassification",
        "ErnieWithHiddenPoolingForSequenceClassification",
        "ErnieWithCrfForTokenClassification",
        "ErnieForEntitySpanClassification",
    ]
}

_parser_import_structure = {
    "parser.sequence_parser": [
        "SequenceParser"
    ],
}

_import_structure.update(_factory_import_structure)
_import_structure.update(_functional_import_structure)
_import_structure.update(_layers_import_structure)
_import_structure.update(_loss_import_structure)
_import_structure.update(_models_import_structure)
_import_structure.update(_metrics_import_structure)
_import_structure.update(_parser_import_structure)

if TYPE_CHECKING:
    from .functional import *
    from .factory import *
    from .layers import *
    from .loss import *
    from .metrics import *
    from .models import *
    from .parser import *


else:
    import sys

    sys.modules[__name__] = LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": VERSION},
    )
