# @Time     : 2024/7/9 16:21
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["TextClassificationTaskArguments",
           "TextClassificationPredictArguments",
           "TextClassificationPipelineArguments"]

from dataclasses import dataclass, field
from typing import Optional

from tt4l.factory.base.arguments import BaseTaskArguments, BasePredictArguments, BasePipelineArguments


@dataclass
class TextClassificationTaskArguments(BaseTaskArguments):
    text_column_name: Optional[str] = field(
        default='sentence', metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    text_pair_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text pair to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default='label', metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )


@dataclass
class TextClassificationPredictArguments(BasePredictArguments):
    text_column_name: Optional[str] = field(
        default='sentence', metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    text_pair_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text pair to input in the file (a csv or JSON file)."}
    )


@dataclass
class TextClassificationPipelineArguments(BasePipelineArguments):
    is_multi_label: bool = field(
        default=False, metadata={"help": "Whether the task is multi-label classification."}
    )
    use_text_pair: bool = field(
        default=False, metadata={"help": "Add text_pair into inputs."}
    )
    device: Optional[str] = field(
        default=None,
        metadata={"help": "The device to run the model on, if None, will find the best device automatically."}
    )
    mark_line: float = field(
        default=0.5, metadata={"help": "The mark line to mark label in multi-label classification task."}
    )
    disable_sigmoid: bool = field(
        default=False,
        metadata={"help": "Whether to disable sigmoid in the output logits in multi-label classification task."}
    )
