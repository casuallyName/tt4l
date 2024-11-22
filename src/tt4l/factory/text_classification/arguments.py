# @Time     : 2024/7/9 16:21
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["TextClassificationTaskArguments",
           "TextClassificationPredictArguments"]

from dataclasses import dataclass, field
from typing import Optional

from ...factory.base.arguments import BaseTaskArguments, BasePredictArguments


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
