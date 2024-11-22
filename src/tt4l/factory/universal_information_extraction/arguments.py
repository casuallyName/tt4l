# @Time     : 2024/8/5 16:35
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["UniversalInformationExtractionTaskArguments", "UniversalInformationExtractionPredictArguments"]

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

from ...factory.base.arguments import BaseTaskArguments, BasePredictArguments


@dataclass
class UniversalInformationExtractionTaskArguments(BaseTaskArguments):
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    prompt_column_name: Optional[str] = field(
        default='prompt', metadata={"help": "The column name of prompt."}
    )
    text_column_name: Optional[str] = field(
        default='sentence', metadata={"help": "The column name of text."}
    )
    entities_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of entities."}
    )
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )


@dataclass
class UniversalInformationExtractionPredictArguments(BasePredictArguments):
    schema: Union[str, List, Dict] = field(
        default=None, metadata={"help": "The schema of extraction."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    text_column_name: Optional[str] = field(
        default='sentence', metadata={"help": "The column name of text."}
    )
    split_text: bool = field(
        default=False, metadata={"help": "Whether to split the sentences into shorter than max_text_split_length."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_text_split_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum sequence length after split. Sequences longer than this will be truncated."
            )
        },
    )
