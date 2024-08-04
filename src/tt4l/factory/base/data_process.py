# @Time     : 2024/7/9 17:13
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["DataPreProcess"]

from typing import Optional

from transformers.trainer import logger


class DataPreProcess:
    def __init__(self,
                 tokenizer,
                 padding: Optional[bool],
                 max_seq_length: Optional[int],
                 **kwargs):
        self.tokenizer = tokenizer
        self.padding = padding
        if max_seq_length > self.tokenizer.model_max_length:
            logger.warning("Max sequence length exceeds model max length")
            max_seq_length = self.tokenizer.model_max_length
        self.max_seq_length = max_seq_length
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, examples):
        return self.process(examples=examples)

    def process(self, examples):
        raise NotImplementedError
