# @Time     : 2024/8/5 17:50
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

__all__ = ["DataPreProcessForUniversalInformationExtraction"]

from typing import Optional, Union

import numpy as np
from tt4l.factory.base.data_process import DataPreProcess


class DataPreProcessForUniversalInformationExtraction(DataPreProcess):
    def __init__(self, tokenizer, padding: Optional[Union[bool, str]], max_seq_length: Optional[int],
                 prompt_key: str = 'prompt', text_key: str = 'text', entities_key: str = 'entities',
                 **kwargs):
        super().__init__(tokenizer=tokenizer, padding=padding, max_seq_length=max_seq_length, **kwargs)
        self.prompt_key = prompt_key
        self.text_key = text_key
        self.entities_key = entities_key

    def process(self, examples):
        padding_type = "max_length"
        padding_type = "longest"
        tokenized_inputs = self.tokenizer(
            text=examples[self.prompt_key],
            text_pair=examples[self.text_key],
            stride=2,
            truncation=True,
            padding=padding_type,
            max_length=self.max_seq_length,
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        num_examples = len(examples[self.entities_key])
        seq_length = len(tokenized_inputs["input_ids"][0])
        start_positions = np.zeros((num_examples, seq_length,), dtype=np.float32)
        end_positions = np.zeros((num_examples, seq_length,), dtype=np.float32)

        for idx, (entities, offset_mapping) in enumerate(
                zip(examples[self.entities_key], tokenized_inputs["offset_mapping"])):
            offset_mapping = self.mask_offset_mapping(offset_mapping)
            for text, start_id, end_id in zip(entities['text'], entities['start'], entities['end']):
                start_positions[idx][self.map_offset(start_id, offset_mapping)] = 1
                end_positions[idx][self.map_offset(end_id - 1, offset_mapping)] = 1
        tokenized_inputs["start_positions"] = start_positions
        tokenized_inputs["end_positions"] = end_positions
        return tokenized_inputs

    @staticmethod
    def map_offset(ori_offset, offset_mapping):
        """
        map ori offset to token offset
        """
        for index, span in enumerate(offset_mapping):
            if span[0] <= ori_offset < span[1]:
                return index
        return -1

    @staticmethod
    def mask_offset_mapping(offset_mapping):
        prompt_end_token_id = offset_mapping[1:].index((0, 0,))
        bias = offset_mapping[prompt_end_token_id][1] + 1
        for i in range(bias):
            offset_mapping[i] = (-1, -1)
        return offset_mapping
