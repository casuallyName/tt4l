# @Time     : 2024/7/9 17:15
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["DataPreProcessForSequenceClassification"]

from typing import Optional, List, Union

from tt4l.factory.base.data_process import DataPreProcess


class DataPreProcessForSequenceClassification(DataPreProcess):
    def __init__(self, tokenizer, padding: Optional[Union[bool, str]], max_seq_length: Optional[int],
                 label_to_id: Optional[dict] = None,
                 sentence1_key: Optional[str] = 'sentence',
                 sentence2_key: Optional[str] = None,
                 label_key: Optional[str] = 'label',
                 **kwargs):
        super().__init__(tokenizer=tokenizer, padding=padding, max_seq_length=max_seq_length, **kwargs)

        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.label_key = label_key
        self.label_to_id = label_to_id if label_to_id is not None else {}
        self.id_to_label = {v: k for k, v in label_to_id.items()}

    def expand_label(self, label: List[int]) -> List[int]:
        label_line = [0] * len(self.label_to_id)
        for label_idx in label:
            label_line[label_idx] = 1
        return label_line

    def process(self, examples):
        if self.sentence2_key is not None:
            args = (examples[self.sentence1_key], examples[self.sentence2_key])
        else:
            args = (examples[self.sentence1_key],)

        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        if self.label_key in list(examples.keys()):
            if isinstance(examples[self.label_key][0], list):
                result["label"] = [self.expand_label(label) for label in examples[self.label_key]]
            else:
                result["label"] = examples[self.label_key]
        return result
