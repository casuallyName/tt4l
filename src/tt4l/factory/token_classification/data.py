# @Time     : 2024/7/19 15:53
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["DataPreProcessForTokenClassification"]

from typing import Optional, Union

from ...factory.base.data_process import DataPreProcess


class DataPreProcessForTokenClassification(DataPreProcess):
    def __init__(self, tokenizer, padding: Optional[Union[bool, str]], max_seq_length: Optional[int],
                 label_to_id: Optional[dict] = None,
                 token_key: Optional[str] = 'token',
                 label_key: Optional[str] = 'label',
                 label_all_tokens: bool = False,
                 label_is_name: bool = False,
                 add_special_tokens: bool = True,
                 ignore_token_id: int = -100,
                 **kwargs):
        super().__init__(tokenizer=tokenizer, padding=padding, max_seq_length=max_seq_length, **kwargs)
        self.token_key = token_key
        self.label_key = label_key
        self.label_to_id = label_to_id if label_to_id is not None else {}
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        self.label_all_tokens = label_all_tokens
        self.label_list = list(label_to_id.keys())
        self.label_is_name = label_is_name
        self.add_special_tokens = add_special_tokens
        self.ignore_token_id = ignore_token_id

        # Map that sends B-Xxx label to its I-Xxx counterpart
        self.b_to_i_label = []
        for idx, label in enumerate(self.label_list):
            if label.startswith("B-") and label.replace("B-", "I-") in self.label_list:
                self.b_to_i_label.append(self.label_list.index(label.replace("B-", "I-")))
            else:
                self.b_to_i_label.append(idx)

    def process(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.token_key],
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            add_special_tokens=self.add_special_tokens,
        )
        if self.label_key and examples[self.label_key][0]:
            labels = []
            for i, label in enumerate(examples[self.label_key]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # 特殊Token使用 ignore_token_id 填充
                    if word_idx is None:
                        label_ids.append(self.ignore_token_id)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        if self.label_is_name:
                            label_ids.append(self.label_to_id[label[word_idx]])
                        else:
                            label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        if self.label_all_tokens:
                            label_ids.append(self.b_to_i_label[self.label_to_id[label[word_idx]]])
                        else:
                            # 使用 ignore_token_id 进行填充
                            label_ids.append(self.ignore_token_id)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
        else:
            tokenized_inputs["word_ids"] = [tokenized_inputs.word_ids(batch_index=i)
                                            for i in range(len(tokenized_inputs['input_ids']))]
        return tokenized_inputs
