# @Time     : 2024/7/5 12:02
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import importlib
from unittest import TestCase

import torch


class BaseTestCase(TestCase):
    test_seq_length = 10
    test_batch_size = 4

    def config(self, **kwargs):
        raise NotImplementedError

    @property
    def input_ids(self):
        return torch.LongTensor([[200] * self.test_seq_length for _ in range(self.test_batch_size)])

    @property
    def attention_mask(self):
        return torch.LongTensor([[1] * self.test_seq_length for _ in range(self.test_batch_size)])

    @property
    def token_type_ids(self):
        return torch.LongTensor([[0] * self.test_seq_length for _ in range(self.test_batch_size)])

    def fake_inputs(self, return_attention_mask=True, return_token_type_ids=True):
        inputs_tensors = dict(input_ids=self.input_ids)
        if return_attention_mask:
            inputs_tensors['attention_mask'] = self.attention_mask
        if return_token_type_ids:
            inputs_tensors['token_type_ids'] = self.token_type_ids
        return inputs_tensors

    @staticmethod
    def model_import_test(import_path):
        path = import_path.split('.')
        for i in range(1, len(path)):
            try:
                importlib.import_module('.'.join(path[:i]))
            except:
                raise ModuleNotFoundError('.'.join(path[:i]))
            assert hasattr(importlib.import_module('.'.join(path[:i])), path[-1]), \
                f"import failure '{'.'.join(path[:i])}.{path[-1]}'"
