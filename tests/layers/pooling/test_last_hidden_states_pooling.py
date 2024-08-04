# @Time     : 2024/7/5 14:18
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import unittest

import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from tt4l.utils.testing_utils import BaseTestCase


class LastHiddenStatesPoolingTestCase(BaseTestCase):
    num_hidden_layers = 4
    hidden_size = 8

    @property
    def model_output(self):
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=torch.FloatTensor(
                [[[self.num_hidden_layers - 1] * self.hidden_size] * self.test_seq_length] * self.test_batch_size),
            pooler_output=torch.FloatTensor(
                [[self.num_hidden_layers - 1] * self.test_seq_length] * self.test_batch_size),
            hidden_states=tuple(
                [torch.FloatTensor([[[i] * self.hidden_size] * self.test_seq_length] * self.test_batch_size)
                 for i in range(self.num_hidden_layers)])
        )

    @property
    def attention_mask(self):
        return torch.LongTensor([[1] * self.test_seq_length for _ in range(self.test_batch_size)])

    def test_HiddenStatesPooling(self):
        self.model_import_test('tt4l.layers.pooling.last_hidden_states_pooling.HiddenStatesPooling')
        self.model_import_test('tt4l.layers.pooling.last_hidden_states_pooling.PoolingStrategy')
        from tt4l import PoolingStrategy, HiddenStatesPooling
        layer = HiddenStatesPooling(cls_in_first=False)
        assert layer.pool_strategy == PoolingStrategy.FL_AVG_POOLING
        assert layer.cls_in_first is False
        assert layer.cls_index == -1

        # PoolingStrategy.CLS_POOLING
        layer = HiddenStatesPooling(pool_strategy='cls')
        assert layer.pool_strategy == PoolingStrategy.CLS_POOLING
        predictions = layer(output=self.model_output,
                            attention_mask=self.attention_mask)
        result = self.model_output.last_hidden_state[:, layer.cls_index]
        assert (predictions == result).sum() == self.hidden_size * self.test_batch_size

        # PoolingStrategy.MEAN_POOLING
        layer = HiddenStatesPooling(pool_strategy='mean')
        assert layer.pool_strategy == PoolingStrategy.MEAN_POOLING
        predictions = layer(output=self.model_output, attention_mask=self.attention_mask)
        result = (torch.sum(self.model_output.last_hidden_state * self.attention_mask[:, :, None], dim=1)
                  / self.attention_mask.sum(dim=1)[:, None])
        assert (predictions == result).sum() == self.hidden_size * self.test_batch_size

        # PoolingStrategy.MAX_POOLING
        layer = HiddenStatesPooling(pool_strategy='max')
        assert layer.pool_strategy == PoolingStrategy.MAX_POOLING
        predictions = layer(output=self.model_output, attention_mask=self.attention_mask)
        result = (self.model_output.last_hidden_state * self.attention_mask[:, :, None]).max(dim=-2).values
        assert (predictions == result).sum() == self.hidden_size * self.test_batch_size

        # PoolingStrategy.FL_AVG_POOLING
        layer = HiddenStatesPooling(pool_strategy=PoolingStrategy.FL_AVG_POOLING)
        assert layer.pool_strategy == PoolingStrategy.FL_AVG_POOLING
        assert layer.cls_in_first is True
        predictions = layer(output=self.model_output, attention_mask=self.attention_mask)
        result = (self.model_output.hidden_states[1] * self.attention_mask[:, :, None]).sum(dim=1)
        result += (self.model_output.hidden_states[-1] * self.attention_mask[:, :, None]).sum(dim=1)
        result = result / (2 * torch.sum(self.attention_mask, dim=1)[:, None])
        assert (predictions == result).sum() == self.hidden_size * self.test_batch_size


if __name__ == '__main__':
    unittest.main(verbosity=2)
