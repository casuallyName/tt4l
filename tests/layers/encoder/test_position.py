# -*- coding: utf-8 -*-
# @Time     : 2024/7/26 23:02
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import torch

from utils.testing_utils import BaseTestCase


class PositionTestCase(BaseTestCase):

    def test_SinusoidalPositionEmbedding(self):
        self.model_import_test('tt4l.layers.encoder.position.SinusoidalPositionEmbedding')
        from tt4l import SinusoidalPositionEmbedding
        from transformers import BertConfig
        config = BertConfig()
        inputs = torch.rand((self.test_batch_size, self.test_seq_length, config.hidden_size))

        pos = SinusoidalPositionEmbedding(output_dim=config.hidden_size)
        assert pos(inputs).shape == torch.Size((self.test_batch_size, self.test_seq_length, config.hidden_size))
        pos = SinusoidalPositionEmbedding(output_dim=config.hidden_size, merge_mode='mul')

        assert pos(inputs).shape == torch.Size((self.test_batch_size, self.test_seq_length, config.hidden_size))
        pos = SinusoidalPositionEmbedding(output_dim=config.hidden_size, merge_mode='zero')
        assert pos(inputs).shape == torch.Size((1, self.test_seq_length, config.hidden_size))

        pos = SinusoidalPositionEmbedding(output_dim=config.hidden_size, merge_mode='concat')
        assert pos(inputs).shape == torch.Size((self.test_batch_size * 2, self.test_seq_length, config.hidden_size))
