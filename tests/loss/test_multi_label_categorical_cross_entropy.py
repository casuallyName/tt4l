# -*- coding: utf-8 -*-
# @Time     : 2024/7/27 22:26
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import torch

from utils.testing_utils import BaseTestCase


class PositionTestCase(BaseTestCase):
    test_num_classes = 5

    def test_multi_label_categorical_cross_entropy(self):
        self.model_import_test('tt4l.loss.multi_label_categorical_cross_entropy.MultiLabelCategoricalCrossEntropy')
        self.model_import_test(
            'tt4l.loss.multi_label_categorical_cross_entropy.multi_label_categorical_cross_entropy_loss')
        from tt4l import MultiLabelCategoricalCrossEntropy, multi_label_categorical_cross_entropy_loss

        size = (self.test_batch_size, self.test_seq_length, self.test_num_classes)
        x = torch.randn(size)
        y = torch.where(torch.randint(0, self.test_num_classes - 1, size) > 0, 1, 0)

        criterion = MultiLabelCategoricalCrossEntropy()
        assert criterion(x, y) is not None

        criterion = MultiLabelCategoricalCrossEntropy(reduction='none')
        assert criterion(x, y).shape == torch.Size((self.test_batch_size, self.test_seq_length))

        criterion = MultiLabelCategoricalCrossEntropy(reduction='sum')
        assert criterion(x, y) is not None

        criterion = MultiLabelCategoricalCrossEntropy(reduction='mean')
        assert criterion(x, y) is not None

        loss = multi_label_categorical_cross_entropy_loss(x, y)
        assert loss is not None
