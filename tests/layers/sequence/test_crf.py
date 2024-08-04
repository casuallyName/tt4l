# @Time     : 2024/7/19 18:41
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import random
import unittest

import torch
from tt4l.utils.testing_utils import BaseTestCase


class CRFTestCase(BaseTestCase):
    test_num_tags = 3

    def test_CRF(self):
        self.model_import_test('tt4l.layers.sequence.CRF')
        from tt4l import CRF
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        emissions = torch.randn(self.test_batch_size, self.test_seq_length, self.test_num_tags, requires_grad=True)

        model_inputs = self.fake_inputs()
        attention_mask = model_inputs['attention_mask']
        for i in range(self.test_batch_size):
            attention_mask[i][-(i + 1):] = 0

        tags = torch.LongTensor([[random.randint(0, self.test_num_tags - 1) if i else -100 for i in mask]
                                 for mask in attention_mask])

        emissions = emissions.to(device)
        tags = tags.to(device)
        model = CRF(self.test_num_tags, batch_first=True).to(device)

        _ = model(emissions, tags, attention_mask=attention_mask)
        result = model.decode(emissions, attention_mask=attention_mask, padding=True)
        assert torch.Tensor(result).shape == attention_mask.shape


if __name__ == '__main__':
    unittest.main(verbosity=2)
