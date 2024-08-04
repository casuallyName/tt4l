# @Time     : 2024/8/2 16:36
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

import unittest

import torch

from tt4l.utils.testing_utils import BaseTestCase


class ModelingBertTestCase(BaseTestCase):

    def test_ErnieForUniversalInformationExtraction(self):
        self.model_import_test('tt4l.models.modeling_bert.ErnieForUniversalInformationExtraction')
        from tt4l import ErnieForUniversalInformationExtraction

        model_inputs = self.fake_inputs()

        model = ErnieForUniversalInformationExtraction(self.ernie_config())

        model_inputs['start_positions'] = torch.zeros(self.test_batch_size, self.test_seq_length)
        model_inputs['end_positions'] = torch.zeros(self.test_batch_size, self.test_seq_length)
        predictions = model(**model_inputs)

        assert predictions.loss is not None


if __name__ == '__main__':
    unittest.main(verbosity=2)
