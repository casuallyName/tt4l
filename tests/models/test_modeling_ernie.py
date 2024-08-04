# @Time     : 2024/8/2 16:36
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import random
import unittest

import torch
from tt4l.utils.testing_utils import BaseTestCase


class ModelingErnieTestCase(BaseTestCase):

    def config(self, **kwargs):
        from transformers import ErnieConfig
        return ErnieConfig(**kwargs)

    def test_ErnieForUniversalInformationExtraction(self):
        self.model_import_test('tt4l.models.modeling_ernie.ErnieForUniversalInformationExtraction')
        from tt4l import ErnieForUniversalInformationExtraction

        model_inputs = self.fake_inputs()

        model = ErnieForUniversalInformationExtraction(self.config())

        model_inputs['start_positions'] = torch.zeros(self.test_batch_size, self.test_seq_length)
        model_inputs['end_positions'] = torch.zeros(self.test_batch_size, self.test_seq_length)
        predictions = model(**model_inputs)

        assert predictions.loss is not None

    def test_ErnieForMultiLabelSequenceClassification(self):
        self.model_import_test('tt4l.models.modeling_ernie.ErnieForMultiLabelSequenceClassification')

        from tt4l import ErnieForMultiLabelSequenceClassification

        model_inputs = self.fake_inputs()

        model = ErnieForMultiLabelSequenceClassification(self.config(pool_strategy='first_last_avg'))

        model_inputs['labels'] = torch.FloatTensor([[0., 1.]] * self.test_batch_size)
        predictions = model(**model_inputs)

        assert predictions.loss is not None

    def test_ErnieWithHiddenPoolingForSequenceClassification(self):
        self.model_import_test('tt4l.models.modeling_ernie.ErnieWithHiddenPoolingForSequenceClassification')

        from tt4l import ErnieWithHiddenPoolingForSequenceClassification

        model_inputs = self.fake_inputs()

        model = ErnieWithHiddenPoolingForSequenceClassification(self.config(pool_strategy='first_last_avg'))
        assert hasattr(model, 'hidden_states_pooling')
        assert model.hidden_states_pooling.pool_strategy.value == 'first_last_avg'

        # 测试单标签
        model_inputs['labels'] = torch.LongTensor([0] * self.test_batch_size)
        predictions = model(**model_inputs)
        assert predictions.logits.shape == torch.Size([self.test_batch_size, model.config.num_labels])
        assert predictions.loss is not None

        # 测试Multi-Label
        model.config.problem_type = None
        model_inputs['labels'] = torch.FloatTensor([[0., 1.]] * self.test_batch_size)
        predictions = model(**model_inputs)
        assert predictions.loss is not None
        assert predictions.logits.shape == torch.Size([self.test_batch_size, model.config.num_labels])

    def test_ErnieWithCrfForTokenClassification(self):
        self.model_import_test('tt4l.models.modeling_ernie.ErnieWithCrfForTokenClassification')
        from tt4l import ErnieWithCrfForTokenClassification
        model_inputs = self.fake_inputs()

        model = ErnieWithCrfForTokenClassification(self.config(num_labels=3))

        assert hasattr(model, 'crf')
        model_inputs['attention_mask'][0][-1] = 0

        predictions = model(**model_inputs)
        assert predictions.decode_label is not None

        model_inputs['labels'] = torch.LongTensor([[random.randint(0, 2) if i else -100 for i in mask]
                                                   for mask in model_inputs['attention_mask']])
        assert predictions.logits.shape == torch.Size(
            [self.test_batch_size, self.test_seq_length, model.config.num_labels])

        predictions = model(**model_inputs)
        assert predictions.loss is not None

    def test_BertForEntitySpanClassification(self):
        self.model_import_test('tt4l.models.modeling_ernie.ErnieForEntitySpanClassification')
        from tt4l import ErnieForEntitySpanClassification
        model_inputs = self.fake_inputs()

        model = ErnieForEntitySpanClassification(self.config(num_labels=3))

        assert hasattr(model, 'pointers')
        assert len(model.pointers) == model.config.num_labels

        model_inputs['labels'] = torch.FloatTensor([
                                                       [
                                                           [
                                                               [0., 0.]  # (2,) ->  start, end
                                                           ] * model.config.num_labels  # (num_labels, 2)
                                                       ] * self.test_seq_length  # (seq_length, num_labels, 2)
                                                   ] * self.test_batch_size)  # (batch_size, seq_length, num_labels, 2)
        predictions = model(**model_inputs)
        assert predictions.logits.shape == torch.Size(
            [self.test_batch_size, self.test_seq_length, model.config.num_labels, 2])
        assert predictions.loss is not None


if __name__ == '__main__':
    unittest.main(verbosity=2)
