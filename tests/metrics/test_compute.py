# @Time     : 2024/8/2 18:36
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import warnings

from utils.testing_utils import BaseTestCase


class ToDo(FutureWarning):
    ...


class TextClassificationComputeTestCase(BaseTestCase):
    test_num_classes = 5

    def test_ComputeMseMetrics(self):
        self.model_import_test('tt4l.metrics.compute.text_classification.ComputeMseMetrics')
        warnings.warn("Need add more testing methods.", ToDo)

    def test_ComputeAccuracyMetrics(self):
        self.model_import_test('tt4l.metrics.compute.text_classification.ComputeAccuracyMetrics')
        warnings.warn("Need add more testing methods.", ToDo)

    def test_ComputeF1Metrics(self):
        self.model_import_test('tt4l.metrics.compute.text_classification.ComputeF1Metrics')
        warnings.warn("Need add more testing methods.", ToDo)
