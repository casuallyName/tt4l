# @Time     : 2024/8/2 18:36
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import numpy as np
from transformers import EvalPrediction

from utils.testing_utils import BaseTestCase


class TextClassificationComputeMetricsTestCase(BaseTestCase):

    def test_ComputeMetricsForMse(self):
        self.model_import_test('tt4l.metrics.compute.text_classification.ComputeMetricsForMse')
        from tt4l import ComputeMetricsForMse
        compute_metrics = ComputeMetricsForMse()
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        metrics = compute_metrics(
            EvalPrediction(predictions=y_pred, label_ids=y_true)
        )
        assert metrics['mse'] == 0.375

    def test_ComputeMetricsForAccuracy(self):
        self.model_import_test('tt4l.metrics.compute.text_classification.ComputeMetricsForAccuracy')
        from tt4l import ComputeMetricsForAccuracy
        compute_metrics = ComputeMetricsForAccuracy()
        y_true = np.array([0, 0, 1, 0])
        y_pred = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        metrics = compute_metrics(
            EvalPrediction(predictions=y_pred, label_ids=y_true)
        )
        assert metrics['accuracy'] == 0.75

    def test_ComputeMetricsForF1(self):
        self.model_import_test('tt4l.metrics.compute.text_classification.ComputeMetricsForF1')
        from tt4l import ComputeMetricsForF1
        compute_metrics = ComputeMetricsForF1()
        y_true = np.array([[1, 0], [1, 0], [1, 0], [1, 0]])
        y_pred = np.array([[0.8, 0.2], [0.8, 0.2], [0.8, 0.2], [0.8, 0.2]])
        # check -> np.where(p > 0, 1, 0)  -> 0.5
        metrics = compute_metrics(
            EvalPrediction(predictions=y_pred, label_ids=y_true)
        )
        assert metrics['f1'] == 0.5


class TokenClassificationComputeMetricsTestCase(BaseTestCase):

    def test_ComputeMetricsWithSeqeval(self):
        self.model_import_test('tt4l.metrics.compute.token_classification.ComputeMetricsWithSeqeval')
        from tt4l import ComputeMetricsWithSeqeval
        y_true = np.array([[0, 0, 0, 1, 2, 2, 0], [3, 4, 0, -100, -100, -100, -100]])
        y_pred = np.array([[[0.5, 0, 0, 0, 0],
                            [0.5, 0, 0, 0, 0],
                            [0, 0.5, 0, 0, 0],
                            [0, 0, 0.5, 0, 0],
                            [0, 0, 0.5, 0, 0],
                            [0, 0, 0.5, 0, 0],
                            [0.5, 0, 0, 0, 0]],
                           [[0, 0, 0, 0.5, 0],
                            [0, 0, 0, 0, 0.5],
                            [0.5, 0, 0, 0, 0],
                            [0.5, 0, 0, 0, 0],
                            [0.5, 0, 0, 0, 0],
                            [0.5, 0, 0, 0, 0],
                            [0.5, 0, 0, 0, 0]]])
        label_list = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER']

        compute_metrics = ComputeMetricsWithSeqeval(label_list=label_list)
        metrics = compute_metrics(EvalPrediction(predictions=y_pred, label_ids=y_true))
        assert metrics['accuracy'] == 0.8
        assert metrics['f1'] == 0.5
        assert metrics['precision'] == 0.5
        assert metrics['recall'] == 0.5


class UniversalInformationExtractionComputeMetricsTestCase(BaseTestCase):
    def test_ComputeUniversalInformationExtractionMetrics(self):
        self.model_import_test(
            'tt4l.metrics.compute.universal_information_extraction.ComputeMetricsForUniversalInformationExtraction')
        from tt4l import ComputeMetricsForUniversalInformationExtraction
        compute_metrics = ComputeMetricsForUniversalInformationExtraction()
        y_true_start = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        y_true_end = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        y_true = (y_true_start, y_true_end,)
        y_pred_start = np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        y_pred_end = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]])
        y_pred = (y_pred_start, y_pred_end,)
        metrics = compute_metrics(EvalPrediction(predictions=y_true, label_ids=y_pred))
        assert metrics['precision'] == 0.5
        assert metrics['recall'] == 0.5
        assert metrics['f1'] == 0.5
