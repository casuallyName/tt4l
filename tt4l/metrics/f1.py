# @Time     : 2024/7/19 16:44
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from tt4l.output_decoders.base import BaseDecoder
from tt4l.metrics.base import BaseMetric



class EntityOverallF1Metric:
    """
    overall f1 metric for ner.
    """

    description = "f1 metric for ner."

    def __init__(self, label_list: List[str], decoder: BaseDecoder, return_entity_level_metrics: bool = False):
        super(EntityOverallF1Metric, self).__init__()
        self._label_list = label_list
        self._num_labels = len(label_list)
        self.decoder = decoder
        self.return_entity_level_metrics = return_entity_level_metrics

    @classmethod
    def get_p_r_f(cls, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return np.array([p, r, f1])

    @classmethod
    def calculate_metric(cls, entities_predict, entities_true):
        """
        计算 tp fp fn
        """
        tp, fp, fn = 0, 0, 0
        for entity_predict in entities_predict:
            flag = 0
            for entity_true in entities_true:
                if entity_predict[0] == entity_true[0] and entity_predict[1] == entity_true[1]:
                    flag = 1
                    tp += 1
                    break
            if flag == 0:
                fp += 1

        fn = len(entities_true) - tp

        return np.array([tp, fp, fn])

    def get_role_metric(self, pred_entities, true_entities):
        total_count = [0 for _ in range(self._num_labels)]
        role_metric = np.zeros([self._num_labels, 3])

        for _pred, _true in zip(pred_entities, true_entities):
            # print("========================")
            # print(_pred)
            # print(_true)
            # print("========================")
            for idx, _type in enumerate(self._label_list):
                if _type not in _pred:
                    _pred[_type] = []
                total_count[idx] += len(_true.get(_type, []))
                role_metric[idx] += self.calculate_metric(_pred.get(_type, []), _true.get(_type, []))

        return role_metric, total_count

    def get_metric_result(self, pred_entities, true_entities):
        role_metric, _ = self.get_role_metric(pred_entities=pred_entities, true_entities=true_entities)

        mirco_metrics = np.sum(role_metric, axis=0)
        mirco_metrics = self.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])

        return dict(
            precision=mirco_metrics[0].item(),
            recall=mirco_metrics[1].item(),
            f1=mirco_metrics[2].item(),
        )

    def metric(self, predictions: np.array, labels: np.array, **kwargs) -> Dict:
        entities = self.decoder.eval_decoder(predictions=predictions, labels=labels)
        results = self.get_metric_result(pred_entities=entities.pred_entities, true_entities=entities.true_entities)
        if self.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return results
