# @Time     : 2024/7/26 15:21
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

# class SpanDecoder(BaseDecoder):
#     def __init__(self, id2label: Dict):
#         super().__init__(id2label=id2label)
#
#     def __call__(self, labels: np.array):
#         return self.decoder(labels)
#
#     def _get_true_labels(self):
#         return list(self.id2label.values())
#
#     def decoder(self, labels: np.array, **kwargs):
#         entities = []
#         for seq_labels in labels:
#             seq_entities = {}
#             for label_id, seq_sub_labels in enumerate(seq_labels.transpose((1, 0, 2))):
#                 label = self.id2label[label_id]
#                 for start_id, end_id in zip(np.argwhere(seq_sub_labels[:, 0] == 1),
#                                             np.argwhere(seq_sub_labels[:, 1] == 1)):
#                     seq_entities[label] = seq_entities.get(label, [])
#                     seq_entities[label].append([start_id[0], end_id[0]])
#             entities.append(seq_entities)
#         return entities
#
#     def eval_decoder(self, predictions: np.array, labels: np.array, **kwargs):
#         return DecoderOutput(
#             pred_entities=self.decoder(np.where(predictions > 0.5, 1, 0)),
#             true_entities=self.decoder(labels)
#         )
