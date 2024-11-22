# -*- coding: utf-8 -*-
# @Time     : 2024/7/23 22:41
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["SequenceType", "BioSequenceParser", "SequenceParser"]

from enum import Enum
from typing import Dict, List, Union

import numpy as np
from ..parser.base import BaseParser, Node, Nodes


class SequenceType(Enum):
    BIO = 'BIO'
    # BIEO = 'BIEO'


class BioSequenceParser(BaseParser):
    def __init__(self, id2label: Dict[int, str], is_decode_to_label: bool = False):
        super().__init__(id2label=id2label)
        self.is_decode_to_label = is_decode_to_label

    def row_decoder(self, text: str, prediction: Union[List, np.array]) -> Nodes:
        entities = Nodes()
        for index, (word, label) in enumerate(zip(text, prediction)):
            if not self.is_decode_to_label:
                label = self._id2label[label]
            if label.startswith('B'):
                entities.append(
                    Node(
                        text=word,
                        label=label.split('-')[1],
                        start=index,
                        end=index + 1
                    )
                )
            elif label.startswith('I'):
                if len(entities):
                    if entities[-1].label == label.split('-')[1]:
                        entities[-1].text += word
                        entities[-1].end += 1
                    else:
                        entities.append(Node(
                            text=word,
                            label=label.split('-')[1],
                            start=index,
                            end=index + 1
                        ))
                else:
                    entities.append(Node(
                        text=word,
                        label=label.split('-')[1],
                        start=index,
                        end=index + 1
                    ))
            else:
                if len(entities) and entities[-1].label == 'O':
                    entities[-1].text += word
                    entities[-1].end += 1
                else:
                    entities.append(Node(
                        text=word,
                        label='O',
                        start=index,
                        end=index + 1
                    ))
        return entities

    def parser(self, texts: List[str], predictions: Union[List, np.array],
               show_progress: bool = False, **kwargs) -> List[Nodes]:
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(zip(texts, predictions), total=len(texts))
        else:
            iterator = zip(texts, predictions)
        return [self.row_decoder(text, prediction) for text, prediction, in iterator]


class SequenceParser(BaseParser):
    def __init__(self, sequence_type: Union[SequenceType, str], id2label: Dict[int, str],
                 is_decode_to_label: bool = False):
        super().__init__(id2label=id2label)
        self.sequence_type = SequenceType(sequence_type) if isinstance(sequence_type, str) else sequence_type
        if self.sequence_type == SequenceType.BIO:
            self._parser = BioSequenceParser(id2label=self._id2label, is_decode_to_label=is_decode_to_label)

    def parser(self, texts: List[str], predictions: Union[List, np.array], show_progress: bool = False, **kwargs):
        return self._parser.parser(texts=texts, predictions=predictions, show_progress=show_progress, **kwargs)


if __name__ == '__main__':
    label2id = {
        "B-assist": 3,
        "B-cell_no": 17,
        "B-city": 5,
        "B-community": 9,
        "B-country": 25,
        "B-detail": 21,
        "B-dev_zone": 29,
        "B-district": 1,
        "B-floor_no": 15,
        "B-house_no": 31,
        "B-poi": 27,
        "B-prov": 11,
        "B-road": 23,
        "B-road_no": 13,
        "B-room_no": 33,
        "B-sub_poi": 19,
        "B-town": 7,
        "I-assist": 4,
        "I-cell_no": 18,
        "I-city": 6,
        "I-community": 10,
        "I-country": 26,
        "I-detail": 22,
        "I-dev_zone": 30,
        "I-district": 2,
        "I-floor_no": 16,
        "I-house_no": 32,
        "I-poi": 28,
        "I-prov": 12,
        "I-road": 24,
        "I-road_no": 14,
        "I-room_no": 34,
        "I-sub_poi": 20,
        "I-town": 8,
        "O": 0
    }
    _text = ["朝阳区惠新东街6号"]
    _predictions = [[label2id[i] for i in ["B-district", "I-district", "I-district",
                                           "B-road", "I-road", "I-road", "I-road",
                                           "B-road_no", "I-road_no"]]]
    parser = SequenceParser('BIO', {j: i for i, j in label2id.items()})
    print(parser.parser(_text, np.array(_predictions)))
