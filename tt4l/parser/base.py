# -*- coding: utf-8 -*-
# @Time     : 2024/7/23 22:43
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from typing import Dict, List, Union, Literal

import numpy as np
from pydantic import BaseModel


class BaseParser:
    def __init__(self, id2label: Dict[int, str]):
        self._id2label = id2label

    def __call__(self, texts: List[str], predictions: Union[List, np.array], show_progress: bool = False, **kwargs):
        return self.parser(texts=texts, predictions=predictions, show_progress=show_progress, **kwargs)

    def parser(self, texts: List[str], predictions: Union[List, np.array], show_progress: bool = False, **kwargs):
        raise NotImplementedError


class Node(BaseModel):
    text: str
    label: str
    start: int
    end: int


class Nodes(List[Node]):
    def __repr__(self):
        return f'EntityNodes({super().__repr__()})'

    def append(self, entity: Node):
        super().append(entity)

    def to_dict(self, mode: Literal['json', 'python'] = 'python'):
        return [i.model_dump(mode=mode) for i in self]


if __name__ == '__main__':
    node = Node(text='朝阳区', label='district', start=0, end=3)
    entities = Nodes([node])
    print(entities)
    entities.append(node)
    print(entities.to_dict())
