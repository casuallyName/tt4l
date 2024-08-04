# @Time     : 2024/7/4 18:07
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ['Pointer', 'EntitySpan', 'EntityPointers']

from typing import Iterable, Optional

import torch
import torch.nn as nn


class Pointer(nn.Module):
    def __init__(self, config, num_labels: int = 1) -> None:
        super(Pointer, self).__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, num_labels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        return x


class EntitySpan(nn.Module):
    def __init__(self, config) -> None:
        super(EntitySpan, self).__init__()
        self.start = Pointer(config, 1)
        self.end = Pointer(config, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.start(x), self.end(x)], dim=-1)


class EntityPointers(nn.ModuleList):
    def __init__(self, modules: Optional[Iterable[EntitySpan]]) -> None:
        super().__init__()
        self.extend(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([span_pointer(x) for span_pointer in self], dim=-2)
