# @Time     : 2024/7/4 17:01
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :

__all__ = ['HiddenStatesPooling', 'PoolingStrategy']

from typing import Union, Dict

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .utils import PoolingStrategy


class HiddenStatesPooling(nn.Module):

    def __init__(self, pool_strategy: Union[str, PoolingStrategy] = PoolingStrategy.FL_AVG_POOLING,
                 cls_in_first: bool = True,
                 hidden_state_name: str = 'hidden_states',
                 last_hidden_state_name: str = 'last_hidden_state'):
        """

        Args:
            pool_strategy: `Union[str, PoolingStrategy]`,default to PoolingStrategy.FL_AVG_POOLING
            cls_in_first: `bool`,default to True
            hidden_state_name: `str`,default to 'hidden_states'
            last_hidden_state_name: `str`,default to 'last_hidden_state'
        """
        super().__init__()
        if isinstance(pool_strategy, PoolingStrategy):
            pool_strategy = pool_strategy.value

        self._pool_strategy = PoolingStrategy(pool_strategy)
        self.cls_in_first = cls_in_first
        self.hidden_state_name = hidden_state_name
        self.last_hidden_state_name = last_hidden_state_name

    @property
    def cls_index(self):
        return 0 if self.cls_in_first else -1

    @property
    def pool_strategy(self):
        return self._pool_strategy

    @pool_strategy.setter
    def pool_strategy(self, pool_strategy: Union[str, PoolingStrategy]):
        if isinstance(pool_strategy, PoolingStrategy):
            pool_strategy = pool_strategy.value

        self._pool_strategy = PoolingStrategy(pool_strategy)

    @property
    def output_hidden_states(self):
        return True if self._pool_strategy == PoolingStrategy.FL_AVG_POOLING else False

    def extra_repr(self) -> str:
        return f"pool_strategy='{self._pool_strategy.value}', cls_in_first={self.cls_in_first}"

    def forward(self, output: Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict], attention_mask: torch.Tensor):
        """

        Args:
            output: `Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict]`
            attention_mask: `torch.Tensor`

        Returns:
            torch.Tensor
        """
        return getattr(self, self._pool_strategy.value)(output=output, attention_mask=attention_mask)

    def cls(self, output: Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict], attention_mask: torch.Tensor):
        """
        返回最后一层隐层的CLS所在位置的 hidden state

        Args:
            output: `Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict]`
            attention_mask: `torch.Tensor`

        Returns:
            torch.Tensor
        """
        return output[self.last_hidden_state_name][:, self.cls_index]

    def mean(self, output: Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict], attention_mask: torch.Tensor):
        """
        返回最后一层隐层的均值

        Args:
            output: `Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict]`
            attention_mask: `torch.Tensor`

        Returns:
            torch.Tensor
        """
        hidden_state = torch.sum(output[self.last_hidden_state_name] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hidden_state / attention_mask

    def max(self, output: Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict], attention_mask: torch.Tensor):
        """
        返回最后一层隐层的最大值

        Args:
            output: `Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict]`
            attention_mask: `torch.Tensor`

        Returns:
            torch.Tensor
        """
        hid = output[self.last_hidden_state_name] * attention_mask[:, :, None]
        return hid.max(dim=-2).values

    def first_last_avg(self, output: Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict],
                       attention_mask: torch.Tensor):
        """
        返回第一层隐层和最后一层隐层的均值

        Args:
            output: `Union[BaseModelOutputWithPoolingAndCrossAttentions, Dict]`
            attention_mask: `torch.Tensor`

        Returns:
            torch.Tensor
        """
        hid = torch.sum(output[self.hidden_state_name][1] * attention_mask[:, :, None], dim=1)
        hid += torch.sum(output[self.last_hidden_state_name][-1] * attention_mask[:, :, None], dim=1)
        attention_mask = torch.sum(attention_mask, dim=1)[:, None]
        return hid / (2 * attention_mask)
