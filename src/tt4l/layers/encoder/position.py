# @Time     : 2024/7/4 18:10
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ['SinusoidalPositionEmbedding']

from typing import Literal

import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos位置Embedding
    Reference:
        https://github.com/bojone/bert4keras/blob/70a7eb9ace18b9f4806b6386e5183f32d024bc37/bert4keras/layers.py#L849
    """

    def __init__(
            self,
            output_dim: int,
            merge_mode: Literal['add', 'mul', 'zero', 'concat'] = 'add',
            custom_position_ids: bool = False
    ) -> None:
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs) -> torch.Tensor:
        input_shape = inputs.shape
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)
        else:
            if not self.custom_position_ids:
                embeddings = torch.tile(embeddings, [batch_size, 1, 1])
            return torch.concatenate([inputs, embeddings])
