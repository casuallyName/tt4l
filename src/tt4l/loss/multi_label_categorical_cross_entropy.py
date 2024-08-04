# @Time     : 2024/7/4 16:56
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["multi_label_categorical_cross_entropy_loss", "MultiLabelCategoricalCrossEntropy"]

import torch
from torch.nn.modules.loss import _Loss


def multi_label_categorical_cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor,
                                               reduction: str = 'mean') -> torch.Tensor:
    """多标签分类的交叉熵
    说明：targets和inputs的shape一致，targets的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证inputs的值域是全体实数，换言之一般情况下inputs
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出inputs大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    inputs = (1 - 2 * targets) * inputs
    inputs_neg = inputs - targets * 1e12
    inputs_pos = inputs - (1 - targets) * 1e12
    zeros = torch.zeros_like(inputs[..., :1])
    inputs_pos = torch.cat([inputs_pos, zeros], dim=-1)
    inputs_neg = torch.cat([inputs_neg, zeros], dim=-1)
    neg_loss = torch.logsumexp(inputs_neg, dim=-1)
    pos_loss = torch.logsumexp(inputs_pos, dim=-1)
    if reduction == 'mean':
        return (neg_loss + pos_loss).mean()
    elif reduction == 'sum':
        return (neg_loss + pos_loss).sum()
    else:
        return neg_loss + pos_loss


class MultiLabelCategoricalCrossEntropy(_Loss):
    """
    Reference:
        [1] https://spaces.ac.cn/archives/7359

    多标签分类的交叉熵
    说明：targets和inputs的shape一致，targets的元素非0即1，1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证inputs的值域是全体实数，换言之一般情况下input不用加激活函数，尤其是不能加sigmoid或者softmax！
         预测阶段则输出inputs大于0的类。如有疑问，请仔细阅读并理解本文。
    """

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MultiLabelCategoricalCrossEntropy, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return multi_label_categorical_cross_entropy_loss(inputs, targets, reduction=self.reduction)
