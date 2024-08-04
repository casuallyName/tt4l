# @Time     : 2024/7/25 17:57
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["OptimizerBuilderForCrf"]

from typing import Optional, Tuple, Any

from transformers import Trainer, TrainingArguments, PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names


class _BaseOptimizerBuilder:
    def __call__(self, args: TrainingArguments, model: Optional[PreTrainedModel] = None) -> Tuple[Any, Any]:
        return self.get_optimizer_cls_and_kwargs(args=args, model=model)

    def get_optimizer_cls_and_kwargs(
            self, args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        raise NotImplementedError


class OptimizerBuilderForCrf(_BaseOptimizerBuilder):
    def __init__(self, crf_parameter_key: str = 'crf', crf_learning_rate: Optional[float] = None) -> None:
        super().__init__()
        self.crf_learning_rate = crf_learning_rate
        self.crf_parameter_key = crf_parameter_key

    def get_optimizer_cls_and_kwargs(
            self, args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.
            model (`transformers.modeling_bert.BertModel`, optional):

        """
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)

        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if self.crf_learning_rate is not None:
            crf_learning_rate = self.crf_learning_rate
        else:
            crf_learning_rate = args.learning_rate * 1000

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if
                    (n in decay_parameters and p.requires_grad and self.crf_parameter_key not in n)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if ('crf' in n)
                ],
                "weight_decay": 0.0,
                'lr': crf_learning_rate
            },
        ]

        optimizer_kwargs["optimizer_dict"] = optimizer_grouped_parameters

        return optimizer_cls, optimizer_kwargs
