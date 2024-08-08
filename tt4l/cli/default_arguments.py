# @Time     : 2024/7/11 17:53
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from dataclasses import dataclass
from typing import Optional, List, Union

from transformers import TrainingArguments


@dataclass
class DefaultTrainingArguments:
    output_dir: str
    overwrite_output_dir: bool = TrainingArguments.overwrite_output_dir

    do_train: bool = True
    do_eval: bool = TrainingArguments.do_eval
    do_predict: bool = TrainingArguments.do_predict

    resume_from_checkpoint: Optional[str] = TrainingArguments.resume_from_checkpoint

    per_device_train_batch_size: int = TrainingArguments.per_device_train_batch_size
    per_device_eval_batch_size: int = TrainingArguments.per_device_eval_batch_size

    auto_find_batch_size: bool = TrainingArguments.auto_find_batch_size

    gradient_accumulation_steps: int = TrainingArguments.gradient_accumulation_steps
    eval_accumulation_steps: Optional[int] = TrainingArguments.eval_accumulation_steps

    num_train_epochs: float = TrainingArguments.num_train_epochs

    lr_scheduler_type: str = TrainingArguments.lr_scheduler_type

    learning_rate: float = TrainingArguments.learning_rate

    weight_decay: float = TrainingArguments.weight_decay
    adam_beta1: float = TrainingArguments.adam_beta1
    adam_beta2: float = TrainingArguments.adam_beta2
    adam_epsilon: float = TrainingArguments.adam_epsilon
    max_grad_norm: float = TrainingArguments.max_grad_norm
    warmup_ratio: float = TrainingArguments.warmup_ratio
    warmup_steps: int = TrainingArguments.warmup_steps

    # TrainingArguments.evaluation_strategy
    if hasattr(TrainingArguments, 'eval_strategy'):
        eval_strategy: str = 'epoch'
    else:
        evaluation_strategy: str = 'epoch'

    eval_delay: Optional[float] = TrainingArguments.eval_delay
    eval_steps: Optional[float] = TrainingArguments.eval_steps

    logging_strategy: str = TrainingArguments.logging_strategy
    logging_steps: float = TrainingArguments.logging_steps

    save_strategy: str = TrainingArguments.save_strategy
    save_steps: int = TrainingArguments.save_steps

    save_total_limit: Optional[int] = TrainingArguments.save_total_limit
    save_safetensors: Optional[bool] = TrainingArguments.save_safetensors

    load_best_model_at_end: Optional[bool] = TrainingArguments.load_best_model_at_end
    greater_is_better: Optional[bool] = TrainingArguments.greater_is_better

    report_to: Union[None, str, List[str]] = None
