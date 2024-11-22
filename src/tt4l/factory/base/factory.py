# @Time     : 2024/7/22 11:50
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["DatasetType", "BaseTaskFactory"]

import os
from typing import Callable, Union, Optional, Any

import datasets
import pandas as pd
import transformers
import transformers.utils.logging
from transformers import (trainer_utils, TrainingArguments, Trainer,
                          AutoConfig, AutoTokenizer,
                          PreTrainedTokenizer, PretrainedConfig, PreTrainedModel, )
from transformers.trainer import logger

from factory.base.arguments import BaseTaskArguments, BasePredictArguments

DatasetType = Union[datasets.Dataset, datasets.DatasetDict, datasets.IterableDatasetDict, datasets.IterableDataset]


class _BaseTaskUtilFactory:
    @staticmethod
    def set_logger(training_args: TrainingArguments):
        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()
            log_level = training_args.get_process_log_level()

            datasets.utils.logging.set_verbosity(log_level)
            transformers.utils.logging.set_verbosity(log_level)
            transformers.utils.logging.enable_default_handler()
            transformers.utils.logging.enable_explicit_format()

    @staticmethod
    def set_seed(seed: int):
        trainer_utils.set_seed(seed)

    def load_last_checkpoint(self, training_args: TrainingArguments, **kwargs):
        if os.path.isdir(
                training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = trainer_utils.get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
            return last_checkpoint


class _BaseTaskDatasetFactory:
    def before_load_datasets(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            **kwargs
    ):
        pass

    def load_datasets(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            **kwargs
    ) -> Union[datasets.DatasetDict, datasets.IterableDatasetDict]:
        raw_datasets = datasets.load_dataset(
            task_args.dataset_name,
            task_args.dataset_config_name,
            cache_dir=task_args.cache_dir,
            token=task_args.token,
            trust_remote_code=task_args.trust_remote_code,
        )

        if task_args.max_train_samples:
            raw_datasets["train"] = raw_datasets["train"].select(range(task_args.max_train_samples))
        if task_args.max_eval_samples:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(task_args.max_eval_samples))
        if task_args.max_predict_samples:
            raw_datasets["test"] = raw_datasets["test"].select(range(task_args.max_predict_samples))
        return raw_datasets

    def after_load_dataset(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            **kwargs
    ):
        pass

    # Predict 相关

    def load_data_from_file_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            **kwargs):
        file_name, file_type = os.path.splitext(predict_args.data_path)
        if file_type.lower() in ['.xlsx', '.xls', '.et']:
            data = pd.read_excel(predict_args.data_path)
        else:
            data = pd.read_csv(predict_args.data_path)

        return data, data.columns.tolist()

    def init_predict_dataset_from_dataframe(
            self, training_args: TrainingArguments, predict_args: BasePredictArguments, data: pd.DataFrame,
            data_preprocess_function: Callable, **kwargs
    ) -> DatasetType:
        raise NotImplementedError

    def after_load_data_from_file_for_predict(self, training_args: TrainingArguments,
                                              predict_args: BasePredictArguments,
                                              data: pd.DataFrame, **kwargs):
        pass


class _BaseTaskModelFactory:

    def before_load_config(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            **kwargs
    ):
        pass

    def load_config(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            **kwargs
    ) -> PretrainedConfig:
        return AutoConfig.from_pretrained(
            task_args.config_name if task_args.config_name else task_args.model_name_or_path,
            cache_dir=task_args.cache_dir,
            revision=task_args.model_revision,
            use_auth_token=True if task_args.token else None,
            trust_remote_code=task_args.trust_remote_code,
            **kwargs
        )

    def after_load_config(self,
                          training_args: TrainingArguments,
                          task_args: Union[BaseTaskArguments, BasePredictArguments],
                          config: PretrainedConfig,
                          **kwargs):
        pass

    def load_pretrained_config(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            **kwargs
    ) -> PretrainedConfig:
        raise NotImplementedError

    def before_load_tokenizer(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            config: PretrainedConfig,
            **kwargs):
        pass

    def load_tokenizer(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            config: PretrainedConfig,
            **kwargs
    ) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(
            task_args.tokenizer_name if task_args.tokenizer_name else task_args.model_name_or_path,
            cache_dir=task_args.cache_dir,
            use_fast=task_args.use_fast_tokenizer,
            revision=task_args.model_revision,
            use_auth_token=True if task_args.token else None,
            trust_remote_code=task_args.trust_remote_code,
            **kwargs
        )

    def after_load_tokenizer(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ):
        pass

    def before_load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ):
        pass

    def load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ) -> PreTrainedModel:
        raise NotImplementedError

    def after_load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ):
        pass


class _BaseTaskDataProcessFactory:

    def before_init_data_preprocess_function(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ):
        pass

    def init_data_preprocess_function(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        raise NotImplementedError

    def after_init_data_preprocess_function(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            **kwargs
    ):
        pass

    def before_data_preprocess(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            **kwargs):
        pass

    def data_preprocess(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            **kwargs):
        train_dataset = None
        eval_dataset = None
        predict_dataset = None
        with training_args.main_process_first(desc="dataset map pre-processing"):
            if training_args.do_train:
                train_dataset = raw_datasets['train']
                if task_args.max_train_samples:
                    train_dataset = train_dataset.select(range(task_args.max_train_samples))
                train_dataset = train_dataset.map(
                    data_preprocess_function,
                    batched=True,
                    num_proc=task_args.preprocessing_num_workers,
                    load_from_cache_file=not task_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            if training_args.do_eval:
                eval_dataset = raw_datasets['validation']
                if task_args.max_eval_samples:
                    eval_dataset = eval_dataset.select(range(task_args.max_eval_samples))
                eval_dataset = eval_dataset.map(
                    data_preprocess_function,
                    batched=True,
                    num_proc=task_args.preprocessing_num_workers,
                    load_from_cache_file=not task_args.overwrite_cache,
                    desc="Running tokenizer on eval dataset",
                )
            if training_args.do_predict:
                predict_dataset = raw_datasets['test']
                if task_args.max_predict_samples:
                    predict_dataset = predict_dataset.select(range(task_args.max_predict_samples))
                predict_dataset = predict_dataset.map(
                    data_preprocess_function,
                    batched=True,
                    num_proc=task_args.preprocessing_num_workers,
                    load_from_cache_file=not task_args.overwrite_cache,
                    desc="Running tokenizer on predict dataset",
                )

        return train_dataset, eval_dataset, predict_dataset

    def after_data_preprocess(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            **kwargs
    ):
        pass

    def before_init_data_collator_function(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            **kwargs
    ):
        pass

    def init_data_collator_function(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            **kwargs
    ) -> Callable:
        raise NotImplementedError

    def after_init_data_collator_function(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            **kwargs
    ):
        pass

    # Predict 相关

    def before_init_data_preprocess_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ):
        pass

    def init_data_preprocess_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        raise NotImplementedError

    def after_init_data_preprocess_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            **kwargs
    ):
        pass

    def before_data_preprocess_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            **kwargs):
        pass

    def data_preprocess_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            **kwargs):
        with training_args.main_process_first(desc="dataset map pre-processing"):
            predict_dataset = raw_datasets['predict']
            predict_dataset = predict_dataset.map(data_preprocess_function,
                                                  batched=True,
                                                  load_from_cache_file=not predict_args.overwrite_cache,
                                                  desc="Running tokenizer on dataset",
                                                  remove_columns=self._predict_columns
                                                  )
        return predict_dataset

    def after_data_preprocess_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            **kwargs
    ):
        pass

    def before_init_data_collator_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            **kwargs
    ):
        pass

    def init_data_collator_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            **kwargs
    ) -> Callable:
        raise NotImplementedError

    def after_init_data_collator_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            **kwargs
    ):
        pass

    def save_result_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: BasePredictArguments,
            data: pd.DataFrame,
            result: Any,
            config: Optional[PretrainedConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            model: Optional[PreTrainedModel] = None,
    ) -> str:
        raise NotImplementedError


class _BaseTaskMetricsFactory:
    def before_load_metrics(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            **kwargs
    ):
        pass

    def load_metrics(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            **kwargs
    ) -> Callable:
        raise NotImplementedError

    def after_load_metrics(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            compute_metrics_function: Optional[Callable] = None,
            **kwargs
    ):
        pass


class _BaseTaskTrainerFactory:

    def before_init_trainer(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            compute_metrics_function: Optional[Callable] = None,
            **kwargs
    ):
        pass

    def init_trainer(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            compute_metrics_function: Optional[Callable] = None,
            **kwargs
    ):
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics_function,
            tokenizer=tokenizer,
            data_collator=data_collator_function,
            **kwargs
        )

    def after_init_trainer(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            data_preprocess_function: Optional[Callable] = None,
            train_dataset=None,
            eval_dataset=None,
            predict_dataset=None,
            data_collator_function: Optional[Callable] = None,
            compute_metrics_function: Optional[Callable] = None,
            trainer: Optional[Trainer] = None,
            **kwargs
    ):
        pass

    def do_trainer_train(
            self,
            training_args: TrainingArguments,
            task_args: BaseTaskArguments,
            last_checkpoint,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            train_dataset,
            trainer: Trainer,
            **kwargs
    ):
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            task_args.max_train_samples if task_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def do_trainer_eval(self,
                        training_args: TrainingArguments,
                        task_args: BaseTaskArguments,
                        last_checkpoint,
                        raw_datasets: DatasetType,
                        config: PretrainedConfig,
                        tokenizer: PreTrainedTokenizer,
                        model: PreTrainedModel,
                        eval_dataset,
                        trainer: Trainer,
                        **kwargs
                        ):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = task_args.max_eval_samples if task_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    def do_trainer_predict(
            self,
            training_args: TrainingArguments,
            task_args: Union[BaseTaskArguments, BasePredictArguments],
            last_checkpoint,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            predict_dataset,
            trainer: Trainer,
            save_metrics: bool = True,
            **kwargs
    ) -> trainer_utils.PredictionOutput:
        if "labels" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("labels")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", predictions.metrics)
        if save_metrics:
            trainer.save_metrics("predict", predictions.metrics)
        return predictions

    def predict_result_parser(self,
                              training_args: TrainingArguments,
                              task_args: Union[BaseTaskArguments, BasePredictArguments],
                              last_checkpoint,
                              raw_datasets: DatasetType,
                              config: PretrainedConfig,
                              tokenizer: PreTrainedTokenizer,
                              model: PreTrainedModel,
                              predict_dataset,
                              trainer: Trainer,
                              predictions: trainer_utils.PredictionOutput,
                              **kwargs
                              ) -> Any:
        raise NotImplementedError

    def save_trainer_predict_result(self,
                                    training_args: TrainingArguments,
                                    task_args: Union[BaseTaskArguments, BasePredictArguments],
                                    trainer: Trainer,
                                    predict_dataset,
                                    result: Any
                                    ):
        raise NotImplementedError

    def final_of_trainer_training(self,
                                  training_args,
                                  task_args,
                                  config,
                                  tokenizer,
                                  model,
                                  trainer
                                  ):
        kwargs = {
            "finetuned_from": task_args.model_name_or_path,
            "tasks": self.tasks_name
        }
        if task_args.dataset_name is not None:
            kwargs["dataset_tags"] = task_args.dataset_name
            if task_args.dataset_config_name is not None:
                kwargs["dataset_tags"] = task_args.dataset_config_name
                kwargs["dataset"] = f"{task_args.dataset_name} {task_args.dataset_config_name}"
            else:
                kwargs["dataset"] = task_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


class BaseTaskFactory(_BaseTaskUtilFactory,
                      _BaseTaskDatasetFactory,
                      _BaseTaskModelFactory,
                      _BaseTaskDataProcessFactory,
                      _BaseTaskMetricsFactory,
                      _BaseTaskTrainerFactory
                      ):
    tasks_name: str
    description: str
    description_zh: str
    default_args_yaml_name: str

    task_args_cls: Callable[[Any], BaseTaskArguments]
    predict_args_cls: Callable[[Any], BasePredictArguments]

    _last_checkpoint = None
    _raw_datasets = None
    _train_dataset = None
    _eval_dataset = None
    _predict_dataset = None
    _compute_metrics = None
    _preprocess_function = None
    _data_collator = None
    _model = None
    _tokenizer = None
    _config = None
    _trainer = None
    _predict_data = None
    _predict_columns = None

    # 默认推理方法
    @staticmethod
    def inference(*args, **kwargs):
        pass

    # 训练主入口
    def train(self, task_args: BaseTaskArguments, training_args: TrainingArguments):
        self.set_logger(training_args=training_args)

        logger.info(task_args)
        logger.info(training_args)

        self._last_checkpoint = self.load_last_checkpoint(training_args=training_args)

        self.set_seed(training_args.seed)

        # 加载数据
        self.before_load_datasets(
            training_args=training_args,
            task_args=task_args
        )
        self._raw_datasets = self.load_datasets(training_args=training_args,
                                                task_args=task_args)
        self.after_load_dataset(
            training_args=training_args,
            task_args=task_args,
            raw_datasets=self._raw_datasets
        )

        # 加载 Config
        self.before_load_config(training_args=training_args, task_args=task_args)
        self._config = self.load_pretrained_config(training_args=training_args, task_args=task_args)
        self.after_load_config(training_args=training_args, task_args=task_args, config=self._config)

        # 加载 Tokenizer
        self.before_load_tokenizer(training_args=training_args, task_args=task_args, config=self._config)
        self._tokenizer = self.load_tokenizer(training_args=training_args, task_args=task_args, config=self._config)
        self.after_load_tokenizer(training_args=training_args, task_args=task_args, config=self._config,
                                  tokenizer=self._tokenizer)

        # 加载模型
        self.before_load_model(training_args=training_args, task_args=task_args, config=self._config,
                               tokenizer=self._tokenizer)
        self._model = self.load_model(training_args=training_args, task_args=task_args, config=self._config,
                                      tokenizer=self._tokenizer)
        self.after_load_model(training_args=training_args, task_args=task_args, config=self._config,
                              tokenizer=self._tokenizer, model=self._model)

        # 加载 DataPreProcess
        self.before_init_data_preprocess_function(training_args=training_args, task_args=task_args,
                                                  raw_datasets=self._raw_datasets, config=self._config,
                                                  tokenizer=self._tokenizer, model=self._model)
        self._preprocess_function = self.init_data_preprocess_function(training_args=training_args, task_args=task_args,
                                                                       raw_datasets=self._raw_datasets,
                                                                       config=self._config,
                                                                       tokenizer=self._tokenizer,
                                                                       model=self._model)
        self.after_init_data_preprocess_function(
            training_args=training_args,
            task_args=task_args,
            raw_datasets=self._raw_datasets,
            config=self._config,
            tokenizer=self._tokenizer,
            model=self._model,
            data_preprocess_function=self._preprocess_function
        )

        # 处理数据
        self.before_data_preprocess(training_args=training_args, task_args=task_args, raw_datasets=self._raw_datasets,
                                    config=self._config, tokenizer=self._tokenizer, model=self._model,
                                    data_preprocess_function=self._preprocess_function
                                    )
        self._train_dataset, self._eval_dataset, self._predict_dataset = self.data_preprocess(
            training_args=training_args, task_args=task_args, raw_datasets=self._raw_datasets,
            config=self._config, tokenizer=self._tokenizer, model=self._model,
            data_preprocess_function=self._preprocess_function
        )
        self.after_data_preprocess(training_args=training_args, task_args=task_args, raw_datasets=self._raw_datasets,
                                   config=self._config, tokenizer=self._tokenizer, model=self._model,
                                   data_preprocess_function=self._preprocess_function,
                                   train_dataset=self._train_dataset, eval_dataset=self._eval_dataset,
                                   predict_dataset=self._predict_dataset,
                                   )

        # 初始化 DataCollator
        self.before_init_data_collator_function(training_args=training_args, task_args=task_args,
                                                raw_datasets=self._raw_datasets,
                                                config=self._config, tokenizer=self._tokenizer, model=self._model,
                                                data_preprocess_function=self._preprocess_function,
                                                train_dataset=self._train_dataset, eval_dataset=self._eval_dataset,
                                                predict_dataset=self._predict_dataset,
                                                )
        self._data_collator = self.init_data_collator_function(training_args=training_args, task_args=task_args,
                                                               raw_datasets=self._raw_datasets,
                                                               config=self._config, tokenizer=self._tokenizer,
                                                               model=self._model,
                                                               data_preprocess_function=self._preprocess_function,
                                                               train_dataset=self._train_dataset,
                                                               eval_dataset=self._eval_dataset,
                                                               predict_dataset=self._predict_dataset,
                                                               )
        self.after_init_data_collator_function(training_args=training_args, task_args=task_args,
                                               raw_datasets=self._raw_datasets,
                                               config=self._config, tokenizer=self._tokenizer, model=self._model,
                                               data_preprocess_function=self._preprocess_function,
                                               train_dataset=self._train_dataset,
                                               eval_dataset=self._eval_dataset,
                                               predict_dataset=self._predict_dataset,
                                               data_collator_function=self._data_collator)

        # 加载 Metrics
        self.before_load_metrics(training_args=training_args, task_args=task_args,
                                 raw_datasets=self._raw_datasets,
                                 config=self._config, tokenizer=self._tokenizer, model=self._model,
                                 data_preprocess_function=self._preprocess_function,
                                 train_dataset=self._train_dataset,
                                 eval_dataset=self._eval_dataset,
                                 predict_dataset=self._predict_dataset,
                                 data_collator_function=self._data_collator)
        self._compute_metrics = self.load_metrics(training_args=training_args, task_args=task_args,
                                                  raw_datasets=self._raw_datasets,
                                                  config=self._config, tokenizer=self._tokenizer, model=self._model,
                                                  data_preprocess_function=self._preprocess_function,
                                                  train_dataset=self._train_dataset,
                                                  eval_dataset=self._eval_dataset,
                                                  predict_dataset=self._predict_dataset,
                                                  data_collator_function=self._data_collator)
        self.after_load_metrics(training_args=training_args, task_args=task_args,
                                raw_datasets=self._raw_datasets,
                                config=self._config, tokenizer=self._tokenizer, model=self._model,
                                data_preprocess_function=self._preprocess_function,
                                train_dataset=self._train_dataset,
                                eval_dataset=self._eval_dataset,
                                predict_dataset=self._predict_dataset,
                                data_collator_function=self._data_collator,
                                compute_metrics=self._compute_metrics)

        # 初始化 Trainer
        self.before_init_trainer(training_args=training_args, task_args=task_args,
                                 raw_datasets=self._raw_datasets,
                                 config=self._config, tokenizer=self._tokenizer, model=self._model,
                                 data_preprocess_function=self._preprocess_function,
                                 train_dataset=self._train_dataset,
                                 eval_dataset=self._eval_dataset,
                                 predict_dataset=self._predict_dataset,
                                 data_collator_function=self._data_collator,
                                 compute_metrics_function=self._compute_metrics)
        self._trainer = self.init_trainer(training_args=training_args, task_args=task_args,
                                          raw_datasets=self._raw_datasets,
                                          config=self._config, tokenizer=self._tokenizer, model=self._model,
                                          data_preprocess_function=self._preprocess_function,
                                          train_dataset=self._train_dataset,
                                          eval_dataset=self._eval_dataset,
                                          predict_dataset=self._predict_dataset,
                                          data_collator_function=self._data_collator,
                                          compute_metrics_function=self._compute_metrics)
        self.after_init_trainer(training_args=training_args, task_args=task_args,
                                raw_datasets=self._raw_datasets,
                                config=self._config, tokenizer=self._tokenizer, model=self._model,
                                data_preprocess_function=self._preprocess_function,
                                train_dataset=self._train_dataset,
                                eval_dataset=self._eval_dataset,
                                predict_dataset=self._predict_dataset,
                                data_collator_function=self._data_collator,
                                compute_metrics_function=self._compute_metrics, trainer=self._trainer)

        if training_args.do_train:
            self.do_trainer_train(training_args=training_args, task_args=task_args,
                                  last_checkpoint=self._last_checkpoint,
                                  raw_datasets=self._raw_datasets,
                                  config=self._config, tokenizer=self._tokenizer, model=self._model,
                                  data_preprocess_function=self._preprocess_function,
                                  train_dataset=self._train_dataset,
                                  data_collator_function=self._data_collator,
                                  compute_metrics_function=self._compute_metrics, trainer=self._trainer
                                  )

        if training_args.do_eval:
            self.do_trainer_eval(training_args=training_args, task_args=task_args,
                                 last_checkpoint=self._last_checkpoint,
                                 raw_datasets=self._raw_datasets,
                                 config=self._config, tokenizer=self._tokenizer, model=self._model,
                                 data_preprocess_function=self._preprocess_function,
                                 eval_dataset=self._eval_dataset,
                                 data_collator_function=self._data_collator,
                                 compute_metrics_function=self._compute_metrics, trainer=self._trainer
                                 )

        if training_args.do_predict:
            predictions = self.do_trainer_predict(training_args=training_args, task_args=task_args,
                                                  last_checkpoint=self._last_checkpoint,
                                                  raw_datasets=self._raw_datasets,
                                                  config=self._config, tokenizer=self._tokenizer, model=self._model,
                                                  data_preprocess_function=self._preprocess_function,
                                                  predict_dataset=self._predict_dataset,
                                                  data_collator_function=self._data_collator,
                                                  compute_metrics_function=self._compute_metrics, trainer=self._trainer)

            result = self.predict_result_parser(training_args=training_args, task_args=task_args,
                                                last_checkpoint=self._last_checkpoint,
                                                raw_datasets=self._raw_datasets,
                                                config=self._config, tokenizer=self._tokenizer,
                                                model=self._model,
                                                data_preprocess_function=self._preprocess_function,
                                                predict_dataset=self._predict_dataset,
                                                data_collator_function=self._data_collator,
                                                compute_metrics_function=self._compute_metrics,
                                                trainer=self._trainer,
                                                predictions=predictions
                                                )
            self.save_trainer_predict_result(training_args=training_args, task_args=task_args,
                                             trainer=self._trainer, predict_dataset=self._predict_dataset,
                                             result=result
                                             )

        self.final_of_trainer_training(
            training_args=training_args,
            task_args=task_args,
            config=self._config,
            tokenizer=self._tokenizer,
            model=self._model,
            trainer=self._trainer
        )

    # 预测主入口
    def predict(self, predict_args: BasePredictArguments):
        training_args = TrainingArguments(output_dir=predict_args.result_output_dir,
                                          per_device_eval_batch_size=predict_args.per_device_batch_size,
                                          report_to=predict_args.report_to)
        self.set_logger(training_args=training_args)

        if predict_args.result_output_dir is None or predict_args.result_output_dir == "":
            raise ValueError(f"result_output_dir must be specified!")
        if not os.path.exists(predict_args.result_output_dir):
            os.makedirs(predict_args.result_output_dir)

        self._predict_data, self._predict_columns = self.load_data_from_file_for_predict(training_args=training_args,
                                                                                         predict_args=predict_args)
        self.after_load_data_from_file_for_predict(training_args=training_args, predict_args=predict_args,
                                                   data=self._predict_data)

        # 加载 Config
        self._config = self.load_config(training_args=training_args, task_args=predict_args)
        # 加载 Tokenizer
        self._tokenizer = self.load_tokenizer(training_args=training_args, task_args=predict_args, config=self._config)
        # 加载模型
        self._model = self.load_model(training_args=training_args, task_args=predict_args, config=self._config,
                                      tokenizer=self._tokenizer)

        # 初始化 DataPreProcess
        self.before_init_data_preprocess_function_for_predict(training_args=training_args,
                                                              predict_args=predict_args,
                                                              raw_datasets=self._raw_datasets,
                                                              config=self._config,
                                                              tokenizer=self._tokenizer,
                                                              model=self._model)
        self._preprocess_function = self.init_data_preprocess_function_for_predict(training_args=training_args,
                                                                                   predict_args=predict_args,
                                                                                   raw_datasets=self._raw_datasets,
                                                                                   config=self._config,
                                                                                   tokenizer=self._tokenizer,
                                                                                   model=self._model)
        self.after_init_data_preprocess_function_for_predict(training_args=training_args,
                                                             predict_args=predict_args,
                                                             raw_datasets=self._raw_datasets,
                                                             config=self._config,
                                                             tokenizer=self._tokenizer,
                                                             model=self._model,
                                                             data_preprocess_function=self._preprocess_function)

        # DataFrame -> DatasetDict
        self._raw_datasets = self.init_predict_dataset_from_dataframe(
            training_args=training_args,
            predict_args=predict_args,
            data=self._predict_data,
            data_preprocess_function=self._preprocess_function)
        self._predict_dataset = self.data_preprocess_for_predict(training_args=training_args, predict_args=predict_args,
                                                                 raw_datasets=self._raw_datasets,
                                                                 config=self._config, tokenizer=self._tokenizer,
                                                                 model=self._model,
                                                                 data_preprocess_function=self._preprocess_function)

        # 初始化 DataCollator
        self.before_init_data_collator_function_for_predict(training_args=training_args, predict_args=predict_args,
                                                            raw_datasets=self._raw_datasets,
                                                            config=self._config, tokenizer=self._tokenizer,
                                                            model=self._model,
                                                            predict_dataset=self._predict_dataset
                                                            )

        self._data_collator = self.init_data_collator_function_for_predict(training_args=training_args,
                                                                           predict_args=predict_args,
                                                                           raw_datasets=self._raw_datasets,
                                                                           config=self._config,
                                                                           tokenizer=self._tokenizer,
                                                                           model=self._model,
                                                                           )
        self.after_init_data_collator_function_for_predict(training_args=training_args, predict_args=predict_args,
                                                           raw_datasets=self._raw_datasets,
                                                           config=self._config, tokenizer=self._tokenizer,
                                                           model=self._model,
                                                           data_preprocess_function=self._preprocess_function)

        # 初始化 Trainer
        self.before_init_trainer(
            training_args=training_args, task_args=predict_args, raw_datasets=self._raw_datasets,
            config=self._config, tokenizer=self._tokenizer, model=self._model,
            data_preprocess_function=self._preprocess_function,
            predict_dataset=self._predict_dataset
        )
        self._trainer = self.init_trainer(
            training_args=training_args, task_args=predict_args, raw_datasets=self._raw_datasets,
            config=self._config, tokenizer=self._tokenizer, model=self._model,
            data_preprocess_function=self._preprocess_function,
            predict_dataset=self._predict_dataset,
            data_collator_function=self._data_collator,
        )
        self.after_init_trainer(
            training_args=training_args, task_args=predict_args, raw_datasets=self._raw_datasets,
            config=self._config, tokenizer=self._tokenizer, model=self._model,
            data_preprocess_function=self._preprocess_function,
            predict_dataset=self._predict_dataset,
            data_collator_function=self._data_collator,
            trainer=self._trainer
        )

        # Predict
        predictions = self.do_trainer_predict(training_args=training_args, task_args=predict_args,
                                              last_checkpoint=None, raw_datasets=self._raw_datasets,
                                              config=self._config, tokenizer=self._tokenizer, model=self._model,
                                              predict_dataset=self._predict_dataset,
                                              trainer=self._trainer,
                                              save_metrics=False)

        # 解析结果
        result = self.predict_result_parser(training_args=training_args, task_args=predict_args,
                                            last_checkpoint=None,
                                            raw_datasets=self._raw_datasets,
                                            config=self._config, tokenizer=self._tokenizer,
                                            model=self._model,
                                            data_preprocess_function=self._preprocess_function,
                                            predict_dataset=self._predict_dataset,
                                            data_collator_function=self._data_collator,
                                            compute_metrics_function=self._compute_metrics,
                                            trainer=self._trainer,
                                            predictions=predictions
                                            )

        # 保存数据
        result_path = self.save_result_for_predict(training_args=training_args, predict_args=predict_args,
                                                   data=self._predict_data, result=result,
                                                   config=self._config, tokenizer=self._tokenizer, model=self._model
                                                   )
        return result_path
