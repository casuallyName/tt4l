# @Time     : 2024/7/10 18:15
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["TextClassificationFactory"]

import os
from typing import Callable, Union, Optional, Any, List

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from transformers import (
    TrainingArguments,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    trainer_utils,
    Trainer, PreTrainedTokenizerBase, AutoModelForSequenceClassification
)
from transformers.trainer import logger
from tt4l.factory.base import BaseTaskFactory, DatasetType
from tt4l.factory.text_classification import (
    DataPreProcessForSequenceClassification,
    TextClassificationPredictArguments,
    TextClassificationTaskArguments,
)
from tt4l.metrics.compute.text_classification import ComputeMetricsForAccuracy, ComputeMetricsForF1


class TextClassificationFactory(BaseTaskFactory):
    tasks_name = "text-classification"
    description = "Text classification is a common NLP task that assigns a label or class to text. "
    description_zh = "文本分类任务，为文本标记一个标签或分类"
    default_args_yaml_name = 'text_classification_task.yaml'

    task_args_cls = TextClassificationTaskArguments
    predict_args_cls = TextClassificationPredictArguments

    def __init__(self):
        self.labels = None
        self.label_feature_is_sequence = False

    @staticmethod
    def inference(
            texts: List[str],
            tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel,
            text_pairs: Optional[List[str]] = None, probable_score: float = 0, batch_size: int = 8,
            max_length: int = 512, show_progress: bool = False
    ) -> Union[List[str], List[float]]:
        """
        text classification inference
        Args:
            texts (`List[str]`): list of texts
            tokenizer (`PreTrainedTokenizerBase`): Tokenizer
            model (`PreTrainedModel`): Model
            text_pairs (`List[str]`, defaults to `None`): second sequence of texts
            probable_score (`float`, defaults to `0`): position prob for multi-label classification
            batch_size (`int`, defaults to `8`): Batch size pre inference
            max_length (`int`, defaults to `512`): maximum length of texts
            show_progress (`bool`, defaults to `False`): Whether to show progress bar

        Returns:
            Union[List[str], List[float]]

        """
        if text_pairs is not None:
            assert len(texts) == len(text_pairs)
        input_length = len(texts)
        encoded_inputs = tokenizer(
            text=texts,
            text_pair=text_pairs,
            truncation=True,
            max_length=max_length,
            padding="longest",
            add_special_tokens=True,
            return_tensors="np")
        if show_progress and input_length > batch_size:
            iterator = trange(0, input_length, batch_size, desc="Inference", leave=False, delay=5)
        else:
            iterator = range(0, input_length, batch_size)
        task_type = getattr(model.config, 'problem_type', None)
        if task_type is None:
            raise ValueError(f"No task type in found, 'model.config.problem_type' is None")
        if task_type not in ["regression", "single_label_classification", "multi_label_classification"]:
            raise ValueError(f"Unknown task type: {task_type}")
        model.eval()
        predictions = []
        for i in iterator:
            outputs = model(**{key: torch.LongTensor(value[i:i + batch_size]).to(model.device)
                               for key, value in encoded_inputs.items()})
            if task_type == 'multi_label_classification':
                predictions.extend([
                    ';;'.join([model.config.id2label[item[0]] for item in np.argwhere(prediction > probable_score)])
                    for prediction in outputs.logits.detach().cpu().numpy()])
            elif task_type == "single_label_classification":
                predictions.extend([model.config.id2label[item]
                                    for item in np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)])
            else:
                predictions.extend(outputs.logits.detach().cpu().tolist())

        return predictions

    def load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ):
        return AutoModelForSequenceClassification.from_pretrained(
            task_args.model_name_or_path,
            from_tf=bool(".ckpt" in task_args.model_name_or_path),
            config=config,
            cache_dir=task_args.cache_dir,
            revision=task_args.model_revision,
            use_auth_token=True if task_args.token else None,
            trust_remote_code=task_args.trust_remote_code,
            **kwargs
        )

    def after_load_dataset(
            self,
            training_args: TrainingArguments,
            task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
            raw_datasets: Union[datasets.Dataset, datasets.IterableDatasetDict],
            **kwargs
    ):
        label_feature = raw_datasets['train'].info.features[task_args.label_column_name]  # .names
        self.label_feature_is_sequence = isinstance(label_feature, datasets.Sequence)
        if self.label_feature_is_sequence:
            self.labels = label_feature.feature.names

        else:
            self.labels = label_feature.names

    def load_pretrained_config(
            self,
            training_args: TrainingArguments,
            task_args: TextClassificationTaskArguments,
            **kwargs
    ) -> PretrainedConfig:
        return self.load_config(
            training_args=training_args,
            task_args=task_args,
            num_labels=len(self.labels)
        )

    def load_metrics(
            self,
            training_args: TrainingArguments,
            task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
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
        if self.label_feature_is_sequence:
            _compute_metrics = ComputeMetricsForF1()
        else:
            _compute_metrics = ComputeMetricsForAccuracy()
        return _compute_metrics

    def after_load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ):
        if model.config.label2id != PretrainedConfig(num_labels=len(self.labels)).label2id:
            if sorted(model.config.label2id.keys()) != sorted(self.labels):
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {sorted(self._model.config.label2id.keys())}, dataset labels:"
                    f" {sorted(self.labels)}.\nIgnoring the model labels as a result.",
                )
        else:
            model.config.label2id = {label: idx for idx, label in enumerate(self.labels)}
            model.config.id2label = {idx: label for idx, label in enumerate(self.labels)}
            config.label2id = model.config.label2id
            config.id2label = model.config.id2label

    def init_data_preprocess_function(
            self,
            training_args: TrainingArguments,
            task_args: TextClassificationTaskArguments,
            raw_datasets: Union[datasets.DatasetDict, datasets.IterableDatasetDict],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        return DataPreProcessForSequenceClassification(tokenizer=tokenizer,
                                                       padding=("max_length"
                                                                if task_args.pad_to_max_length else
                                                                False),
                                                       max_seq_length=task_args.max_seq_length,
                                                       label_to_id=model.config.label2id,
                                                       sentence1_key=task_args.text_column_name,
                                                       sentence2_key=task_args.text_pair_column_name,
                                                       label_key=(task_args.label_column_name
                                                                  if isinstance(task_args,
                                                                                TextClassificationTaskArguments)
                                                                  else None)
                                                       )

    def init_data_collator_function(
            self,
            training_args: TrainingArguments,
            task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
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
        if task_args.pad_to_max_length:
            from transformers import default_data_collator

            data_collator = default_data_collator
        elif training_args.fp16:
            from transformers import DataCollatorWithPadding

            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        return data_collator

    def predict_result_parser(self,
                              training_args: TrainingArguments,
                              task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
                              last_checkpoint,
                              raw_datasets: DatasetType,
                              config: PretrainedConfig,
                              tokenizer: PreTrainedTokenizer,
                              model: PreTrainedModel,
                              predict_dataset,
                              trainer: Trainer,
                              predictions: trainer_utils.PredictionOutput,
                              **kwargs
                              ):
        predictions = predictions.predictions
        if model.config.problem_type == 'multi_label_classification':
            predictions = [
                ';;'.join([config.id2label[item[0]] for item in np.argwhere(prediction > 0)])
                for prediction in predictions]
        else:
            predictions = [config.id2label[item] for item in np.argmax(predictions, axis=1)]
        return predictions

    def save_trainer_predict_result(
            self,
            training_args: TrainingArguments,
            task_args: Union[TextClassificationTaskArguments, TextClassificationPredictArguments],
            trainer: Trainer,
            predict_dataset,
            result: Any
    ):
        output_predictions_file = os.path.join(training_args.output_dir, "predict_results.csv")
        if trainer.is_world_process_zero():
            data = pd.DataFrame({'text': predict_dataset[task_args.text_column_name]})
            if task_args.text_pair_column_name:
                data['text_pair'] = predict_dataset[task_args.text_pair_column_name]
            data['labels'] = result
            data.to_csv(output_predictions_file, index=False, encoding='utf-8-sig')

            logger.info("Predict results saved at {}".format(output_predictions_file))

    def after_load_predict_data(self, training_args: TrainingArguments,
                                predict_args: TextClassificationPredictArguments,
                                data: pd.DataFrame, **kwargs):
        self._predict_columns = [predict_args.text_column_name]
        if predict_args.text_column_name not in data.columns:
            raise ValueError(f"Column {predict_args.text_column_name} not in data!")
        if predict_args.text_pair_column_name is not None:
            self._predict_columns.append(predict_args.text_pair_column_name)
            if predict_args.text_pair_column_name not in data.columns:
                raise ValueError(f"Column {predict_args.text_column_name} not in data!")

    def init_predict_dataset_from_dataframe(
            self, training_args: TrainingArguments, predict_args: TextClassificationPredictArguments,
            data: pd.DataFrame,
            data_preprocess_function: Callable, **kwargs
    ) -> DatasetType:
        dataset = datasets.Dataset.from_pandas(data.fillna('').loc[:, self._predict_columns])
        return datasets.DatasetDict({'predict': dataset})

    def init_data_preprocess_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: TextClassificationPredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        return DataPreProcessForSequenceClassification(tokenizer=tokenizer,
                                                       padding=("max_length"
                                                                if predict_args.pad_to_max_length else
                                                                False),
                                                       max_seq_length=predict_args.max_seq_length,
                                                       label_to_id=model.config.label2id,
                                                       sentence1_key=predict_args.text_column_name,
                                                       sentence2_key=predict_args.text_pair_column_name,
                                                       label_key=None)

    def init_data_collator_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: TextClassificationPredictArguments,
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
        if predict_args.pad_to_max_length:
            from transformers import default_data_collator

            data_collator = default_data_collator
        elif training_args.fp16:
            from transformers import DataCollatorWithPadding

            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        return data_collator

    def save_result_for_predict(
            self, training_args: TrainingArguments,
            predict_args: TextClassificationPredictArguments,
            data: pd.DataFrame, result: Any,
            config: Optional[PretrainedConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            model: Optional[PreTrainedModel] = None,
    ) -> str:
        data['predictions'] = result
        result_output_path = os.path.join(predict_args.result_output_dir, 'predictions.csv')
        data.to_csv(result_output_path, index=False, encoding='utf-8-sig')
        return result_output_path
