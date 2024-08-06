# @Time     : 2024/7/19 14:26
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["TokenClassificationFactory"]

import json
import os
from typing import Union, Callable, Optional, Any

import datasets
import numpy as np
import pandas as pd
from datasets import ClassLabel
from transformers import (
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    DataCollatorForTokenClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
    trainer_utils
)
from transformers.trainer import logger

from tt4l.factory.base import DatasetType, BaseTaskFactory
from tt4l.factory.token_classification import (
    TokenClassificationTaskArguments,
    TokenClassificationPredictArguments,
    DataPreProcessForTokenClassification,
    utils
)
from tt4l.metrics.compute.token_classification import ComputeMetricsWithSeqeval
from tt4l.parser.sequence_parser import SequenceParser


class TokenClassificationFactory(BaseTaskFactory):
    tasks_name = "token-classification"
    description = ("Token classification assigns a label to individual tokens in a sentence. "
                   "One of the most common task is Named Entity Recognition (NER)。")
    description_zh = "Token分类任务，为文本中的每一个Token标记一个标签，常见实体识别任务（NER）"
    default_args_yaml_name = 'token_classification_task.yaml'

    task_args_cls = TokenClassificationTaskArguments
    predict_args_cls = TokenClassificationPredictArguments

    def __init__(self):
        self.token_column_name = None
        self.label_column_name = None
        self.label_list = None
        self.num_labels = None
        self.label_is_name = False
        self.load_model = self._load_token_classification_model

    def after_load_dataset(
            self,
            training_args: TrainingArguments,
            task_args: Union[TokenClassificationTaskArguments, TokenClassificationPredictArguments],
            raw_datasets: DatasetType,
            **kwargs
    ):
        features = raw_datasets["train"].features
        if task_args.token_column_name is not None:
            self.token_column_name = task_args.token_column_name
        else:
            self.token_column_name = [k for k, v in features.items() if isinstance(v, datasets.Sequence)][0]

        if task_args.label_column_name is not None:
            self.label_column_name = task_args.label_column_name
        else:
            self.label_column_name = [k for k, v in features.items() if isinstance(v, datasets.Sequence)][1]

        self.label_is_name = not isinstance(raw_datasets['train'].features[self.label_column_name].feature, ClassLabel)
        if self.label_is_name:
            self.label_list = utils.get_label_list(raw_datasets["train"][self.label_column_name])
        else:
            self.label_list = raw_datasets['train'].features[self.label_column_name].feature.names

        self.num_labels = len(self.label_list)

    def load_pretrained_config(
            self,
            training_args: TrainingArguments,
            task_args: TokenClassificationTaskArguments,
            **kwargs
    ) -> PretrainedConfig:
        return self.load_config(
            training_args=training_args,
            task_args=task_args,
            num_labels=self.num_labels,
        )

    def after_load_tokenizer(
            self,
            training_args: TrainingArguments,
            task_args: Union[TokenClassificationTaskArguments, TokenClassificationPredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ):
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
                " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
                " this requirement"
            )

    def after_load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[TokenClassificationTaskArguments, TokenClassificationPredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ):
        if model.config.label2id != PretrainedConfig(num_labels=len(self.label_list)).label2id:
            if sorted(model.config.label2id.keys()) != sorted(self.label_list):
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
                    f" {sorted(self.label_list)}.\nIgnoring the model labels as a result.",
                )
        else:
            model.config.label2id = {label: idx for idx, label in enumerate(self.label_list)}
            model.config.id2label = {idx: label for idx, label in enumerate(self.label_list)}
            config.label2id = model.config.label2id
            config.id2label = model.config.id2label

    def init_data_preprocess_function(
            self,
            training_args: TrainingArguments,
            task_args: Union[TokenClassificationTaskArguments, TokenClassificationPredictArguments],
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        return DataPreProcessForTokenClassification(
            tokenizer=tokenizer,
            padding=("max_length"
                     if task_args.pad_to_max_length else
                     False),
            max_seq_length=task_args.max_seq_length,
            label_to_id=model.config.label2id,
            token_key=task_args.token_column_name,
            label_key=task_args.label_column_name,
            label_all_tokens=task_args.label_all_tokens,
            label_is_name=self.label_is_name,
            add_special_tokens=not hasattr(model, 'crf'),
            ignore_token_id=-100,
        )

    def init_data_collator_function(
            self,
            training_args: TrainingArguments,
            task_args: TokenClassificationTaskArguments,
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
        return DataCollatorForTokenClassification(tokenizer,
                                                  pad_to_multiple_of=8 if training_args.fp16 else None)

    def load_metrics(
            self,
            training_args: TrainingArguments,
            task_args: TokenClassificationTaskArguments,
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
        return ComputeMetricsWithSeqeval(
            label_list=self.label_list,
            return_entity_level_metrics=task_args.return_entity_level_metrics
        )

    def after_init_trainer(
            self,
            training_args: TrainingArguments,
            task_args: Union[TokenClassificationTaskArguments, TokenClassificationTaskArguments],
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
        from tt4l.optimizer_builder import OptimizerBuilderForCrf
        # 放大crf层学习率
        logger.warning(
            "Use `OptimizerBuilderForCrf.get_optimizer_cls_and_kwargs` instead of `get_optimizer_cls_and_kwargs`.")
        trainer.get_optimizer_cls_and_kwargs = OptimizerBuilderForCrf()

    def predict_result_parser(self,
                              training_args: TrainingArguments,
                              task_args: Union[TokenClassificationTaskArguments, TokenClassificationTaskArguments],
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
        if isinstance(predictions.predictions, tuple):
            predictions = predictions.predictions[-1]
        else:
            predictions = np.argmax(predictions.predictions, axis=2)
        # Remove ignored index (special tokens)
        if predict_dataset.features.get('word_ids'):
            true_predictions = [
                [p for (p, m) in zip(prediction, attention_mask) if m is not None]
                for prediction, attention_mask in zip(predictions, predict_dataset['word_ids'])
            ]
        else:
            true_predictions = [
                [p for (p, l) in zip(prediction, labels) if l != -100]
                for prediction, labels in zip(predictions, predict_dataset['labels'])
            ]
        return true_predictions

    def save_trainer_predict_result(self,
                                    training_args: TrainingArguments,
                                    task_args: Union[
                                        TokenClassificationTaskArguments, TokenClassificationPredictArguments],
                                    trainer: Trainer,
                                    predict_dataset,
                                    result: Any
                                    ):
        output_predictions_file = os.path.join(training_args.output_dir, "predict_results.jsonl")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, 'w', encoding='utf-8') as f:
                for index, predictions in enumerate(result):
                    row = json.dumps({'idx': index, 'predictions': predictions}, ensure_ascii=False)
                    f.write(f"{row}\n")
            logger.info("Predict results saved at {}".format(output_predictions_file))

    def after_load_data_from_file_for_predict(self, training_args: TrainingArguments,
                                              predict_args: TokenClassificationPredictArguments,
                                              data: pd.DataFrame, **kwargs):
        if predict_args.text_column_name not in data.columns:
            raise ValueError(f"Column {predict_args.text_column_name} not in data!")

    def init_data_preprocess_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: TokenClassificationPredictArguments,
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        return DataPreProcessForTokenClassification(tokenizer=tokenizer,
                                                    padding=("max_length"
                                                             if predict_args.pad_to_max_length else
                                                             False),
                                                    max_seq_length=predict_args.max_seq_length,
                                                    label_to_id=model.config.label2id,
                                                    token_key=predict_args.text_column_name,
                                                    label_key=None,
                                                    add_special_tokens=not hasattr(model, 'crf')
                                                    )

    def init_predict_dataset_from_dataframe(
            self, training_args: TrainingArguments, predict_args: TokenClassificationPredictArguments,
            data: pd.DataFrame,
            data_preprocess_function: Callable, **kwargs
    ) -> DatasetType:
        data[predict_args.text_column_name] = (data[predict_args.text_column_name]
                                               .fillna('')
                                               .apply(lambda text: [word for word in str(text)]))

        _columns = [predict_args.text_column_name]
        dataset = datasets.Dataset.from_pandas(data.fillna('').loc[:, _columns])
        # data[predict_args.text_column_name] = (data[predict_args.text_column_name]
        #                                        .apply(lambda words: ''.join(words)))
        return datasets.DatasetDict({'predict': dataset})

    def init_data_collator_function_for_predict(
            self,
            training_args: TrainingArguments,
            predict_args: TokenClassificationPredictArguments,
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
        return DataCollatorForTokenClassification(tokenizer)

    def save_result_for_predict(
            self, training_args: TrainingArguments,
            predict_args: TokenClassificationPredictArguments,
            data: pd.DataFrame,
            result: Any,
            config: Optional[PretrainedConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            model: Optional[PreTrainedModel] = None,
    ) -> str:
        parser = SequenceParser(sequence_type='BIO', id2label=config.id2label)
        data[predict_args.text_column_name] = (data[predict_args.text_column_name]
                                               .apply(lambda words: ''.join(words)))
        results = parser.parser(texts=data[predict_args.text_column_name].tolist(),
                                predictions=result,
                                show_progress=True
                                )
        data['predictions'] = [json.dumps(result.to_dict(), ensure_ascii=False) for result in results]
        result_output_path = os.path.join(predict_args.result_output_dir, 'predictions.csv')
        data.to_csv(result_output_path, index=False, encoding='utf-8-sig')
        return result_output_path
