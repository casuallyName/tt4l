# @Time     : 2024/8/2 11:40
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["UniversalInformationExtractionFactory"]

import json
import os
from typing import List, Dict, Union, Any, Callable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
    TrainingArguments,
    PretrainedConfig,
    PreTrainedTokenizer,
    DataCollatorForTokenClassification,
    AutoModel,
    Trainer,
    trainer_utils
)
from transformers.trainer import logger
from ...factory.base import BaseTaskFactory, DatasetType
from ...factory.universal_information_extraction.arguments import (
    UniversalInformationExtractionTaskArguments,
    UniversalInformationExtractionPredictArguments
)
from .data import DataPreProcessForUniversalInformationExtraction
from .modules import (
    PositionInfo,
    Schema,
    Result,
    SchemaForBatch
)
from ...metrics.compute.universal_information_extraction import ComputeMetricsForUniversalInformationExtraction
from ...text_utils import ChineseSentenceSpliter


class UniversalInformationExtractionFactory(BaseTaskFactory):
    tasks_name = "universal-information-extraction"
    description = ("The Unified Information Extraction Framework (UIE) "
                   "for general information extraction achieves unified modeling of tasks such as "
                   "entity extraction, relationship extraction, event extraction, and sentiment analysis, "
                   "and enables good transferability and generalization ability between different tasks.")
    description_zh = ("通用信息抽取统一框架UIE，"
                      "该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，"
                      "并使得不同任务间具备良好的迁移和泛化能力。")
    default_args_yaml_name = 'universal_information_extraction_task.yaml'

    task_args_cls = UniversalInformationExtractionTaskArguments
    predict_args_cls = UniversalInformationExtractionPredictArguments

    def __init__(self):
        self.entities_column_name = None
        self.text_column_name = None
        self.prompt_column_name = None

    @staticmethod
    def inference(
            prompts: List[str],
            texts: List[str],
            tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, batch_size: int = 8,
            position_prob: float = 0.5, max_length: int = 512, show_progress: bool = False
    ) -> List[List[PositionInfo]]:
        """
        uie inference

        Args:
            prompts (`List[str]`): Prompt
            texts (`List[str]`): Text
            tokenizer (`PreTrainedTokenizerBase`): Tokenizer
            model (`PreTrainedModel`): UIE model
            batch_size (`int`, default to `8`): Batch size pre inference
            position_prob (`float`, default to `0.5`): Position probability line
            max_length (`int`, default to `512`): Max length
            show_progress (`bool`, default to `False`): Show progress bar

        Returns:
            List[List[PositionInfo]]

        """
        assert len(texts) == len(prompts)
        input_length = len(prompts)
        encoded_inputs = tokenizer(
            text=prompts,
            text_pair=texts,
            stride=2,
            truncation=True,
            max_length=max_length,
            padding="longest",
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np")
        start_map = {s_id: [] for s_id in range(input_length)}
        end_map = {s_id: [] for s_id in range(input_length)}
        results = [[] for _ in range(input_length)]
        offset_maps = encoded_inputs.pop("offset_mapping")
        if show_progress and input_length > batch_size:
            iterator = trange(0, input_length, batch_size, desc="Inference", leave=False, delay=5)
        else:
            iterator = range(0, input_length, batch_size)
        model.eval()
        for i in iterator:
            outputs = model(**{key: torch.LongTensor(value[i:i + batch_size]).to(model.device)
                               for key, value in encoded_inputs.items()})
            for s_id, t_id in torch.argwhere(outputs.start_prob.detach() > position_prob).cpu().numpy():
                start_map[s_id + i].append((t_id, outputs.start_prob[s_id][t_id].detach().cpu().item()))
            for s_id, t_id in torch.argwhere(outputs.end_prob.detach() > position_prob).cpu().numpy():
                end_map[s_id + i].append((t_id, outputs.end_prob[s_id][t_id].detach().cpu().item()))
        for i in range(input_length):
            for start, end in zip(start_map[i], end_map[i]):
                score = (start[1] + end[1]) / 2
                token_type = encoded_inputs['token_type_ids'][i][start[0]]
                start = offset_maps[i][start[0]][0]
                end = offset_maps[i][end[0]][1]
                if token_type == 0:
                    results[i].append(
                        PositionInfo(
                            text=prompts[i][start:end],
                            score=score,
                            token_type=token_type)
                    )
                else:

                    results[i].append(
                        PositionInfo(
                            text=texts[i][start:end],
                            start=start,
                            end=end,
                            score=score,
                            token_type=token_type)
                    )

        return results

    @staticmethod
    def predict_single(schema: Union[Schema, List, Dict, str], text: str,
                       tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, split_text: bool = False,
                       inference_batch_size: int = 8, text_max_length: int = 512, token_max_length: int = 512,
                       truncation: bool = False, position_prob: float = 0.5, show_progress: bool = True,
                       ) -> Dict[str, Any]:
        """
        single predict by schema

        Args:
            schema (`Union[List, Dict, str]`):
            text (`str`):
            tokenizer (`PreTrainedTokenizerBase`):
            model (`PreTrainedModel`):
            split_text (`bool`, default to `False`): Split text
            inference_batch_size (`int`, default to `8`): Batch size of inference
            text_max_length (`int`, default to `512`): Max length of text (before tokenizer)
            token_max_length (`int`, default to `512`): Max length of token (after tokenizer)
            truncation (`bool`, default to `False`): Truncate if over max length
            position_prob (`float`, default to `0.5`): Position probability line
            show_progress (`bool`, default to `False`): Show progress bar

        Returns:

        """
        if isinstance(schema, Schema):
            tree = schema
        else:
            tree = Schema(schema)
        if split_text:
            spliter = ChineseSentenceSpliter(max_length=text_max_length, truncation=truncation)
            split_texts = spliter.split(text)
        else:
            split_texts = [(text, 0)]

        tree.reset_status(split_texts)
        for inputs in tree:
            inference_result = UniversalInformationExtractionFactory.inference(
                prompts=inputs.prompts,
                texts=inputs.texts,
                tokenizer=tokenizer,
                model=model,
                batch_size=inference_batch_size,
                position_prob=position_prob,
                max_length=token_max_length,
                show_progress=show_progress
            )

            results = {}
            for prompt, idx_result in zip(inputs.prompts, inference_result):
                for prob_info in idx_result:
                    results[prompt] = results.get(prompt, [])
                    results[prompt].append(
                        Result(
                            text=prob_info.text,
                            probability=prob_info.score,
                            start=prob_info.start,
                            end=prob_info.end,
                        ) if prob_info.token_type else
                        Result(
                            text=prob_info.text,
                            probability=prob_info.score,
                        )
                    )
            tree.step(results)
        return tree.result_dump()

    @staticmethod
    def predict_batch(schema: Union[SchemaForBatch, List, Dict, str], texts: List[str],
                      tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, split_text: bool = False,
                      inference_batch_size: int = 8, text_batch_size: int = 8,
                      text_max_length: int = 512, token_max_length: int = 512,
                      truncation: bool = False, position_prob: float = 0.5,
                      show_progress: bool = True, show_inference_progress: bool = False
                      ):
        """
        batch predict by schema

        Args:
            schema (`Union[List, Dict, str]`):
            texts (`List[str]`):
            tokenizer (`PreTrainedTokenizerBase`):
            model (`PreTrainedModel`):
            split_text (`bool`, default to `False`): Split text
            inference_batch_size (`int`, default to `8`): Batch size of inference
            text_batch_size (`int`, default to `8`): Batch size of text
            text_max_length (`int`, default to `512`): Max length of text (before tokenizer)
            token_max_length (`int`, default to `512`): Max length of token (after tokenizer)
            truncation (`bool`, default to `False`): Truncate if over max length
            position_prob (`float`, default to `0.5`): Position probability line
            show_progress (`bool`, default to `True`): Show progress bar
            show_inference_progress (`bool`, default to `False`): Show inference progress bar

        Returns:

        """
        if isinstance(schema, SchemaForBatch):
            tree = schema
        else:
            tree = SchemaForBatch(schema)
        spliter = ChineseSentenceSpliter(max_length=text_max_length, truncation=truncation)
        results = []
        if show_progress and len(texts) > text_batch_size:
            iterator = trange(0, len(texts), text_batch_size, desc="Batch", delay=5)
        else:
            iterator = range(0, len(texts), text_batch_size)
        for batch_start_idx in iterator:
            batch_texts = texts[batch_start_idx:batch_start_idx + text_batch_size]
            if split_text:
                split_texts = spliter.split_documents(batch_texts)
            else:
                split_texts = [(idx, text[:text_max_length], 0) for idx, text in enumerate(batch_texts)]
            tree.reset_status(split_texts)
            for batch_inputs in tree:
                batch_inference_results = UniversalInformationExtractionFactory.inference(
                    prompts=batch_inputs.prompts,
                    texts=batch_inputs.texts,
                    tokenizer=tokenizer,
                    model=model,
                    batch_size=inference_batch_size,
                    position_prob=position_prob,
                    max_length=token_max_length,
                    show_progress=show_inference_progress
                )
                batch_step_results = {document_id: {} for document_id in range(len(batch_texts))}
                for input_ids, (prompt, inference_results) in enumerate(
                        zip(batch_inputs.prompts, batch_inference_results)):
                    for inference_result in inference_results:
                        sentence_id_in_batch = batch_inputs.text_mapping[input_ids]
                        batch_step_results[sentence_id_in_batch][prompt] = (
                            batch_step_results[sentence_id_in_batch].get(prompt, [])
                        )
                        batch_step_results[sentence_id_in_batch][prompt].append(
                            Result(
                                text=inference_result.text,
                                probability=inference_result.score,
                                start=inference_result.start,
                                end=inference_result.end,
                            ) if inference_result.token_type else
                            Result(
                                text=inference_result.text,
                                probability=inference_result.score,
                            )
                        )
                tree.step(batch_step_results)
            results.extend(tree.result_dump())
        return results

    def load_model(
            self,
            training_args: TrainingArguments,
            task_args: Union[
                UniversalInformationExtractionTaskArguments, UniversalInformationExtractionPredictArguments],
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            **kwargs
    ) -> PreTrainedModel:
        return AutoModel.from_pretrained(
            task_args.model_name_or_path,
            from_tf=bool(".ckpt" in task_args.model_name_or_path),
            config=config,
            cache_dir=task_args.cache_dir,
            revision=task_args.model_revision,
            use_auth_token=True if task_args.token else None,
            trust_remote_code=task_args.trust_remote_code,
            **kwargs
        )

    def load_pretrained_config(
            self,
            training_args: TrainingArguments,
            task_args: UniversalInformationExtractionTaskArguments,
            **kwargs
    ) -> PretrainedConfig:
        return self.load_config(
            training_args=training_args,
            task_args=task_args,
        )

    def after_load_dataset(
            self,
            training_args: TrainingArguments,
            task_args: Union[UniversalInformationExtractionTaskArguments, UniversalInformationExtractionTaskArguments],
            raw_datasets: DatasetType,
            **kwargs
    ):
        features = raw_datasets["train"].features
        if task_args.prompt_column_name is not None:
            self.prompt_column_name = task_args.prompt_column_name
        else:
            self.prompt_column_name = [k for k, v in features.items()
                                       if isinstance(v, datasets.Value) and v.dtype == 'string'][0]

        if task_args.text_column_name is not None:
            self.text_column_name = task_args.text_column_name
        else:
            self.text_column_name = [k for k, v in features.items()
                                     if isinstance(v, datasets.Value) and v.dtype == 'string'][1]

        if task_args.entities_column_name is not None:
            self.entities_column_name = task_args.entities_column_name
        else:
            self.entities_column_name = [k for k, v in features.items() if isinstance(v, datasets.Sequence)][1]

    def init_data_preprocess_function(
            self,
            training_args: TrainingArguments,
            task_args: Union[UniversalInformationExtractionTaskArguments, UniversalInformationExtractionTaskArguments],
            raw_datasets: DatasetType,
            config: PretrainedConfig,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            **kwargs
    ) -> Callable:
        return DataPreProcessForUniversalInformationExtraction(
            tokenizer=tokenizer,
            padding=("max_length"
                     if task_args.pad_to_max_length else
                     False),
            max_seq_length=task_args.max_seq_length,
            label_to_id=model.config.label2id,
            prompt_key=task_args.prompt_column_name,
            text_key=task_args.text_column_name,
            entities_key=task_args.entities_column_name,
        )

    def init_data_collator_function(
            self,
            training_args: TrainingArguments,
            task_args: UniversalInformationExtractionTaskArguments,
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
            task_args: UniversalInformationExtractionTaskArguments,
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
        return ComputeMetricsForUniversalInformationExtraction()

    def after_init_trainer(
            self,
            training_args: TrainingArguments,
            task_args: Union[
                UniversalInformationExtractionTaskArguments,
                UniversalInformationExtractionPredictArguments
            ],
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
        # 添加label name
        trainer.label_names = ['start_positions', 'end_positions']

    def predict_result_parser(self,
                              training_args: TrainingArguments,
                              task_args: Union[
                                  UniversalInformationExtractionTaskArguments,
                                  UniversalInformationExtractionPredictArguments
                              ],
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
        predictions_entities = []
        start_positions, end_positions = predictions.predictions
        for prompt, content, token_type_id, offset_mapping, start_position, end_position in zip(
                predict_dataset['prompt'], predict_dataset['content'], predict_dataset['token_type_ids'],
                predict_dataset['offset_mapping'], start_positions, end_positions
        ):
            prediction_entities = []
            for start_id, end_id in zip(np.argwhere(start_position > .5).reshape(-1),
                                        np.argwhere(end_position > .5).reshape(-1)):
                true_start_id = offset_mapping[start_id][0]
                true_end_id = offset_mapping[end_id][1]
                if token_type_id[start_id] == 0:
                    prediction_entities.append(
                        {
                            'text': prompt[true_start_id:true_end_id]
                        }
                    )
                else:
                    prediction_entities.append(
                        {
                            'text': content[true_start_id:true_end_id],
                            'start': true_start_id,
                            'end': true_end_id
                        }
                    )
            predictions_entities.append(prediction_entities)
        return predictions_entities

    def save_trainer_predict_result(self,
                                    training_args: TrainingArguments,
                                    task_args: Union[
                                        UniversalInformationExtractionTaskArguments,
                                        UniversalInformationExtractionPredictArguments
                                    ],
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
                                              predict_args: UniversalInformationExtractionPredictArguments,
                                              data: pd.DataFrame, **kwargs):
        if predict_args.text_column_name not in data.columns:
            raise ValueError(f"Column {predict_args.text_column_name} not in data!")

    def save_result_for_predict(
            self, training_args: TrainingArguments,
            predict_args: UniversalInformationExtractionPredictArguments,
            data: pd.DataFrame,
            result: Any,
            config: Optional[PretrainedConfig] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            model: Optional[PreTrainedModel] = None,
    ) -> str:
        data['predictions'] = [json.dumps(r, ensure_ascii=False) for r in result]
        result_output_path = os.path.join(predict_args.result_output_dir, 'predictions.csv')
        data.to_csv(result_output_path, index=False, encoding='utf-8-sig')
        return result_output_path

    def predict(self, predict_args: UniversalInformationExtractionPredictArguments):
        if predict_args.schema is None:
            raise ValueError(f"Schema must be provided!")
        training_args = TrainingArguments(output_dir=predict_args.result_output_dir,
                                          per_device_eval_batch_size=predict_args.per_device_batch_size,
                                          report_to=predict_args.report_to)
        self.set_logger(training_args=training_args)
        schema = SchemaForBatch(schema=predict_args.schema)

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
        logger.info(f"Schema: {schema.as_schema_dict}")
        result = self.predict_batch(schema=schema,
                                    texts=self._predict_data[predict_args.text_column_name].fillna('').tolist(),
                                    tokenizer=self._tokenizer,
                                    model=self._model,
                                    split_text=predict_args.split_text,
                                    inference_batch_size=predict_args.per_device_batch_size,
                                    text_batch_size=predict_args.per_device_batch_size,
                                    text_max_length=predict_args.max_text_split_length,
                                    token_max_length=predict_args.max_seq_length,
                                    truncation=True
                                    )

        # 保存数据
        result_path = self.save_result_for_predict(training_args=training_args, predict_args=predict_args,
                                                   data=self._predict_data, result=result,
                                                   config=self._config, tokenizer=self._tokenizer, model=self._model
                                                   )
        return result_path
