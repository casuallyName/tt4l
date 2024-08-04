# @Time     : 2024/8/2 11:40
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["UniversalInformationExtractionFactory"]

from typing import List, Dict, Union, Any

import torch
from tqdm import trange
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from tt4l.factory.base import BaseTaskFactory
from tt4l.factory.universal_information_extraction.modules import (
    PositionInfo,
    Schema,
    Result,
    SchemaForBatch
)
from tt4l.text_utils import ChineseSentenceSpliter


class UniversalInformationExtractionFactory(BaseTaskFactory):
    tasks_name = "uie"
    description = ("Token classification assigns a label to individual tokens in a sentence. "
                   "One of the most common task is Named Entity Recognition (NER)。")
    description_zh = ("通用信息抽取统一框架UIE，"
                      "该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，"
                      "并使得不同任务间具备良好的迁移和泛化能力。")
    default_args_yaml_name = 'uie_task.yaml'

    @staticmethod
    def uie_inference(
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
    def predict_single(schema: Union[List, Dict, str], text: str,
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
        tree = Schema(schema)
        if split_text:
            spliter = ChineseSentenceSpliter(max_length=text_max_length, truncation=truncation)
            split_texts = spliter.split(text)
        else:
            split_texts = [(text, 0)]

        tree.reset_status(split_texts)
        for inputs in tree:
            inference_result = UniversalInformationExtractionFactory.uie_inference(
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
    def predict_batch(schema: Union[List, Dict, str], texts: List[str],
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
                batch_inference_results = UniversalInformationExtractionFactory.uie_inference(
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
