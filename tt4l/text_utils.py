# @Time     : 2024/8/2 14:51
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["ChineseSentenceSpliter", ]

import re
from typing import List, Tuple

from tqdm import tqdm


class ChineseSentenceSpliter:
    def __init__(self, max_length: int = 512, truncation: bool = False):
        """
        中文文本切分器

        Args:
            max_length (`int`, default to `512`): 句子长度
            truncation (`bool`, defaults to `True`): 是否截断句子
        """
        self._regex = re.compile("[?!。？！\r\n]+")
        self.max_length = max_length
        self.truncation = truncation

    def _split(self, text: str) -> Tuple[List[str], List[int]]:
        """

        Args:
            text (`str`): 待切分文本

        Returns:
            切分后的文本及其对应偏移量
        """
        offsets = [text.find(i) for i in self._regex.split(text) if i]
        return [text[offsets[i - 1]:offsets[i]] for i in range(1, len(offsets))] + [text[offsets[-1]:]], offsets

    def split(self, document: str) -> List[Tuple[str, int]]:
        """
        切分文本

        Args:
            document (`str`): 待切分文本

        Returns:
            切分后的文本及其对应偏移量

        """
        all_parts: List[Tuple[str, int]] = []
        texts, offsets = self._split(document)
        part_text = ''
        _offset = 0
        for i, (text, offset) in enumerate(zip(texts, offsets)):
            if len(part_text + text) >= self.max_length and part_text != '':
                if self.truncation:
                    part_text = part_text[:self.max_length]
                all_parts.append((part_text, _offset))
                _offset = offset
                part_text = text
            else:
                offset += len(text)
                part_text += text
        if part_text:
            all_parts.append((part_text, _offset))
        return all_parts

    def split_documents(self, documents: List[str], show_progress: bool = False) -> List[Tuple[int, str, int]]:
        """
        批量切分文本

        Args:
            documents (`List[str]`): 带切分文本列别
            show_progress (`bool`, defaults to `False): 是否显示进度条

        Returns:
            文本序号，切分后的文本及，其对应偏移量

        """
        split_texts: List[Tuple[int, str, int]] = []
        if show_progress:
            iterator = tqdm(documents)
        else:
            iterator = documents
        for document_id, document in enumerate(iterator):
            for text, offset in self.split(document):
                split_texts.append((document_id, text, offset))
        return split_texts
