# @Time     : 2024/8/2 11:42
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["PositionInfo", "InferenceInput", "BatchInferenceInput", "Schema", "SchemaForBatch", "Result"]

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Iterator, Any, Union, Set, Final, Literal

from pydantic import BaseModel


@dataclass
class PositionInfo:
    text: str
    score: float
    start: Optional[int] = None
    end: Optional[int] = None
    token_type: Literal[0, 1] = 0


@dataclass
class InferenceInput:
    prompts: List[str]
    texts: List[str]

    def __post_init__(self):
        assert len(self.prompts) == len(self.texts)

    def __len__(self):
        return len(self.prompts)


@dataclass
class BatchInferenceInput(InferenceInput):
    text_mapping: Dict[int, int]

    def __post_init__(self):
        assert len(self.prompts) == len(self.texts) == len(self.text_mapping)


class SchemaStepError(Exception):
    ...


class Result(BaseModel):
    text: str
    probability: float

    relations: Dict[str, List[Union['Result', Dict]]] = {}
    start: Optional[int] = None
    end: Optional[int] = None

    @property
    def as_dict(self):
        result = {
            'text': self.text,
            'probability': self.probability,

        }
        if self.start is not None:
            result['start'] = self.start
        if self.end is not None:
            result['end'] = self.end
        if len(self.relations):
            result['relations'] = self.relations

        return result


class Node(BaseModel):
    name: str
    father: Optional[Union['Node', 'Schema', 'SchemaForBatch']] = None
    results: List[Result] = []
    _children: List['Node'] = []
    _is_end: bool = True

    def __repr_str__(self, join_str: str) -> str:
        reprs = [
            f"name={repr(self.name)}",
            f"father={repr(self.father.name)}",
            f"is_end={repr(self.is_end)}"
        ]
        if len(self._children):
            reprs.append("children=[...]")
        return join_str.join(reprs)

    @property
    def as_schema_dict(self):
        if self.is_end:
            return self.name
        else:
            return {
                self.name: [c.as_schema_dict for c in self._children]
            }

    @property
    def is_end(self):
        return self._is_end

    @property
    def children(self) -> List['Node']:
        return self._children

    def add_child(self, child: 'Node'):
        self._is_end = False
        self._children.append(child)

    def format_result(self, result: Dict[str, List[Result]], prefix: str = None) -> List[Dict[str, Any]]:
        prefix = '' if prefix is None else f'{prefix}的'
        format_results: List[Result] = []
        if r := result.get(f"{prefix}{self.name}"):
            format_results.extend(r)
        if not self._is_end:
            for format_result in format_results:
                for child in self.children:
                    format_result.relations[child.name] = child.format_result(result, format_result.text)
        return [format_result.as_dict for format_result in format_results]

    @classmethod
    def build(cls, name: str, father: Union['Node', 'Schema', 'SchemaForBatch'] = None,
              children: Union[List, Dict, str] = None) -> 'Node':
        root = cls(name=name, father=father)
        if isinstance(children, dict):
            for k, v in children.items():
                root.add_child(cls.build(name=k, father=root, children=v))
        elif isinstance(children, list):
            for child in children:
                if isinstance(child, str):
                    root.add_child(cls.build(name=child, father=root))
                elif isinstance(child, Dict):
                    for k, v in child.items():
                        root.add_child(cls.build(name=k, father=root, children=v))
        elif isinstance(children, str):
            root.add_child(cls.build(name=children, father=root))
        return root


class Schema(BaseModel):
    name: Final[str] = 'ROOT'
    _children: List[Node] = []
    _is_ready: bool = False

    _step_done: bool = True
    _result: Dict[str, List[Result]] = {}

    _step_texts: List[str] = []  # step of text
    _step_prompts: List[str] = []  # step of prompts
    _step_offset_map: List[int] = []  # step of offset_map
    _step_node: List[Node] = []  # step of node

    def __init__(self, schema: Union[List, Dict, str]):
        super(Schema, self).__init__()
        self.reset_schema(schema)

    def __repr_str__(self, join_str: str) -> str:
        reprs = [
            f"is_ready={repr(self._is_ready)}"
        ]
        if len(self._children):
            reprs.append("children=[...]")

        return join_str.join(reprs)

    @property
    def as_schema_dict(self) -> List:
        return [child.as_schema_dict for child in self._children]

    @property
    def results(self) -> Dict[str, List[Result]]:
        return self._result

    @property
    def children(self) -> List[Node]:
        return self._children

    def reset_schema(self, schema: Union[List, Dict, str]):
        self._children = []
        self._is_ready = False
        self._step_done = True
        self._result = {}
        self._step_texts = []
        self._step_prompts = []
        self._step_offset_map = []
        self._step_node = []

        if isinstance(schema, dict):
            for k, v in schema.items():
                self.add_child(Node.build(name=k, children=v, father=self))
        elif isinstance(schema, list):
            for child in schema:
                if isinstance(child, str):
                    self.add_child(Node.build(name=child, father=self))
                elif isinstance(child, dict):
                    for k, v in child.items():
                        self.add_child(Node.build(name=k, children=v, father=self))
        elif isinstance(schema, str):
            self.add_child(Node.build(name=schema, father=self))

    def add_child(self, child: Node) -> None:
        self._children.append(child)

    def reset_status(self, texts: List[Tuple[str, int]]) -> None:
        self._result = {}
        self._step_texts = []  # step of text
        self._step_prompts = []  # step of prompts
        self._step_offset_map = []  # step of offset_map
        self._step_node = []  # step of node
        for node in self._children:
            for text, offset in texts:
                prompt = node.name
                self._step_node.append(node)
                self._step_offset_map.append(offset)
                self._step_texts.append(text)
                self._step_prompts.append(prompt)

        # 修改状态
        self._is_ready = True
        self._step_done = False

    def __iter__(self) -> Iterator[InferenceInput]:
        while self._is_ready:
            self._is_ready = False
            # 只有执行 step 才会继续yield
            yield InferenceInput(
                prompts=self._step_prompts,
                texts=self._step_texts,
            )
        if not self._step_done:
            raise SchemaStepError("Last step is not done, you must call `step()` before next step.")

    def __next__(self) -> InferenceInput:
        if self._is_ready:
            self._is_ready = False
            # 只有执行 step 才会继续yield
            return InferenceInput(
                prompts=self._step_prompts,
                texts=self._step_texts,
            )
        else:
            if not self._step_done:
                raise SchemaStepError("Last step is not done, you must call `step()` before next step.")
            else:
                raise StopIteration

    def step(self, step_result):
        next_texts = []
        next_prompts = []
        next_offset_map = []
        next_nodes = []

        for prompt in set(self._step_prompts):
            prompt_result = step_result.get(prompt)
            if prompt_result is not None:
                self._result[prompt] = prompt_result

        for node, prompt, text, offset in zip(self._step_node, self._step_prompts, self._step_texts,
                                              self._step_offset_map):
            if not node.is_end:
                for child_node in node.children:
                    for node_result in self._result.get(prompt, []):
                        next_nodes.append(child_node)
                        next_prompts.append(f"{node_result.text}的{child_node.name}")
                        next_texts.append(text)
                        next_offset_map.append(offset)

        self._step_texts = next_texts
        self._step_prompts = next_prompts
        self._step_offset_map = next_offset_map
        self._step_node = next_nodes

        # 修改状态
        self._is_ready = len(self._step_node) > 0
        self._step_done = True

    def result_dump(self) -> Dict[str, Any]:
        result = {}
        for node in self._children:
            result[node.name] = node.format_result(self._result)
        return result


class SchemaForBatch(Schema):
    _result: Dict[int, Dict[str, List[Result]]] = {}

    _step_text_index_mapping: Dict[int, int] = {}  # step_of_text[id] -> texts[id]
    _step_prompts_mapping: Dict[int, Set[str]] = {}  # texts[id] -> set of prompts
    _step_text_id_of_step_text: List[int] = []  # text_id of step_text, use list idx find text in texts

    @property
    def results(self) -> Dict[int, Dict[str, List[Result]]]:
        return self._result

    def reset_schema(self, schema: Union[List, Dict, str]):
        self._step_text_index_mapping = {}
        self._step_prompts_mapping = {}
        self._step_text_id_of_step_text = []
        super(SchemaForBatch, self).reset_schema(schema)

    def reset_status(self, texts: List[Tuple[int, str, int]]):
        self._result = {i: {} for i, _, _ in texts}
        self._step_texts = []  # step of text
        self._step_prompts = []  # step of prompts
        self._step_offset_map = []  # step of offset_map
        self._step_node = []  # step of node
        self._step_text_index_mapping = {}  # step_of_text[id] -> texts[id]
        self._step_prompts_mapping = {}  # texts[id] -> set of prompts
        self._step_text_id_of_step_text = []  # text_id of step_text, use list idx find text in texts
        _idx = 0
        for node in self._children:
            for idx, text, offset in texts:
                prompt = node.name
                self._step_text_id_of_step_text.append(idx)
                self._step_text_index_mapping[_idx] = idx
                self._step_node.append(node)
                self._step_offset_map.append(offset)
                self._step_texts.append(text)
                self._step_prompts.append(prompt)
                self._step_prompts_mapping[idx] = self._step_prompts_mapping.get(idx, set())
                self._step_prompts_mapping[idx].add(prompt)
                _idx += 1

        # 修改状态
        self._is_ready = True
        self._step_done = False

    def __iter__(self) -> Iterator[BatchInferenceInput]:
        while self._is_ready:
            self._is_ready = False
            # 只有执行 step 才会继续yield
            yield BatchInferenceInput(
                prompts=self._step_prompts,
                texts=self._step_texts,
                text_mapping=self._step_text_index_mapping
            )
        if not self._step_done:
            raise SchemaStepError("Last step is not done, you must call `step()` before next step.")

    def __next__(self) -> BatchInferenceInput:
        if self._is_ready:
            self._is_ready = False
            # 只有执行 step 才会继续yield
            yield BatchInferenceInput(
                prompts=self._step_prompts,
                texts=self._step_texts,
                text_mapping=self._step_text_index_mapping
            )
        else:
            raise StopIteration

    def step(self, step_result: Dict[int, Dict[str, List[Result]]]):
        next_texts = []
        next_prompts = []
        next_offset_map = []
        next_nodes = []

        next_text_index_mapping = {}
        next_prompts_mapping = {}
        next_text_id_of_step_text = []

        # 保存結果
        for idx, prompts in self._step_prompts_mapping.items():
            text_result = step_result.get(idx)
            if text_result is not None:
                for prompt in set(prompts):
                    prompt_result = text_result.get(prompt)
                    if prompt_result is not None:
                        self._result[idx][prompt] = self._result[idx].get(prompt, [])
                        self._result[idx][prompt].extend(prompt_result)

        # 查找子节点，生成下一step的数据inxi
        _idx = 0
        for node, prompt, text, offset, idx in zip(self._step_node, self._step_prompts, self._step_texts,
                                                   self._step_offset_map, self._step_text_id_of_step_text):
            if not node.is_end:
                for child_node in node.children:
                    for node_result in self._result[self._step_text_index_mapping[idx]].get(prompt, []):
                        next_nodes.append(child_node)
                        next_prompts.append(f"{node_result.text}的{child_node.name}")
                        next_prompts_mapping[idx] = next_prompts_mapping.get(idx, set())
                        next_prompts_mapping[idx].add(f"{node_result.text}的{child_node.name}")
                        next_texts.append(text)
                        next_offset_map.append(offset)
                        next_text_index_mapping[_idx] = idx
                        next_text_id_of_step_text.append(idx)

                        _idx += 1

        self._step_texts = next_texts
        self._step_prompts = next_prompts
        self._step_offset_map = next_offset_map
        self._step_node = next_nodes

        self._step_text_index_mapping = next_text_index_mapping
        self._step_prompts_mapping = next_prompts_mapping
        self._step_text_id_of_step_text = next_text_id_of_step_text

        # 修改状态
        self._is_ready = len(self._step_node) > 0
        self._step_done = True

    def result_dump(self) -> List[Dict[str, Any]]:
        results = [{} for _ in self._result]
        for idx, result in self._result.items():
            for node in self._children:
                if res := node.format_result(result):
                    results[idx][node.name] = res
        return results
