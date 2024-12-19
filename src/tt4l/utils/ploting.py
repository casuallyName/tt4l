# @Time     : 2024/11/22 12:15
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import json
import math
import os
from typing import Union, List, Literal, Dict

from pydantic import BaseModel, PrivateAttr, ConfigDict
from transformers.trainer import TRAINER_STATE_NAME

from .import_utils import is_matplotlib_available

if is_matplotlib_available():
    import matplotlib.pyplot as plt

STEP_KEY_NAME = "step"
EPOCH_KEY_NAME = "epoch"
IGNORE_LOG_KEYS = ['epoch', 'step']
TRAIN_LOG_FLAT_KEY = 'grad_norm'
EVAL_LOG_FLAT_KEY = 'eval_runtime'
PASS_LOG_FLAT_KEY = 'train_runtime'


class LogHistory(BaseModel):
    key: str
    _epochs: List[int] = PrivateAttr(default_factory=list)
    _steps: List[int] = PrivateAttr(default_factory=list)
    _values: List[Union[int, float]] = PrivateAttr(default_factory=list)

    @property
    def steps(self):
        return self._steps

    @property
    def values(self):
        return self._values

    @property
    def smooth_values(self) -> List[float]:
        r"""
        EMA implementation according to TensorBoard.
        """
        if len(self._values) == 0:
            return []

        last = self._values[0]
        smoothed = []
        weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(self._values))) - 0.5)  # a sigmoid function
        for next_val in self._values:
            smoothed_val = last * weight + (1 - weight) * next_val
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def save_plot(self, save_dictionary: str, prefix_name: str):
        plt.figure()
        if len(self.steps) > 1:
            plt.plot(self.steps, self.values, color="#1f77b4", alpha=0.4, label="original")
            plt.plot(self.steps, self.smooth_values, color="#1f77b4", label="smoothed")
        else:
            plt.bar(self.steps, self.values, color="#1f77b4", label="original")
        plt.title(f"{STEP_KEY_NAME} - {self.key}")
        plt.xlabel(STEP_KEY_NAME)
        plt.ylabel(self.key)
        plt.legend()
        name = self.key.replace("/", "_")
        prefix_name = (prefix_name + '_') if prefix_name else ''
        figure_path = os.path.join(save_dictionary, f"{prefix_name}{name}.png")
        plt.savefig(figure_path, format="png", dpi=100)

    def add(self, epoch: int, step: int, value: Union[int, float]):
        self._epochs.append(epoch)
        self._steps.append(step)
        self._values.append(value)


class Logs(BaseModel):
    model_config = ConfigDict(extra='allow')

    _logs: Dict[str, LogHistory] = PrivateAttr(default_factory=dict)

    @property
    def keys(self):
        return list(self._logs.keys())

    def add(self, key: str, epoch: int, step: int, value: Union[int, float]):
        if key not in self._logs.keys():
            self._logs[key] = LogHistory(key=key)
            setattr(self, key, self._logs[key])
        self._logs[key].add(epoch, step, value)

    def save_plots(self, save_dictionary: str, prefix_name: str):
        for key in self._logs.keys():
            self._logs[key].save_plot(save_dictionary=save_dictionary, prefix_name=prefix_name)


class TrainLogs(Logs):
    @property
    def name(self):
        return 'training'


class EvalLogs(Logs):
    @property
    def name(self):
        return 'eval'


class LogsHistory(BaseModel):
    train: Logs = TrainLogs()
    eval: Logs = EvalLogs()

    def add(self, part: Literal['train', 'eval'], key: str, step: int, value: Union[int, float]):
        getattr(self, part).add(key, step, value)

    def save_plots(self, save_dictionary: str, prefix_name: str):
        self.train.save_plots(save_dictionary=save_dictionary, prefix_name=prefix_name)
        self.eval.save_plots(save_dictionary=save_dictionary, prefix_name=prefix_name)

    @classmethod
    def from_trainer_state(cls, save_dictionary: str):
        logs = cls()
        with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
            data = json.load(f)

        for log in data['log_history']:
            if 'grad_norm' in log:
                for key in log.keys():
                    if key not in IGNORE_LOG_KEYS:
                        logs.train.add(key, log[EPOCH_KEY_NAME], log[STEP_KEY_NAME], log[key])
            elif 'eval_runtime' in log:
                for key in log.keys():
                    if key not in IGNORE_LOG_KEYS:
                        logs.eval.add(key, log[EPOCH_KEY_NAME], log[STEP_KEY_NAME], log[key])
            elif PASS_LOG_FLAT_KEY in log:
                pass
            # else:
            #     print(log)
        return logs


def save_log_history_plots(save_dictionary):
    if is_matplotlib_available():
        logs = LogsHistory.from_trainer_state(save_dictionary)
        logs.save_plots(save_dictionary=save_dictionary, prefix_name='training')
