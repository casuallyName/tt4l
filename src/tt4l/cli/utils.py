# @Time     : 2024/7/12 13:40
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import datetime
import inspect
import os
from dataclasses import asdict
from typing import Literal

import typer
import yaml

TRAIN_YAML_TEMPLATE = """# @Time     : {file_create_time}
# @File     : {file_name}
# @About    :

# Task type -> 用于区分任务类型 (请勿随意修改)
{task_type_info}

# 1. 训练参数
# TrainingArguments -> Trainer 初始化参数, 更多参数信息可参考 Transformers 官方文档
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
{training_arguments_info}

# TaskArguments -> 任务参数，该类别下参数用于调整 train 阶段
{task_arguments_info}

# 2. 预测参数
# PredictArguments -> 预测参数，该类别下参数用于调整 predict 阶段 (非训练阶段的 predict)
{predict_arguments_info}

"""


def echo_log_message(log_level: Literal['info', 'warning'], log_message: str) -> None:
    if log_level == 'warning':
        message = typer.style(f'WARNING : {log_message}', fg=typer.colors.YELLOW, bold=True)
    else:
        message = typer.style(f'INFO    : {log_message}', fg=typer.colors.BLUE, bold=True)
    typer.echo(message)


def echo_error_info(error_type: str, error_message: str, exit_cli: bool = True):
    typer.echo(typer.style(f'{error_type}: ', fg=typer.colors.RED, bold=True) + error_message, err=True)
    if exit_cli:
        exit()


def _make_default_value(data_object):
    return {param.name: f"[YOU MUST SET {param.name.upper()}]"
            for param in inspect.signature(data_object).parameters.values()
            if param.default == inspect.Parameter.empty}


def init_yaml(task_type, task_args_cls, predict_args_cls, yaml_name, replace=False):
    from tt4l.cli.default_arguments import DefaultTrainingArguments
    yaml_path = os.path.join(os.getcwd(), yaml_name)
    if os.path.exists(yaml_path) and replace is False:
        echo_log_message(log_level='warning',
                         log_message=f"Yaml file '{yaml_path}' already exists, you can ues '-r' overwrite.",
                         )
    else:
        typer.echo(
            typer.style(f"Write {task_type} task arguments file to '{yaml_path}'.",
                        fg=typer.colors.BLUE,
                        bold=True))
        training_args = DefaultTrainingArguments(**_make_default_value(DefaultTrainingArguments))
        task_args = task_args_cls(task_name=task_type, **_make_default_value(task_args_cls))
        predict_args = predict_args_cls(**_make_default_value(predict_args_cls))

        task_type_info = yaml.safe_dump({'TaskType': task_type}, sort_keys=False)
        training_arguments_info = yaml.safe_dump({'TrainingArguments': asdict(training_args)}, sort_keys=False)
        task_arguments_info = yaml.safe_dump({'TaskArguments': asdict(task_args)}, sort_keys=False)
        predict_arguments_info = yaml.safe_dump({'PredictArguments': asdict(predict_args)}, sort_keys=False)

        with open(yaml_name, 'w', encoding='utf-8') as f:
            f.write(
                TRAIN_YAML_TEMPLATE.format(
                    file_create_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    file_name=yaml_name,
                    task_type_info=task_type_info.replace(': null', ':'),
                    training_arguments_info=training_arguments_info.replace(': null', ':'),
                    task_arguments_info=task_arguments_info.replace(': null', ':'),
                    predict_arguments_info=predict_arguments_info,
                ).replace(': null', ':')

            )

            typer.echo(typer.style("More training arguments you can look at: "
                                   "https://huggingface.co/docs/transformers/main/en/main_classes/trainer"
                                   "#transformers.TrainingArguments",
                                   fg=typer.colors.BLUE,
                                   bold=True))
