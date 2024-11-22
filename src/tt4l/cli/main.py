# @Time     : 2024/7/4 15:51
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import dataclasses
import os
import textwrap
from typing import Annotated

import typer
import yaml
from .inference import inference_app
from .utils import echo_error_info, echo_log_message, init_yaml
from ..factory.auto import AutoFactory, TaskFactoryNotFound, FACTORY_MAP

app = typer.Typer(name='tt4l',
                  help='Training a transformer model for language',
                  hidden=True,
                  add_completion=False,
                  pretty_exceptions_enable=False
                  )

app.add_typer(inference_app, name='inference', help='Inference using a trained model')

ALL_TASKS_STRING = ', '.join(list(FACTORY_MAP.keys()))


def find_similarity_task(task_name: str):
    import difflib
    sim_dict = {}
    for task in FACTORY_MAP.keys():
        sim_dict[task] = difflib.SequenceMatcher(None, task_name, task).ratio()
    return max(sim_dict.items(), key=lambda x: x[1])[0]


def task_name_checker(value: str):
    if value not in FACTORY_MAP:
        similarity_task_name = find_similarity_task(value)
        raise typer.BadParameter(f"You can find more information use 'tt4l tasks'.\n"
                                 f"Do you want to use task '{similarity_task_name}' ?"
                                 )
    return value


@app.command(name='tasks', help='Show supported tasks')
def tasks(zh: Annotated[bool, typer.Option('--zh', help="Show description use Chinese")] = False):
    from rich.console import Console
    from rich.table import Table
    from rich.style import Style
    from rich.progress import track
    console = Console()

    table = Table(title="已支持的任务" if zh else 'Supported Tasks Description', show_lines=True)
    table.add_column("参数" if zh else "Args", style=Style(color='blue'))
    table.add_column("任务名称" if zh else "Task Name", style=Style(color='magenta'))
    table.add_column("简介" if zh else "Short Description")
    for task_type in track(FACTORY_MAP.keys(), description="Finding task"):
        task = AutoFactory.from_task_type(task_type=task_type)
        table.add_row(
            task_type,
            task.tasks_name.replace('-', ' ').title(),
            '，\n'.join(task.description_zh.split('，')) if zh else task.description
        )
    console.print(table)


def arguments_formater(arg_cls):
    for field in dataclasses.fields(arg_cls):
        typer.echo('  ', nl=False)
        # 判断 是否为必选
        if field.default == dataclasses.MISSING:
            typer.echo(typer.style('* ', fg=typer.colors.BRIGHT_RED, bold=True), nl=False)
        else:
            typer.echo('  ', nl=False)
        typer.echo(typer.style(field.name, fg=typer.colors.BRIGHT_BLUE), nl=False)

        # 数据类型
        type_name = (', '.join([arg.__name__ if arg.__name__ != 'NoneType' else 'None' for arg in field.type.__args__])
                     if field.type.__name__ in ('Optional', 'Union') else field.type.__name__)
        typer.echo(typer.style(f' ({type_name})', fg=typer.colors.BRIGHT_CYAN), nl=False)

        # 默认值
        if field.default == dataclasses.MISSING:
            typer.echo()
        else:
            if isinstance(field.default, bool):
                value = typer.style(f'{field.default}',
                                    fg=typer.colors.BRIGHT_YELLOW, bold=True, bg=typer.colors.BLACK)
            elif field.default is None:
                value = typer.style(f'{field.default}',
                                    fg=typer.colors.BRIGHT_MAGENTA, bold=True, bg=typer.colors.BLACK)
            elif isinstance(field.default, int) or isinstance(field.default, float):
                value = typer.style(f'{field.default}',
                                    fg=typer.colors.BRIGHT_GREEN, bold=True, bg=typer.colors.BLACK)
            else:
                value = typer.style(f'{field.default}',
                                    fg=typer.colors.BRIGHT_WHITE, bold=True, bg=typer.colors.BLACK)
            typer.echo(f' = ' + value)

        # Help
        typer.secho(f'        ' + '\n        '.join(textwrap.wrap(field.metadata.get('help', ''))))


@app.command(name='desc', help='Show task description.')
def desc(
        task_type: Annotated[str, typer.Argument(metavar='TASK', callback=task_name_checker,
                                                 help=f"Task type mush one of [{ALL_TASKS_STRING}]")]
):
    from transformers import TrainingArguments
    try:
        task = AutoFactory.from_task_type(task_type=task_type)
        typer.secho('Task description:', bold=True)

        typer.echo('    ' + '\n    '.join(textwrap.wrap(task.description)))

        typer.echo('\n')
        typer.secho('Training Arguments:', bold=True)
        arguments_formater(TrainingArguments)
        typer.echo('\n')
        typer.secho('Task Arguments:', bold=True)
        arguments_formater(task.task_args_cls)
        typer.echo('\n')
        typer.echo('Predict Arguments:')
        arguments_formater(task.predict_args_cls)
    except TaskFactoryNotFound as e:
        echo_error_info(error_type='TaskTypeError', error_message=str(e))


@app.command(name='init', help='Create a default training args config file.')
def init(task_type: Annotated[str, typer.Argument(metavar='task type',
                                                  callback=task_name_checker,
                                                  help=f"Use 'tasks' command to show all available tasks. "
                                                  )],
         replace: Annotated[bool, typer.Option('-r', '--replace',
                                               help="Overwrite file if exists")] = False,
         ):
    try:
        task = AutoFactory.from_task_type(task_type=task_type)
        init_yaml(
            task_type=task_type,
            task_args_cls=task.task_args_cls,
            predict_args_cls=task.predict_args_cls,
            yaml_name=task.default_args_yaml_name,
            replace=replace,
        )
    except TaskFactoryNotFound as e:
        echo_error_info(error_type='TaskTypeError', error_message=str(e))


@app.command(name='train', short_help='Train model')
def train(yaml_path: Annotated[str, typer.Argument(metavar='Path|Text', help='Training arguments yaml file path')]):
    from transformers import TrainingArguments
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"File {yaml_path} does not exist!")
    yaml_args = yaml.safe_load(open(yaml_path, encoding='utf-8'))
    task_type = yaml_args.get('TaskType', None)
    try:
        factory = AutoFactory.from_task_type(task_type)
        factory.train(task_args=factory.task_args_cls(**yaml_args.get('TaskArguments', {})),
                      training_args=TrainingArguments(**yaml_args.get('TrainingArguments', {})))
    except TaskFactoryNotFound as e:
        echo_error_info(error_type='TaskTypeError', error_message=str(e))


@app.command(name='predict', short_help='Predict using a trained model')
def predict(yaml_path: Annotated[str, typer.Argument(metavar='Path|Text', help='Training arguments yaml file path')]):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"File {yaml_path} does not exist!")
    yaml_args = yaml.safe_load(open(yaml_path, encoding='utf-8'))
    task_type = yaml_args.get('TaskType', None)
    try:
        factory = AutoFactory.from_task_type(task_type)
        result_output_path = factory.predict(
            predict_args=factory.predict_args_cls(**yaml_args.get('PredictArguments', {}))
        )
        echo_log_message(log_level='info',
                         log_message=f"Write result file to '{result_output_path}'.")
    except TaskFactoryNotFound as e:
        echo_error_info(error_type='TaskTypeError', error_message=str(e))


if __name__ == '__main__':
    app()
