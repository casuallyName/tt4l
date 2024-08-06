# -*- coding: utf-8 -*-
# @Time     : 2024/8/4 21:19
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import json
import os
import re

import typer
from rich.console import Console
from rich.table import Table, Style
from typing_extensions import Annotated

from tt4l.cli.utils import echo_log_message

inference_app = typer.Typer()


@inference_app.command(name='text_classification', short_help='Inference for text classification task')
def text_classification(
        model_path: Annotated[str, typer.Argument(metavar='model_path')],
        use_text_pair: Annotated[bool, typer.Option('-p', '--use_text_pair')] = False,
        device: Annotated[str, typer.Option('-d', '--device')] = 'cpu',
        mark_line: Annotated[float, typer.Option('-l', '--mark-line')] = .5,
        disable_sigmoid: Annotated[bool, typer.Option('-s', '--sigmoid')] = False
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    echo_log_message('info', 'Loading models ...')
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config,
                                                               trust_remote_code=True)
    model.to(torch.device(device)).eval()
    while True:
        text_input = typer.prompt("Input text")
        if use_text_pair:
            pair_input = typer.prompt("Input pair")
            model_inputs = tokenizer(text=str(text_input),
                                     text_pair=str(pair_input),
                                     return_tensors='pt',
                                     max_length=config.max_position_embeddings,
                                     truncation=True,
                                     )
        else:
            model_inputs = tokenizer(text=str(text_input), return_tensors='pt',
                                     max_length=config.max_position_embeddings,
                                     truncation=True, )
        typer.echo('Predicting ...')
        predictions = model(
            **{k: v.to(torch.device(device)) for k, v in model_inputs.items()}).logits.detach().cpu().numpy()
        if model.config.problem_type == 'multi_label_classification':
            if not disable_sigmoid:
                predictions = 1 / (1 + np.exp(-predictions))
            predictions = [
                ';;'.join([config.id2label[item[0]] for item in np.argwhere(prediction > mark_line)])
                for prediction in predictions]
        else:
            predictions = [config.id2label[item] for item in np.argmax(predictions, axis=1)]
        typer.echo(typer.style(f'Prediction: {predictions[0]}', fg=typer.colors.BLUE))


@inference_app.command(name='token_classification', short_help='Inference for token classification task')
def token_classification(
        model_path: Annotated[str, typer.Argument(metavar='model_path')],
        use_text_pair: Annotated[bool, typer.Option('-p', '--use_text_pair')] = False,
        device: Annotated[str, typer.Option('-d', '--device')] = 'cpu',
        return_word: Annotated[bool, typer.Option('-w', '--return_word')] = False
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    echo_log_message('info', 'Loading models ...')
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
    from tt4l.modeling_outputs import TokenCrfClassifierOutput
    from tt4l.parser.sequence_parser import SequenceParser
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config,
                                                            trust_remote_code=True)
    model.to(torch.device(device)).eval()
    console = Console()
    parser = SequenceParser(sequence_type='BIO', id2label=config.id2label)

    while True:
        text_input = typer.prompt("Input text")
        if use_text_pair:
            pair_input = typer.prompt("Input pair")
            model_inputs = tokenizer(text=str(text_input),
                                     text_pair=str(pair_input),
                                     return_tensors='pt',
                                     max_length=config.max_position_embeddings,
                                     truncation=True,
                                     )
        else:
            model_inputs = tokenizer(text=str(text_input), return_tensors='pt',
                                     max_length=config.max_position_embeddings,
                                     truncation=True)
        typer.echo('Predicting ...')
        predictions = model(
            **{k: v.to(torch.device(device)) for k, v in model_inputs.items()})

        if not isinstance(predictions, TokenCrfClassifierOutput):
            predictions = predictions.decode_label
        else:
            predictions = predictions.logits.detach().cpu().numpy()
            predictions = np.argmax(predictions, axis=2)
            predictions = [
                [p for (p, m) in zip(prediction, attention_mask) if m is not None]
                for prediction, attention_mask in zip(predictions, [model_inputs.word_ids(batch_index=0)])
            ]

        if return_word:
            typer.echo(typer.style(f'', fg=typer.colors.BLUE))
            table = Table(title='Prediction: ', show_lines=False)
            table.add_column("Word", style=Style(color='blue'))
            table.add_column("Tag", style=Style(color='blue'))

            for word, tag in zip(text_input, predictions[0]):
                table.add_row(word, config.id2label.get(tag))
            console.print(table)
        else:
            results = parser.parser(texts=[text_input],
                                    predictions=predictions,
                                    show_progress=False
                                    )
            typer.echo(typer.style(f'Prediction: ', fg=typer.colors.BLUE), nl=False)
            console.print(results)


@inference_app.command(name='universal_information_extraction', short_help='Inference for UIE task')
def universal_information_extraction(
        model_path: Annotated[str, typer.Argument(metavar='model_path')],
        device: Annotated[str, typer.Option('-d', '--device')] = 'cpu',
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    echo_log_message('info', 'Loading models ...')
    import torch
    from tt4l.factory import UniversalInformationExtractionFactory
    from tt4l.factory.universal_information_extraction.modules import Schema
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, config=config,
                                      trust_remote_code=True)
    model.to(torch.device(device)).eval()
    console = Console()
    factory = UniversalInformationExtractionFactory()
    building_schema = True
    schema = None
    while building_schema:
        typer.echo('You can input a json or string for building schema.')
        schema_input = typer.prompt("Input schema")
        try:
            schema_dict = json.loads(schema_input)
        except:
            try:
                schema_dict = eval(schema_input)
            except:
                if ''.join(re.findall("[\u4e00-\u9fa5A-Za-z0-9]+", schema_input)) == schema_input:
                    schema_dict = schema_input
                else:
                    echo_log_message("warning", "Can not parser this schema.")
                    continue
        try:
            schema = Schema(schema_dict)
            typer.echo(typer.style(f'Schema: ', fg=typer.colors.BLUE), nl=False)
            console.print(schema.as_schema_dict)
            building_schema = not typer.confirm("Do you want to use this schema?")
        except:
            echo_log_message("warning", "Can not parser this schema.")
    while True:
        text_input = typer.prompt("Input text")
        typer.echo('Predicting ...')
        result = factory.predict_single(schema=schema,
                                        text=text_input,
                                        tokenizer=tokenizer,
                                        model=model,
                                        truncation=True
                                        )
        typer.echo(typer.style(f'Prediction: ', fg=typer.colors.BLUE), nl=False)
        console.print(result)


if __name__ == '__main__':
    inference_app()
