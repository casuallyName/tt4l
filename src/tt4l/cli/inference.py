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
from .utils import echo_log_message
from typing_extensions import Annotated

inference_app = typer.Typer()


@inference_app.command(name='text_classification', short_help='Inference for text classification task')
def text_classification(
        model_path: Annotated[str, typer.Argument(metavar='model-path', help='model path')],
        trust_remote_code: Annotated[bool, typer.Option('-t', '--trust-remote-code',
                                                        help='Whether to trust remote code when load model')] = False,
        use_text_pair: Annotated[bool, typer.Option('-tp', '--use-text-pair',
                                                    help='Whether has text pair input')] = False,
        device: Annotated[str, typer.Option('-d', '--device',
                                            help="Predict on device")] = 'cpu',
        probable_score: Annotated[float, typer.Option('-p', '--position-prob',
                                                      help="Probable score when multi label classification")] = 0
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    echo_log_message('info', 'Loading models ...')
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
    from ..factory import TextClassificationFactory
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config,
                                                               trust_remote_code=trust_remote_code)
    model.to(torch.device(device)).eval()
    factory = TextClassificationFactory()
    while True:
        text_input = typer.prompt("Input text")
        if use_text_pair:
            pair_input = typer.prompt("Input pair")
            pair_input = [pair_input]
        else:
            pair_input = None

        typer.echo('Predicting ...')
        predictions = factory.inference(texts=[text_input], text_pairs=pair_input,
                                        model=model, tokenizer=tokenizer,
                                        probable_score=probable_score,
                                        max_length=config.max_position_embeddings,
                                        show_progress=True)
        typer.echo(typer.style(f'Prediction: {predictions[0]}', fg=typer.colors.BLUE))


@inference_app.command(name='token_classification', short_help='Inference for token classification task')
def token_classification(
        model_path: Annotated[str, typer.Argument(metavar='model-path', help='model path')],
        trust_remote_code: Annotated[bool, typer.Option('-t', '--trust-remote-code',
                                                        help='Whether to trust remote code when load model')] = False,
        device: Annotated[str, typer.Option('-d', '--device',
                                            help="Predict on device")] = 'cpu',
        return_word: Annotated[bool, typer.Option('-w', '--return-word',
                                                  help="Whether return pre word labels or entities json.")] = False
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    echo_log_message('info', 'Loading models ...')
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
    from ..parser import SequenceParser
    from ..factory import TokenClassificationFactory
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config,
                                                            trust_remote_code=trust_remote_code)
    model.to(torch.device(device)).eval()
    console = Console()
    parser = SequenceParser(sequence_type='BIO', id2label=config.id2label, is_decode_to_label=True)
    factory = TokenClassificationFactory()
    while True:
        text_input = typer.prompt("Input text")
        predictions = factory.inference(texts=[[i for i in text_input]],
                                        is_split_into_words=True,
                                        model=model, tokenizer=tokenizer,
                                        max_length=config.max_position_embeddings,
                                        drop_special_tokens_of_result=True,
                                        show_progress=True
                                        )
        if return_word:
            typer.echo(typer.style(f'', fg=typer.colors.BLUE))
            table = Table(title='Prediction: ', show_lines=False)
            table.add_column("Word", style=Style(color='blue'))
            table.add_column("Tag", style=Style(color='blue'))

            for word, tag in zip(text_input, predictions[0]):
                table.add_row(word, tag)
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
        model_path: Annotated[str, typer.Argument(metavar='model-path', help='model path')],
        device: Annotated[str, typer.Option('-d', '--device',
                                            help="Predict on device")] = 'cpu',
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    echo_log_message('info', 'Loading models ...')
    import torch
    from ..factory import UniversalInformationExtractionFactory
    from ..factory.universal_information_extraction.modules import Schema
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
