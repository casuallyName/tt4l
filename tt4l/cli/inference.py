# -*- coding: utf-8 -*-
# @Time     : 2024/8/4 21:19
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import os
from typing import Dict

import typer
from typing_extensions import Annotated

inference_app = typer.Typer()


@inference_app.command(name='text_classification', short_help='Inference for text classification task')
def text_classification(
        model_path: Annotated[str, typer.Argument(metavar='model_path')],
        text_pair: Annotated[bool, typer.Option('-p', '--text_pair')] = False,
        device: Annotated[str, typer.Option('-d', '--device')] = 'cpu',
        mark_line: Annotated[float, typer.Option('-l', '--mark-line')] = .5,
        disable_sigmoid: Annotated[bool, typer.Option('-s', '--sigmoid')] = False
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File {model_path} does not exist!")
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, trust_remote_code=True).eval()
    while True:
        text_input = typer.prompt("Input text")
        if text_pair:
            pair_input = typer.prompt("Input pair")
            model_inputs = tokenizer(text=str(text_input), text_pair=str(pair_input), return_tensors='pt')
        else:
            model_inputs = tokenizer(text=str(text_input), return_tensors='pt')
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
