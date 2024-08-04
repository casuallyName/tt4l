# Training a transformer model for language

## Install

```shell
pip install tt4l
```

## Usage

### Tasks

find supported tasks

```shell
tt4l tasks
```

| Args                             | Task Name                        | Short Description                                                                                                                                                                                                                                                                                             |
|----------------------------------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| text_classification              | Text Classification              | Text classification is a common NLP task that assigns a label or class to text.                                                                                                                                                                                                                               |
| token_classification             | Token Classification             | Token classification assigns a label to individual tokens in a sentence. One of the most common task is Named Entity Recognition (NER)。                                                                                                                                                                       |
| universal_information_extraction | Universal Information Extraction | The Unified Information Extraction Framework (UIE) for general information extraction achieves unified modeling of tasks such as entity extraction,   relationship extraction, event extraction, and sentiment analysis, and enables good transferability and generalization ability between different tasks. |

Show task description.  
`tt4l desc [TASK_TYPE]`
Eg:

```shell
tt4l desc text_classification
```

### init task config file

`tt4l init [TASK_TYPE]`

Eg:

```shell
tt4l init text_classification
```

**Modify the corresponding configuration**

### Train

`tt4l train [TASK_TYPE]`

Eg:

```shell
tt4l train text_classification_task.yaml
```

### Predict

`tt4l predict [YAML]`

Eg:

```shell
tt4l predict text_classification_task.yaml
```

### Inference

`tt4l inference [TASK_TYPE] [*args]`
