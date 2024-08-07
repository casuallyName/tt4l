# Training a transformer model for language

## Install

#### From Github

```shell
pip install git+https://github.com/casuallyName/tt4l.git
```


## Usage

### 查看支持的任务类型

查看所有支持任务

```shell
tt4l tasks
```

查看某一任务具体信息
`tt4l desc [TASK_TYPE]`
Eg:

```shell
tt4l desc text_classification
```

### 为训练任务初始化配置文件

`tt4l init [TASK_TYPE]`

Eg:

```shell
tt4l init text_classification
```

**修改对应配置**

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
