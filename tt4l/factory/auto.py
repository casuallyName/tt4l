# @Time     : 2024/7/29 18:37
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
__all__ = ["AutoFactory", "FACTORY_MAP", "TaskFactoryNotFound"]

import importlib

from tt4l.factory.base import BaseTaskFactory

FACTORY_MAP = {
    'text_classification': 'TextClassificationFactory',
    'token_classification': 'TokenClassificationFactory',
}


class TaskFactoryNotFound(Exception):
    pass


class AutoFactory(BaseTaskFactory):

    @staticmethod
    def factory_loader(task_type):
        if task_type in FACTORY_MAP:
            pkg = importlib.import_module(f'tt4l.factory.{task_type}')
            return getattr(pkg, FACTORY_MAP[task_type])
        else:
            raise TaskFactoryNotFound(f'Task type {task_type} is not supported')

    @classmethod
    def from_task_type(cls, task_type) -> BaseTaskFactory:
        factory_cls = cls.factory_loader(task_type)
        return factory_cls()
