# @Time     : 2024/7/4 17:38
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
from enum import Enum


class PoolingStrategy(Enum):
    CLS_POOLING = 'cls'
    MEAN_POOLING = 'mean'
    MAX_POOLING = 'max'
    FL_AVG_POOLING = 'first_last_avg'
