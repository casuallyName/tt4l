# -*- coding: utf-8 -*-
# @Time     : 2024/7/26 23:53
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import unittest

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(
        unittest.TestLoader().discover("./", "test_*.py")
    )
