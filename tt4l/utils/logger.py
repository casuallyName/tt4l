# @Time     : 2024/3/4 14:33
# @File     : logger.py
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import logging

# import colorlog

def Logger(name):
    logger = logging.getLogger(name)
    _format = "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
    logger.setLevel("INFO")
    formatter = logging.Formatter(_format)
    log_colors_config = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }

    # color_formatter = colorlog.ColoredFormatter(f"%(log_color)s{_format}",
    #                                                  log_colors=log_colors_config)

    streamHandler = logging.StreamHandler()

    # streamHandler.setFormatter(color_formatter)
    logger.addHandler(streamHandler)
    return logger


class _Logger(object):

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self._format = "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
        self.logger.setLevel("INFO")
        self.formatter = logging.Formatter(self._format)
        self.log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
        # '%(log_color)s%(asctime)s  %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
        self.color_formatter = colorlog.ColoredFormatter(f"%(log_color)s{self._format}",
                                                         log_colors=self.log_colors_config)

        self.streamHandler = logging.StreamHandler()
        # self.streamHandler.setLevel("DEBUG")

        # file_name = time.strftime('%Y_%m_%d_%H', time.localtime()) + '.log'
        # if not os.path.exists(os.path.join(ROOT_PATH, 'logs')):
        #     os.makedirs(os.path.join(ROOT_PATH, 'logs'))
        # self.fileHandler = logging.FileHandler(os.path.join(ROOT_PATH, 'logs', file_name), 'a', encoding='utf-8')
        # self.fileHandler.setLevel("DEBUG")

        # self.streamHandler.setFormatter(self.formatter)
        self.streamHandler.setFormatter(self.color_formatter)
        # self.fileHandler.setFormatter(self.formatter)

        self.logger.addHandler(self.streamHandler)
        # self.logger.addHandler(self.fileHandler)

    def set_level(self, level):
        self.logger.setLevel(level)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)
