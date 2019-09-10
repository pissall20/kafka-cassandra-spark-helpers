import logging
import os


class Logger(object):
    LOGGING_DIR = "logs/"

    def __init__(self, name):
        name = name.replace('.log', '')
        logger = logging.getLogger('logs.%s' % name)  # log_namespace can be replaced with your namespace
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            file_name = os.path.join(self.LOGGING_DIR, '%s.log' % name)
            handler = logging.FileHandler(file_name)
            formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
        self._logger = logger

    def get(self):
        return self._logger
