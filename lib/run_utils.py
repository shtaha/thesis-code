import os
import sys


def create_logger(logger_name, save_dir):
    sys.stdout = Logger(logger_name, save_dir=save_dir)


class Logger(object):
    def __init__(self, logger_name, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, f"{logger_name}-log.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
