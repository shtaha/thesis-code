import sys
import os


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "log.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
