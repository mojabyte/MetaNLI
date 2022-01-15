import sys


class Logger(object):
    def __init__(self, file_path: str = "logs.txt"):
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()
