import sys


class Tee:
    """ Duplicate output to both stdout/err and file """
    def __init__(self, file, err=False):
        self.file = open(file, 'w')
        self.err = err
        if not err:
            self.std = sys.stdout
            sys.stdout = self
        else:
            self.std = sys.stderr
            sys.stderr = self

    def __del__(self):
        if not self.err:
            sys.stdout = self.std
        else:
            sys.stderr = self.std
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.std.write(data)

    def flush(self):
        self.file.flush()