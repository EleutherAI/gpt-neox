# coding=utf-8
# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        try:
            self.file.write(data)
        except OSError:
            pass
        try:
            self.std.write(data)
        except OSError:
            pass

    def flush(self):
        try:
            self.file.flush()
        except OSError:
            pass
