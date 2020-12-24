import GPUtil
from threading import Thread
from tqdm.auto import tqdm
import time
import sys

class GPUMonitor(Thread):
    def __init__(self, delay=10):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.setup_gbar()
        if self.gbars:
            self.start()

    def run(self):
        while not self.stopped:
            self.update_gpus()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
    
    def update_gpus(self):
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            mem_string = f'{gpu.memoryUsed/1000:.2f}/{gpu.memoryTotal/1000:.2f} GB'
            self.gbars[gpu.id].n = gpu.memoryUtil * 100
            self.gbars[gpu.id].set_description(mem_string, refresh=True)

    def setup_gbar(self):
        gpus = GPUtil.getGPUs()
        self.gbars = {}
        self.total_gpus = 0
        if gpus:
            for gpu in gpus:
                _gpubarformat = f'GPU [{gpu.id}] {gpu.name}: ' + '{desc} {bar} {percentage:.02f}% Utilization'
                self.gbars[gpu.id] = tqdm(range(100), colour='blue', bar_format=_gpubarformat, position=self.total_gpus, dynamic_ncols=True, leave=True, file=sys.stdout)
                self.total_gpus += 1


class DictArgs(dict):
    def __init__(self, config):
        for k,v in config.items():
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)