import torch
from megatron import device_backend


class Metric:
    """
    Dumb utility to collect and report average wall-time metrics.
    """

    def __init__(self, label):
        self.label = label
        self.measurements = []

    def collect(self, measurement):
        self.measurements.append(measurement)

    def get_measurements(self):
        return self.measurements[:]

    def report(self):
        print(
            self.label,
            torch.quantile(torch.tensor(self.measurements), torch.arange(10) / 10.0),
        )


def monitor_method_cuda_wall_times(metric, obj, methodname):
    """
    Measure timings for a method on an object or class.

    For instance:

    >>> metric = Metric('!LNORM')
    >>> monitor_method_wall_times(metric, LayerNorm, 'forward')
    """
    oldmeth = getattr(obj, methodname)

    start_event = device_backend.Event(enable_timing=True)
    end_event = device_backend.Event(enable_timing=True)

    def newmeth(*args, **kw):
        start_event.record()
        try:
            return oldmeth(*args, **kw)
        finally:
            end_event.record()
            device_backend.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            metric.collect(elapsed)
            metric.report()

    setattr(obj, methodname, newmeth)
