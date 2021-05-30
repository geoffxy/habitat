from habitat.analysis.kernels import MeasuredKernel
from habitat.utils import ns_to_ms


class RunTime:
    @property
    def run_time_ms(self):
        raise NotImplementedError

    @property
    def ktime_ns(self):
        return sum(map(lambda k: k.run_time_ns, self.kernels))

    @property
    def kernels(self):
        return []

    @property
    def device(self):
        raise NotImplementedError


class RunTimeMeasurement(RunTime):
    def __init__(self, run_time_ms, kernels, device):
        self._run_time_ms = run_time_ms
        self._kernels = kernels
        self._device = device

    @property
    def run_time_ms(self):
        return self._run_time_ms

    @property
    def kernels(self):
        return self._kernels

    @property
    def device(self):
        return self._device


class RunTimePrediction(RunTime):
    def __init__(self, overhead_ns, predicted_kernels, device):
        self._run_time_ms = None
        self._overhead_ns = overhead_ns
        self._predicted_kernels = predicted_kernels
        self._device = device

    @property
    def run_time_ms(self):
        if self._run_time_ms is not None:
            return self._run_time_ms
        run_time_ns = self._overhead_ns + sum(map(
            lambda k: k.run_time_ns,
            self.kernels,
        ))
        self._run_time_ms = ns_to_ms(run_time_ns)
        return self._run_time_ms

    @property
    def kernels(self):
        return self._predicted_kernels

    @property
    def device(self):
        return self._device


class RunTimePurePrediction(RunTime):
    def __init__(self, run_time_ms, device):
        self._run_time_ms = run_time_ms
        self._device = device

    @property
    def run_time_ms(self):
        if self._run_time_ms is not None:
            return self._run_time_ms
        run_time_ns = self._overhead_ns + sum(map(
            lambda k: k.run_time_ns,
            self.kernels,
        ))
        self._run_time_ms = ns_to_ms(run_time_ns)
        return self._run_time_ms

    @property
    def device(self):
        return self._device
