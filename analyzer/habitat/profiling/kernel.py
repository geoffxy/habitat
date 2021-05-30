import logging
import habitat.habitat_cuda as hc

from habitat.analysis import SPECIAL_OPERATIONS
from habitat.analysis.metrics import resolve_metrics
from habitat.analysis.kernels import MeasuredKernel

logger = logging.getLogger(__name__)


class KernelProfiler:
    def __init__(self, device, metrics=None, metrics_threshold_ms=0):
        self._device = device
        self._metrics = resolve_metrics(metrics, self._device)
        self._metrics_threshold_ns = metrics_threshold_ms * 1000000

    def measure_kernels(self, runnable, func_name=None):
        """
        Uses CUPTI to measure the kernels launched by runnable.

        Returns:
          A list of MeasuredKernels
        """
        if func_name is None:
            fname = (
                runnable.__name__ if hasattr(runnable, "__name__")
                else "Unnamed"
            )
        else:
            fname = func_name

        return list(map(
            lambda ks: MeasuredKernel(ks[0], ks[1], self._device),
            self._measure_kernels_raw(runnable, fname)
        ))

    def _measure_kernels_raw(self, runnable, func_name):
        """
        Uses CUPTI to measure the kernels launched by runnable.

        Returns:
          A list of tuples, where
            - tuple[0] is the raw kernel measurement that should be used for
              the kernel's run time
            - tuple[1] is a list of the raw kernel measurements that contain
              the metrics requested
        """
        time_kernels = hc.profile(runnable)
        if (len(self._metrics) == 0 or
                func_name in SKIP_METRICS or
                func_name in SPECIAL_OPERATIONS or
                self._under_threshold(time_kernels)):
            return list(map(lambda tk: (tk, []), time_kernels))

        try:
            metric_kernels = [
                hc.profile(runnable, metric) for metric in self._metrics
            ]
            # Make sure the same number of kernels are recorded for each metric
            assert all(map(
                lambda ks: len(ks) == len(metric_kernels[0]),
                metric_kernels,
            ))
            # metric_kernels is originally (# metrics x # kernels in op)
            # we need to transpose it to become (# kernels in op x # metrics)
            # so that we can join kernels with their metrics.
            transposed = map(list, zip(*metric_kernels))
            # We return a list of (time kernel, [metric kernels])
            return list(zip(time_kernels, transposed))
        except RuntimeError as ex:
            logger.warn(
                'Metrics error "%s" for function "%s".',
                str(ex),
                func_name,
            )
            return list(map(lambda tk: (tk, []), time_kernels))

    def _under_threshold(self, kernels):
        # If under threshold, don't measure metrics
        return (
            sum(map(lambda k: k.run_time_ns, kernels))
            <= self._metrics_threshold_ns
        )


SKIP_METRICS = {
    "detach_",
}
