

class MeasuredKernel:
    def __init__(self, time_kernel, metrics_kernels, device):
        self._c = time_kernel
        self._metrics_kernels = metrics_kernels
        self._device = device
        self._cached_metrics = {}

    def get_metric(self, metric_info, default=None):
        if metric_info in self._cached_metrics:
            return self._cached_metrics[metric_info]

        for metric_kernel in self._metrics_kernels:
            for raw_metric_name, raw_metric_value in metric_kernel.metrics:
                if (raw_metric_name == metric_info.value.cupti_name or
                        raw_metric_name == metric_info.value.legacy_cupti_name):
                    canonical_value = metric_info.value.to_canonical_value(
                        raw_metric_value, self._device)
                    self._cached_metrics[metric_info] = canonical_value
                    return canonical_value

        if default is None:
            raise AttributeError('Unknown metric: {}'.format(metric_info.name))

        return default

    def __getattr__(self, name):
        # Delegate to the underlying C++ object for non-overridden attributes
        return getattr(self._c, name)


class PredictedKernel:
    def __init__(self, measured_kernel, run_time_ns):
        self._measured_kernel = measured_kernel
        self._run_time_ns = run_time_ns

    @property
    def run_time_ns(self):
        return self._run_time_ns

    @property
    def name(self):
        return self._measured_kernel.name
