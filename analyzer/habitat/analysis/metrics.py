from enum import Enum


class _MetricInfo:
    def __init__(
        self,
        cupti_name,
        legacy_cupti_name,
        legacy_to_canonical_fn
    ):
        self._cupti_name = cupti_name
        self._legacy_cupti_name = legacy_cupti_name
        self._legacy_to_canonical_fn = legacy_to_canonical_fn

    @property
    def cupti_name(self):
        return self._cupti_name

    @property
    def legacy_cupti_name(self):
        return self._legacy_cupti_name

    def to_canonical_value(self, value, device):
        if device.compute_capability[0] >= 7:
            return value
        return self._legacy_to_canonical_fn(value)


class Metric(Enum):
    DRAMUtilization = _MetricInfo(
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram_utilization',
        lambda value: value * 10,
    )
    DRAMReadBytes = _MetricInfo(
        'dram__bytes_read.sum',
        'dram_read_bytes',
        lambda value: value,
    )
    DRAMWriteBytes = _MetricInfo(
        'dram__bytes_write.sum',
        'dram_write_bytes',
        lambda value: value,
    )
    SinglePrecisionFLOPEfficiency = _MetricInfo(
        'smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed',
        'flop_sp_efficiency',
        lambda value: value,
    )
    SinglePrecisionAddOps = _MetricInfo(
        'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum',
        'flop_count_sp_add',
        lambda value: value,
    )


def resolve_metrics(metrics, device):
    """
    Converts Metric enum values into raw metric strings that can be passed to
    CUPTI, depending on the compute capability of the given device.

    This is needed because the metrics names changed after (and including)
    compute capability 7.0 (Volta).

    If the metrics passed in are already resolved, this function will return a
    copy of them.
    """
    if metrics is None:
        return []

    if isinstance(metrics, list) or isinstance(metrics, tuple):
        return [
            _get_metric_name(metric, device)
            for metric in metrics
        ]
    else:
        return [_get_metric_name(metrics, device)]


def _get_metric_name(metric, device):
    if isinstance(metric, Metric):
        return (
            metric.value.cupti_name
            if device.compute_capability[0] >= 7
            else metric.value.legacy_cupti_name
        )
    else:
        return metric
