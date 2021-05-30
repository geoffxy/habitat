from habitat.analysis.metrics import Metric
from habitat.analysis.wave_scaling.resimplified import (
    resimplified_wave_scaling,
)
from habitat.analysis.wave_scaling.roofline import roofline_wave_scaling


def unified_wave_scaling(
    kernel,
    origin_device,
    dest_device,
    metadata_manager,
):
    try:
        # Try reading metrics. These calls will raise exceptions if the metrics
        # do not exist.
        _ = kernel.get_metric(Metric.SinglePrecisionFLOPEfficiency)
        _ = kernel.get_metric(Metric.DRAMReadBytes)
        _ = kernel.get_metric(Metric.DRAMWriteBytes)
        return roofline_wave_scaling(
            kernel,
            origin_device,
            dest_device,
            metadata_manager,
        )
    except AttributeError:
        pass

    # Use resimplified wave scaling when metrics are unavailable
    return resimplified_wave_scaling(
        kernel,
        origin_device,
        dest_device,
        metadata_manager,
    )
