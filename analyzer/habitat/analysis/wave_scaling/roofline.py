import math

from habitat.analysis.metrics import Metric
from habitat.analysis.kernels import PredictedKernel
from habitat.analysis.wave_scaling.common import calculate_wave_info


def roofline_wave_scaling(
    kernel,
    origin_device,
    dest_device,
    metadata_manager,
):
    gamma = _roofline_gamma(kernel, origin_device, dest_device)
    gamma_compl = 1.0 - gamma

    origin_wave_size, dest_wave_size, origin_occupancy, dest_occupancy = (
        calculate_wave_info(
            kernel,
            origin_device,
            dest_device,
            metadata_manager,
        )
    )

    # 1. Check if the kernel is too "small" - if it doesn't fill a single wave
    #    on the current device AND if it doesn't fill a single wave on the
    #    destination device
    if (kernel.num_blocks // origin_wave_size == 0 and
            kernel.num_blocks // dest_wave_size == 0):
        # We scale the run time by the compute factor only
        origin_max_occupancy = math.ceil(
            kernel.num_blocks / origin_device.num_sms
        )
        dest_max_occupancy = math.ceil(
            kernel.num_blocks / dest_device.num_sms
        )
        partial_compute_factor = (
            (origin_device.base_clock_mhz / dest_device.base_clock_mhz) *
            (dest_max_occupancy / origin_max_occupancy)
        )
        return PredictedKernel(
            kernel,
            kernel.run_time_ns * math.pow(partial_compute_factor, gamma_compl),
        )

    # 2. Compute the three scaling factors
    bandwidth_factor = (
        origin_device.mem_bandwidth_gb / dest_device.mem_bandwidth_gb
    )
    clock_factor = (
        origin_device.base_clock_mhz / dest_device.base_clock_mhz
    )
    sm_factor = (
        origin_device.num_sms / dest_device.num_sms
    )

    # 3. Scale and return the predicted run time
    scaled_run_time_ns = (
        kernel.run_time_ns *
        math.pow(bandwidth_factor, gamma) *
        math.pow(clock_factor, gamma_compl) *
        math.pow(sm_factor, gamma_compl)
    )
    return PredictedKernel(kernel, scaled_run_time_ns)


def _roofline_gamma(kernel, origin_device, dest_device):
    flop_efficiency = kernel.get_metric(Metric.SinglePrecisionFLOPEfficiency)
    dram_read_bytes = kernel.get_metric(Metric.DRAMReadBytes)
    dram_write_bytes = kernel.get_metric(Metric.DRAMWriteBytes)
    total_gb = (dram_read_bytes + dram_write_bytes) / 1024 / 1024 / 1024

    gflops_per_second = flop_efficiency / 100 * origin_device.peak_gflops_per_second
    num_gflops = gflops_per_second * kernel.run_time_ns / 1e9

    # We only consider the dest ridge point (R).
    # We use a decreasing linear function to interpolate between an intensity
    # of 0 and R, and use a 1/x function to map intensities greater than R.
    #
    #  gamma = -0.5/R * intensity + 1  if 0 <= intensity <= R
    #          0.5R / intensity        otherwise

    if num_gflops < 1e-9:
        # We treat these cases as fully memory bandwidth bound, even though
        # total_gb could also be 0
        gamma = 1.

    elif total_gb == 0:
        # num_gflops must be non-zero, so this means the kernel is fully
        # compute bound
        gamma = 0.

    else:
        intensity_gflops_per_gb = num_gflops / total_gb
        dest_ridge_point = _ridge_point(dest_device)

        if intensity_gflops_per_gb > dest_ridge_point:
            gamma = 0.5 * dest_ridge_point / intensity_gflops_per_gb
        else:
            gamma = -0.5 / dest_ridge_point * intensity_gflops_per_gb + 1.

    assert gamma >= 0 and gamma <= 1
    return gamma


def _ridge_point(device):
    return device.peak_gflops_per_second / device.mem_bandwidth_gb
