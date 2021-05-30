import argparse
import statistics

import habitat
import habitat.habitat_cuda as hc
from habitat.analysis.metrics import Metric
from habitat.profiling.kernel import KernelProfiler


def measure_peak_flops(profiler):
    results = profiler.measure_kernels(hc._diagnostics.run_flop_test)
    assert len(results) == 1
    kernel = results[0]
    gflops_per_second = (
        kernel.get_metric(Metric.SinglePrecisionAddOps) / kernel.run_time_ns
    )
    efficiency = kernel.get_metric(Metric.SinglePrecisionFLOPEfficiency) / 100
    return gflops_per_second / efficiency


def main():
    parser = argparse.ArgumentParser(
        description="Measure the peak performance (FLOP/s) of a GPU."
    )
    parser.add_argument("device", help="The current device (e.g., RTX2070).")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    profiler = KernelProfiler(
        getattr(habitat.Device, args.device),
        metrics=[
            Metric.SinglePrecisionFLOPEfficiency,
            Metric.SinglePrecisionAddOps,
        ],
    )

    results = []
    for trial in range(args.trials):
        print("Running trial {}...".format(trial))
        results.append(measure_peak_flops(profiler))

    print("Peak Performance on the {}".format(args.device))
    print("===============================")
    print("Median: {} GFLOP/s".format(statistics.median(results)))
    print("Mean:   {} GFLOP/s".format(statistics.mean(results)))
    print("Max.:   {} GFLOP/s".format(max(results)))
    print("Min.:   {} GFLOP/s".format(min(results)))
    print("Trials: {}".format(args.trials))


if __name__ == "__main__":
    main()
