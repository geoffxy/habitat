import argparse
import collections
import csv
import os

import numpy as np
import torch

import habitat
from habitat.analysis import SPECIAL_OPERATIONS
from habitat.profiling.run_time import RunTimeProfiler

###############################################################################

# Experiment configuration

DEVICES = [
    habitat.Device.P4000,
    habitat.Device.P100,
    habitat.Device.V100,
    habitat.Device.T4,
    habitat.Device.RTX2070,
    habitat.Device.RTX2080Ti,
]

RESNET50_BATCHES = [16, 32, 64]
GNMT_BATCHES = [16, 32, 48]
TRANSFORMER_BATCHES = [32, 48, 64]
DCGAN_BATCHES = [64, 96, 128]

###############################################################################

Context = collections.namedtuple(
    'Context',
    ['origin_device', 'profiler', 'percentile'],
)

torch.backends.cudnn.benchmark = True


def record_e2e(config_name, origin_device, data):
    file_name = '{}-{}-e2e.csv'.format(config_name, origin_device.name)
    with open(file_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['device', 'run_time_ms'])
        for device, run_time_ms in data:
            writer.writerow([device.name, run_time_ms])


def record_breakdown(config_name, origin_device, dest_device, trace):
    file_name = '{}-{}-{}-breakdown.csv'.format(
        config_name,
        origin_device.name,
        dest_device.name,
    )
    with open(file_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['operation', 'run_time_ms'])
        for op in trace.operations:
            writer.writerow([op.name, op.run_time_ms])


def compute_threshold(runnable, context):
    tracker = habitat.OperationTracker(context.origin_device)
    with tracker.track():
        runnable()

    run_times = []
    trace = tracker.get_tracked_trace()
    for op in trace.operations:
        if op.name in SPECIAL_OPERATIONS:
            continue
        run_times.append(op.forward.run_time_ms)
        if op.backward is not None:
            run_times.append(op.backward.run_time_ms)

    return np.percentile(run_times, context.percentile)


def run_experiment_config(config_name, runnable, context):
    print('Processing:', config_name)
    origin_run_time_ms = context.profiler.measure_ms(runnable)
    e2e_results = [(context.origin_device, origin_run_time_ms)]

    threshold = compute_threshold(runnable, context)
    tracker = habitat.OperationTracker(
        device=context.origin_device,
        metrics=[
            habitat.Metric.SinglePrecisionFLOPEfficiency,
            habitat.Metric.DRAMReadBytes,
            habitat.Metric.DRAMWriteBytes,
        ],
        metrics_threshold_ms=threshold,
    )
    with tracker.track():
        runnable()

    trace = tracker.get_tracked_trace()
    record_breakdown(
        config_name,
        context.origin_device,
        context.origin_device,
        trace,
    )

    for device in DEVICES:
        if device.name == context.origin_device.name:
            continue
        predicted_trace = trace.to_device(device)
        record_breakdown(
            config_name,
            context.origin_device,
            device,
            predicted_trace,
        )
        e2e_results.append((device, predicted_trace.run_time_ms))

    record_e2e(config_name, context.origin_device, e2e_results)


def run_resnet50_experiments(context):
    import resnet.entry_point as rep

    model = rep.skyline_model_provider()
    iteration = rep.skyline_iteration_provider(model)

    for batch_size in RESNET50_BATCHES:
        inputs = rep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            'resnet50+{}'.format(batch_size),
            runnable,
            context,
        )


def run_dcgan_experiments(context):
    import dcgan.entry_point as dep

    models = dep.skyline_model_provider()
    iteration = dep.skyline_iteration_provider(*models)

    for batch_size in DCGAN_BATCHES:
        inputs = dep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            'dcgan+{}'.format(batch_size),
            runnable,
            context,
        )


def run_inception_experiments(context):
    import inception.entry_point as iep

    model = iep.skyline_model_provider()
    iteration = iep.skyline_iteration_provider(model)

    # N.B. We use the same batch sizes as resnet
    for batch_size in RESNET50_BATCHES:
        inputs = iep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            'inception+{}'.format(batch_size),
            runnable,
            context,
        )


def run_gnmt_experiments(context):
    import gnmt.entry_point as gep

    model = gep.skyline_model_provider()
    iteration = gep.skyline_iteration_provider(model)

    for batch_size in GNMT_BATCHES:
        inputs = gep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            'gnmt+{}'.format(batch_size),
            runnable,
            context,
        )


def run_transformer_experiments(context):
    import transformer.entry_point as tep

    model = tep.skyline_model_provider()
    iteration = tep.skyline_iteration_provider(model)

    for batch_size in TRANSFORMER_BATCHES:
        inputs = tep.skyline_input_provider(batch_size=batch_size)

        def runnable():
            iteration(*inputs)

        run_experiment_config(
            'transformer+{}'.format(batch_size),
            runnable,
            context,
        )


def main():
    import habitat.habitat_cuda as hc

    parser = argparse.ArgumentParser()
    parser.add_argument('device', type=str)
    parser.add_argument('--percentile', type=float, default=99.5)
    args = parser.parse_args()

    # Ask the profiler to cache metrics for kernels that share the same name
    # and launch configuration.
    hc.set_cache_metrics(True)

    origin_device = getattr(habitat.Device, args.device)
    profiler = RunTimeProfiler()

    context = Context(
        origin_device=origin_device,
        profiler=profiler,
        percentile=args.percentile,
    )

    run_dcgan_experiments(context)
    run_inception_experiments(context)
    run_resnet50_experiments(context)
    run_gnmt_experiments(context)
    run_transformer_experiments(context)


if __name__ == '__main__':
    main()
