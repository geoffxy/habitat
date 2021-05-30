import argparse
import collections
import itertools
import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

DEVICES = [
    'RTX2070',
    'RTX2080Ti',
    'P4000',
    'T4',
    'P100',
    'V100',
]

E2E_FILE = re.compile(
    '(?P<config_name>[a-zA-Z0-9\+]+)-(?P<origin_device>[a-zA-Z0-9]+)-e2e.csv'
)

OPS_FILE = re.compile(
    '(?P<config_name>[a-zA-Z0-9\+]+)-(?P<origin_device>[a-zA-Z0-9]+)-(?P<dest_device>[a-zA-Z0-9]+)-breakdown.csv'
)


class Index:
    # e2e: Dict(origin_device -> e2e dataframe)
    # ops: Dict((origin_device, dest_device) -> ops dataframe)
    Config = collections.namedtuple(
        'Config',
        ['e2e_predicted', 'ops', 'e2e_actual'],
    )

    @classmethod
    def build(cls, in_dir):
        index = cls()

        for file in os.listdir(in_dir):
            e2e_match = E2E_FILE.match(file)
            ops_match = OPS_FILE.match(file)
            if e2e_match is None and ops_match is None:
                continue

            match = e2e_match if e2e_match is not None else ops_match
            origin_device = match.group('origin_device')
            config_name = match.group('config_name')

            config = index.get_or_create(config_name)
            df = pd.read_csv(os.path.join(in_dir, file))

            if e2e_match is not None:
                predictions = df[df['device'] != origin_device]
                actual = df[df['device'] == origin_device]

                config.e2e_predicted[origin_device] = predictions
                config.e2e_actual[origin_device] = actual.iloc[0]['run_time_ms']
            else:
                dest_device = ops_match.group('dest_device')
                config.ops[(origin_device, dest_device)] = df

        # Make the ground truth e2e measurements a dataframe
        finalized_config = {}
        for config_name, config in index.config.items():
            df_ready = {'device': [], 'run_time_ms': []}
            for device, run_time_ms in config.e2e_actual.items():
                df_ready['device'].append(device)
                df_ready['run_time_ms'].append(run_time_ms)
            finalized_config[config_name] = Index.Config(
                config.e2e_predicted,
                config.ops,
                pd.DataFrame.from_dict(df_ready),
            )
        index.config = finalized_config

        return index

    def __init__(self):
        self.config = {}

    def get_or_create(self, config_name):
        if config_name not in self.config:
            self.config[config_name] = Index.Config({}, {}, {})
        return self.config[config_name]


def percent_error(predicted, actual):
    return (predicted - actual) / actual


def e2e_results(config_name, config, out_e2e):
    all_frames = []
    actual = config.e2e_actual

    for origin_device in DEVICES:
        if origin_device not in config.e2e_predicted:
            continue
        predictions = config.e2e_predicted[origin_device]

        merged = pd.merge(
            predictions,
            actual,
            on='device',
            suffixes=('_predicted', '_measured'),
        )
        merged['origin_device'] = origin_device
        merged = merged.rename(columns={'device': 'dest_device'})
        merged = merged[[
            'origin_device',
            'dest_device',
            'run_time_ms_predicted',
            'run_time_ms_measured',
        ]]
        merged['pct_error'] = percent_error(
            predicted=merged['run_time_ms_predicted'],
            actual=merged['run_time_ms_measured'],
        )
        all_frames.append(merged)

    all_data = pd.concat(all_frames, axis='index', ignore_index=True)
    all_data = all_data.sort_values(by=['origin_device', 'dest_device'])

    file_name = '{}-e2e-combined.csv'.format(config_name)
    all_data.to_csv(os.path.join(out_e2e, file_name), index=False)


def ops_results(config_name, config, out_ops):
    for origin_device, dest_device in itertools.permutations(DEVICES, 2):
        if (origin_device, dest_device) not in config.ops:
            continue
        if (dest_device, dest_device) not in config.ops:
            continue

        predictions = config.ops[(origin_device, dest_device)]
        actual = config.ops[(dest_device, dest_device)]

        if len(predictions) != len(actual):
            logger.warn(
                'Skipping %s -> %s operations because their lengths mismatch '
                '(%d vs. %d)!',
                origin_device.name,
                dest_device.name,
                len(predictions),
                len(actual),
            )
            continue

        if not predictions['operation'].equals(actual['operation']):
            logger.warn(
                'Skipping %s -> %s due to an operation mismatch.',
                origin_device.name,
                dest_device.name,
            )
            continue

        combined = predictions.copy()
        combined['run_time_ms_measured'] = actual['run_time_ms']
        combined = combined.rename(columns={'run_time_ms': 'run_time_ms_predicted'})
        combined['pct_error'] = percent_error(
            predicted=combined['run_time_ms_predicted'],
            actual=combined['run_time_ms_measured'],
        )

        file_name = '{}-{}-{}-breakdown-combined.csv'.format(
            config_name,
            origin_device,
            dest_device,
        )
        combined.to_csv(os.path.join(out_ops, file_name), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--out-e2e', type=str, required=True)
    parser.add_argument('--out-ops', type=str, required=True)
    args = parser.parse_args()

    index = Index.build(args.in_dir)

    for config_name, config in index.config.items():
        e2e_results(config_name, config, args.out_e2e)
        ops_results(config_name, config, args.out_ops)


if __name__ == '__main__':
    main()
