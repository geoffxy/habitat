import argparse
import enum
import functools
import itertools
import os
import re
import sys
import sqlite3
import yaml

import pandas as pd
import torch

from features import FEATURES

pd.options.mode.chained_assignment = None

SELECT_QUERY = """
  SELECT {features}, SUM(run_time_ms) AS run_time_ms
  FROM recordings
  GROUP BY {features}
"""

file_name_regex = re.compile(
    '(?P<operation>[a-zA-Z0-9]+)-(?P<device>[a-zA-Z0-9]+)\.sqlite'
)


class Generation(enum.Enum):
    Pascal = 0
    Volta = 1
    Turing = 2


DEVICE_GENERATION = {
    'RTX2080Ti': Generation.Turing,
    'RTX2070': Generation.Turing,
    'T4': Generation.Turing,
    'V100': Generation.Volta,
    'P100': Generation.Pascal,
    'P4000': Generation.Pascal,
}


def get_merged_data(args):
    relevant_files = extract_relevant_files(args)
    devices = list(relevant_files.keys())
    print('Detected files:')
    print(relevant_files.values())
    print('Detected devices:')
    print(relevant_files.keys())
    print()

    all_data = load_data(args, relevant_files)
    print('Data summary ({}):'.format(args.operation))
    for device, dataset in all_data.items():
        print(device, '->', len(dataset))
    print()

    merged_data = merge_data(args, all_data)
    print('Merged data:', len(merged_data))
    print()
    return merged_data, devices


def extract_relevant_files(args):
    relevant_files = {}
    for file_name in os.listdir(args.in_dir):
        match = file_name_regex.match(file_name)
        if match is None:
            continue
        if match.group('operation') != args.operation:
            continue
        relevant_files[match.group('device')] = file_name
    return relevant_files


def load_data(args, relevant_files):
    query = SELECT_QUERY.format(
        features=', '.join(FEATURES[args.operation]),
    )
    data = {}
    for device, file_name in relevant_files.items():
        connection = sqlite3.connect(os.path.join(args.in_dir, file_name))
        try:
            data[device] = pd.read_sql(query, connection)
        finally:
            connection.close()
    return data


def merge_data(args, dataset):
    if args.max_run_time_ms is not None:
        # We filter here to ensure we have scaling examples between ALL pairs
        # of devices.
        filtered_dataset = {
            device: df[df['run_time_ms'] <= args.max_run_time_ms]
            for device, df in dataset.items()
        }
    else:
        filtered_dataset = dataset

    dataset_renamed = [
        data.rename(columns={'run_time_ms': device})
        for device, data in filtered_dataset.items()
    ]
    return functools.reduce(
        lambda df1, df2: pd.merge(df1, df2, on=FEATURES[args.operation]),
        dataset_renamed,
    )


def load_device_info(args):
    with open(args.device_info) as devices_yaml:
        raw_yaml = yaml.load(devices_yaml, Loader=yaml.Loader)

    columns = ['device', 'mem_bandwidth_gb', 'num_sms']
    dfs = []
    for device, info in raw_yaml.items():
        dfs.append(pd.DataFrame([[
            device, info['mem_bandwidth_gb'], info['num_sms']
        ]], columns=columns))

    return pd.concat(dfs, axis='index', ignore_index=True)


def device_pair_data(args, merged_data, device1, device2):
    features = FEATURES[args.operation]
    data = merged_data[features + [device1, device2]]
    data.columns = features + ['origin_ms', 'dest_ms']
    data['origin_device'] = device1
    data['dest_device'] = device2
    return data


def to_all_pairs(args, merged_data, devices):
    features = FEATURES[args.operation]
    all_data = []
    for (i, j) in itertools.combinations(range(len(devices)), 2):
        all_data.append(device_pair_data(
            args, merged_data, devices[i], devices[j],
        ))
        all_data.append(device_pair_data(
            args, merged_data, devices[j], devices[i],
        ))
    combined = pd.concat(all_data, axis='index')
    combined = combined[
        ['origin_device', 'dest_device'] + features + ['origin_ms', 'dest_ms']
    ]
    return combined.reset_index(drop=True)


def add_device_info(args, all_pairs, device_info):
    w_origin = all_pairs.merge(
        device_info, left_on='origin_device', right_on='device',
    )
    w_dest = w_origin.merge(
        device_info,
        left_on='dest_device',
        right_on='device',
        suffixes=('_origin', '_dest'),
    )
    w_dest['bw_ratio_dest_origin'] = (
        w_dest['mem_bandwidth_gb_dest'] / w_dest['mem_bandwidth_gb_origin']
    )
    w_dest['sm_ratio_dest_origin'] = (
        w_dest['num_sms_dest'] / w_dest['num_sms_origin']
    )
    data = w_dest.drop(columns=[
        'device_origin', 'device_dest',
        'mem_bandwidth_gb_dest', 'mem_bandwidth_gb_origin',
        'num_sms_dest', 'num_sms_origin',
    ])
    data['origin_gen'] = data['origin_device'].map(
        lambda device: DEVICE_GENERATION[device].value
    )
    data['dest_gen'] = data['dest_device'].map(
        lambda device: DEVICE_GENERATION[device].value
    )
    return data


def warn_overlapping(args, dataset):
    if args.exclude_data is None:
        return

    batches = [16, 32, 48, 64]
    all_batches = []
    exclude = pd.read_csv(args.exclude_data)
    exclude.pop('batch')

    for batch in batches:
        df = exclude.copy()
        df['batch'] = batch
        all_batches.append(df)

    to_exclude = pd.concat(all_batches, axis='index', ignore_index=True)
    overlap = pd.merge(dataset, to_exclude, on=list(to_exclude.columns))

    if len(overlap) == 0:
        print('OK - No test and train overlap!')
    else:
        print('!!WARNING!! Train set overlap with test points:', len(overlap))


def partition_dataset(args, dataset):
    features = FEATURES[args.operation]
    data_subset = dataset[
        ['origin_gen',
         'dest_gen',
         'bw_ratio_dest_origin',
         'sm_ratio_dest_origin'] +
        features +
        ['origin_ms', 'dest_ms']
    ]

    train = data_subset.sample(frac=args.train_split, random_state=1337)
    test = data_subset.drop(train.index)
    val = test.sample(frac=0.5, random_state=1337)
    test = test.drop(val.index)

    warn_overlapping(args, train)

    results = {
        'train': train.reset_index(drop=True),
        'val': val.reset_index(drop=True),
        'test': test.reset_index(drop=True),
    }
    print('Partitioned data:')
    for split, data in results.items():
        print(split, '->', len(data))
    print()

    return results


def normalize_and_split_to_torch(args, partitioned_dataset):
    split_data = {}
    for name, data in partitioned_dataset.items():
        targets = data.pop('dest_ms')
        categorical = data[['origin_gen', 'dest_gen']]
        numerical = data.drop(columns=['origin_gen', 'dest_gen'])
        split_data[name] = (categorical, numerical, targets)

    norm_by = split_data['train'][1].describe().transpose()
    normalized = {}
    for name, data in split_data.items():
        categorical = torch.tensor(data[0].values, dtype=torch.int64)
        numerical_df = (data[1] - norm_by['mean']) / norm_by['std']
        numerical = torch.tensor(numerical_df.values, dtype=torch.float32)
        targets = torch.tensor(data[2].values, dtype=torch.float32)

        normalized[name] = torch.utils.data.TensorDataset(
            categorical,
            numerical,
            targets
        )

    return {
        'data': normalized,
        'normalize': {
            'mean': torch.tensor(norm_by['mean'].values, dtype=torch.float32),
            'std': torch.tensor(norm_by['std'].values, dtype=torch.float32),
        },
        'features': [
            'origin_gen',
            'dest_gen',
            'bw_ratio_dest_origin',
            'sm_ratio_dest_origin',
            *FEATURES[args.operation],
            'origin_ms',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str)
    parser.add_argument('--in-dir', type=str, required=True)
    parser.add_argument('--device-info', type=str, required=True)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--exclude-data', type=str)
    parser.add_argument(
        '--max-run-time-ms',
        type=float,
        help='The maximum run time to allow in a data entry. If set, this '
             'script will filter out data samples with run times longer than '
             'this argument.',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    device_info = load_device_info(args)

    output_file = '{}-torch-dataset.pt'.format(args.operation)
    if os.path.exists(output_file):
        print(
            'ERROR: The output file {} already exists. '
            'Aborting to avoid overwriting.'.format(output_file),
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Select all data from the combined SQLite data files and merge the
    #    entries together so that we have measurements for all devices.
    merged_data, devices = get_merged_data(args)

    # 2. Massage the data into "all pairs" format where we create entries for
    #    all combinations of origin -> destination device.
    all_pairs_data = to_all_pairs(args, merged_data, devices)
    del merged_data

    # 3. Add GPU-specific data to each data entry
    with_device_info = add_device_info(args, all_pairs_data, device_info)
    del all_pairs_data

    print('Dataset preview:')
    print(with_device_info.head())
    print()

    # 4. Split the data into test and train data
    partitioned_dataset = partition_dataset(args, with_device_info)
    del with_device_info

    # 5. Normalize and convert to PyTorch TensorDatasets
    torch_dataset = normalize_and_split_to_torch(args, partitioned_dataset)
    del partitioned_dataset

    print('Final dataset statistics:')
    for name, data in torch_dataset['data'].items():
        print(name, '->', len(data))
    print()

    test_input = torch_dataset['data']['train'][0]
    print(
        'Example input sizes:',
        test_input[0].size(),
        test_input[1].size(),
        test_input[2].size(),
    )
    print('Example input:')
    print(test_input)
    print()

    print('All features:')
    print(torch_dataset['features'])
    print()

    if not args.dry_run:
        torch.save(torch_dataset, output_file)
    print('Done!')


if __name__ == '__main__':
    main()
