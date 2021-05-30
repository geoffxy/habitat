import sqlite3
import pandas as pd
import glob
import functools
from tqdm import tqdm

from habitat.analysis.mlp.devices import get_device_features, get_all_devices


def get_dataset(path, features, device_features=None):
    if device_features is None:
        device_features = ['mem', 'mem_bw', 'num_sm', 'single']

    SELECT_QUERY = """
      SELECT {features}, SUM(run_time_ms) AS run_time_ms
      FROM recordings
      GROUP BY {features}
    """

    # read datasets
    files = glob.glob(path + "/*.sqlite")

    # read individual sqlite files and categorize by device
    devices = dict()
    for f in files:
        device_name = f.split("/")[-1].split("-")[1]

        conn = sqlite3.connect(f)
        query = SELECT_QUERY.format(features=",".join(features))

        df = pd.read_sql_query(query, conn)
        df = df.rename(columns={"run_time_ms": device_name})

        print("Loaded file %s (%d entries)" % (f, len(df.index)))

        if device_name not in devices:
            devices[device_name] = df
        else:
            devices[device_name] = devices[device_name].append(df)

    for device in devices.keys():
        print("Device %s contains %d entries" % (device, len(devices[device].index)))

    print()

    print("Merging")
    df_merged = functools.reduce(
        lambda df1, df2: pd.merge(df1, df2, on=features),
        devices.values()
    )

    print("Generating dataset")
    # generate vectorized dataset (one entry for each device with device params)
    device_params = get_all_devices(device_features)

    x, y = [], []
    for device in devices.keys():
        df_merged_device = df_merged[features + [device, ]]
        for row in tqdm(df_merged_device.iterrows(), leave=False, desc=device, total=len(df_merged_device.index)):
            row = row[1]

            x.append(list(row[:-1]) + device_params[device])
            y.append(row[-1])

    return x, y
