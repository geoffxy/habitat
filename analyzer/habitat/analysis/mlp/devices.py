import os
import pandas as pd


def get_device_features(device_name, device_params):
    file_dir = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(file_dir, "devices.csv"))
    df = df[['device', ] + device_params]
    result = df[df['device'] == device_name].iloc[0]
    return list(result)[1:]

def get_all_devices(device_params=None):
    file_dir = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(file_dir, "devices.csv"))
    if type(device_params) is list:
        df = df[['device',] + device_params]

    return {
        row[1]: list(row[2:]) for row in df.itertuples()
    }
