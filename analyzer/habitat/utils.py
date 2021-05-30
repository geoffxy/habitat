import logging
import os
import yaml


def set_up_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M',
    )


def add_common_cmd_args(parser):
    parser.add_argument('model_path', help='The serialized model to analyze')
    parser.add_argument(
        'model_config_path',
        help='The configuration file for the model',
    )
    parser.add_argument(
        '--device-config',
        type=str,
        default='devices.yml',
        help='The config file containing GPU device properties.',
    )
    parser.add_argument(
        '--origin-device',
        type=str,
        required=True,
        help='The GPU on which the analysis is being performed.',
    )
    parser.add_argument(
        '--kernel-lut',
        type=str,
        default=os.path.join('lutfiles', 'kernels.sqlite'),
        help='The path to the kernel metadata look up table.',
    )
    parser.add_argument(
        '--operation-lut',
        type=str,
        default=os.path.join('lutfiles', 'operations.sqlite'),
        help='The path to the operation run time look up table.',
    )


def ns_to_ms(ns):
    return ns / 1e6


def ms_to_ns(ms):
    return ms * 1e6


def name_all_arguments(all_parameters, args, kwargs):
    """
    This function merges positional and keyword arguments
    into one dictionary based on the declared names of the
    function's parameters.
    """
    merged = {**kwargs}
    for arg_name, arg in zip(all_parameters, args):
        merged[arg_name] = arg
    return merged
