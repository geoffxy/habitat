import argparse
import logging

import torch
from record_common import Measurer
import features as f

logger = logging.getLogger(__name__)


def index_to_config(args, index):
    bias = False if index % 2 == 0 else True
    index //= 2

    batch = (index % args.batches) + 1
    index //= args.batches

    in_features = (index % args.in_features) + 1
    index //= args.in_features

    out_features = index + 1

    return (
        bias,
        batch,
        in_features,
        out_features,
    )


def config_to_profiler_args(config):
    (bias, batch, in_features, out_features) = config
    linear = torch.nn.Linear(
        in_features=in_features, out_features=out_features, bias=bias).cuda()
    inp = torch.randn((batch, in_features)).cuda()
    # NOTE: This is important: for most linear layers, we will also need the
    #       gradient with respect to the input to be able to backpropagate to
    #       earlier operations in the network.
    inp = inp.requires_grad_()
    return {
        'func': linear,
        'args': (inp,),
        'kwargs': {},
    }


def index_filter(args, index):
    config = index_to_config(args, index)
    # NOTE: We multiply because the dimensions have different ranges; we want
    #       them to each "contribute equally". We weigh the image size more to
    #       select smaller image sizes.
    # batch * in_features * out_features
    linear_size = config[1] * config[2] * config[3]

    # NOTE: This value was chosen arbitrarily: we don't want the in/out
    #       features and batch to all be too large. This way, large values
    #       for the in/out features would lead to a smaller batch size (and
    #       vice versa).
    return linear_size <= 840000000


def main():
    measurer = Measurer(
        op_name='linear',
        recorder_config=f.linear,
        index_to_config=index_to_config,
        index_filter=index_filter,
        config_to_profiler_args=config_to_profiler_args,
    )
    parser = argparse.ArgumentParser()
    measurer.add_args(parser)
    parser.add_argument('--batches', type=int, default=3500)
    parser.add_argument('--in-features', type=int, default=32768)
    parser.add_argument('--out-features', type=int, default=32768)
    args = parser.parse_args()

    # Linear has filtering, so we won't have exactly 200000 points (the
    # default). So here we increase the number of starting points.
    if args.num_points == 200000:
        args.num_points *= 80

    num_configs = (
        2 * # Whether or not there is a bias
        args.batches *
        args.in_features *
        args.out_features
    )
    measurer.measure_configurations(args, num_configs)


if __name__ == '__main__':
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.INFO,
    }
    logging.basicConfig(**kwargs)
    main()
