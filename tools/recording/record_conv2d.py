import argparse
import logging
import math

import torch
from record_common import Measurer
import features as f

logger = logging.getLogger(__name__)

MIN_IN_CHANNELS = 3
MIN_OUT_CHANNELS = 16

torch.backends.cudnn.benchmark = True


def index_to_config(args, index):
    bias = False if index % 2 == 0 else True
    index //= 2

    batch = (index % args.batches) + 1
    index //= args.batches

    image_size = (index % args.image_size) + 1
    index //= args.image_size

    in_channels = (index % args.in_channels) + 1
    index //= args.in_channels

    out_channels = (index % args.out_channels) + 1
    index //= args.out_channels

    kernel_size = (index % args.kernel_size) + 1
    index //= args.kernel_size

    stride = (index % args.stride) + 1
    index //= args.stride

    # Padding is 0-based
    padding = index

    return (
        bias,
        batch,
        image_size,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    )


def index_filter(args, index):
    config = index_to_config(args, index)
    # NOTE: We multiply because the dimensions have different ranges; we want
    #       them to each "contribute equally". We weigh the image size more to
    #       select smaller image sizes.
    # image_size (1-dim) * in_channels * out_channels * kernel_size
    conv_size = math.pow(config[2], 1.15) * config[3] * config[4] * config[5]

    # NOTE: This value was chosen arbitrarily: we don't want the in/out
    #       channels and image size to all be too large. This way, large values
    #       for the in/out channels would lead to a smaller image size (and
    #       vice versa).
    return conv_size <= 35000000


def config_to_profiler_args(config):
    (bias,
     batch,
     image_size,
     in_channels,
     out_channels,
     kernel_size,
     stride,
     padding) = config

    # Easiest way to exclude certain sample configurations
    if in_channels < MIN_IN_CHANNELS or out_channels < MIN_OUT_CHANNELS:
        return None

    device = torch.device('cuda')
    conv2d = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    ).to(device)
    inp = torch.randn((
        batch,
        in_channels,
        image_size,
        image_size,
    ), device=device)
    # NOTE: This is important: for most convolutions, we will also need the
    #       gradient with respect to the input to be able to backpropagate to
    #       earlier operations in the network.
    inp = inp.requires_grad_()

    return {
        'func': conv2d,
        'args': (inp,),
        'kwargs': {},
    }


def main():
    measurer = Measurer(
        op_name='conv2d',
        recorder_config=f.conv2d,
        index_to_config=index_to_config,
        index_filter=index_filter,
        config_to_profiler_args=config_to_profiler_args,
    )

    parser = argparse.ArgumentParser()
    measurer.add_args(parser)
    parser.add_argument('--batches', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--in-channels', type=int, default=2048)
    parser.add_argument('--out-channels', type=int, default=2048)
    parser.add_argument('--kernel-size', type=int, default=11)
    parser.add_argument('--stride', type=int, default=4)
    # Padding is 0-based, so this means we consider 0 to 3 inclusive
    parser.add_argument('--padding', type=int, default=4)
    args = parser.parse_args()

    num_configs = (
        2 * # Whether or not there is a bias
        args.batches *
        args.image_size *
        args.in_channels *
        args.out_channels *
        args.kernel_size *
        args.stride *
        args.padding
    )

    # Conv2d has filtering, so we won't have exactly 200000 points (the
    # default). So here we increase the number of starting points.
    if args.num_points == 200000:
        args.num_points *= 6

    measurer.measure_configurations(args, num_configs)


if __name__ == '__main__':
    kwargs = {
        "format": "%(asctime)s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M",
        "level": logging.INFO,
    }
    logging.basicConfig(**kwargs)
    main()
