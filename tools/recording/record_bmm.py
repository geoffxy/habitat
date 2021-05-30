import argparse
import logging

import torch
from record_common import Measurer
import features as f

logger = logging.getLogger(__name__)


def index_to_config(args, index):
    batch = (index % args.batches) + 1
    index //= args.batches

    left = (index % args.left) + 1
    index //= args.left

    middle = (index % args.middle) + 1
    index //= args.middle

    right = index + 1

    return (
        batch,
        left,
        middle,
        right,
    )


def config_to_profiler_args(config):
    (batch, left, middle, right) = config
    o1 = torch.randn((batch, left, middle)).cuda()
    o2 = torch.randn((batch, middle, right)).cuda()
    o1.requires_grad_()
    o2.requires_grad_()
    return {
        'func': torch.bmm,
        'args': (o1, o2),
        'kwargs': {},
    }


def main():
    measurer = Measurer(
        op_name='bmm',
        recorder_config=f.bmm,
        index_to_config=index_to_config,
        config_to_profiler_args=config_to_profiler_args,
    )
    parser = argparse.ArgumentParser()
    measurer.add_args(parser)
    parser.add_argument('--batches', type=int, default=128)
    parser.add_argument('--left', type=int, default=1024)
    parser.add_argument('--middle', type=int, default=1024)
    parser.add_argument('--right', type=int, default=1024)
    args = parser.parse_args()

    num_configs = (
        args.batches *
        args.left *
        args.middle *
        args.right
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
