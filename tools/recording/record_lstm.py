import argparse
import logging

import torch
from record_common import Measurer
import features as f

logger = logging.getLogger(__name__)


def index_to_config(args, index):
    bias = index % 2
    index //= 2

    bidirectional = index % 2
    index //= 2

    batch = (index % args.batches) + 1
    index //= args.batches

    seq_len = (index % args.seq_len) + 1
    index //= args.seq_len

    input_size = (index % args.input_size) + 1
    index //= args.input_size

    hidden_size = (index % args.hidden_size) + 1
    index //= args.hidden_size

    num_layers = index + 1

    return (
        bias,
        bidirectional,
        batch,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
    )


def config_to_profiler_args(config):
    (bias,
     bidirectional,
     batch,
     seq_len,
     input_size,
     hidden_size,
     num_layers) = config
    inputs = torch.randn((seq_len, batch, input_size)).cuda()
    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bool(bias),
        bidirectional=bool(bidirectional),
    ).cuda()
    # NOTE: This is important: for most LSTMs, we will also need the gradient
    #       with respect to the input to be able to backpropagate to earlier
    #       operations in the network.
    inputs = inputs.requires_grad_()
    return {
        'func': lstm,
        'args': (inputs,),
        'kwargs': {},
    }


def main():
    measurer = Measurer(
        op_name='lstm',
        recorder_config=f.lstm,
        index_to_config=index_to_config,
        config_to_profiler_args=config_to_profiler_args,
    )
    parser = argparse.ArgumentParser()
    measurer.add_args(parser)
    parser.add_argument('--batches', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--input-size', type=int, default=1280)
    parser.add_argument('--hidden-size', type=int, default=1280)
    parser.add_argument('--num-layers', type=int, default=6)
    args = parser.parse_args()

    num_configs = (
        2 * # bias
        2 * # bidirectional
        args.batches *
        args.seq_len *
        args.input_size *
        args.hidden_size *
        args.num_layers
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
