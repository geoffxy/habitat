import torch


class Arguments:
    """
    Stores representations of an operation's arguments.
    """
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs
        self.special = {}

    @classmethod
    def from_raw_arguments(cls, args, kwargs):
        processed_args = tuple(map(_process_argument, args))
        processed_kwargs = {
            arg_name: _process_argument(arg_value)
            for arg_name, arg_value in kwargs.items()
        }
        return cls(processed_args, processed_kwargs)


def _process_argument(argument):
    if isinstance(argument, tuple):
        return tuple(map(_process_argument, argument))

    if isinstance(argument, list):
        return list(map(_process_argument, argument))

    # At this point we expect the argument to either be a
    # torch.Tensor or to be a scalar (e.g., an integer).
    if isinstance(argument, torch.Tensor):
        # We only store the tensor dimensions
        return argument.size()
    else:
        return argument
