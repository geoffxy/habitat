from itertools import chain
from habitat.analysis.predictor import Predictor


class Trace:
    """
    Represents an operation trace that was measured on a given device.
    """

    # Used by default to make cross-device predictions.
    DefaultPredictor = Predictor()

    def __init__(self, device, operations):
        self._device = device
        self._operations = operations
        self._run_time_ms = None

    @property
    def operations(self):
        return self._operations

    @property
    def device(self):
        return self._device

    @property
    def run_time_ms(self):
        if self._run_time_ms is not None:
            return self._run_time_ms

        self._run_time_ms = sum(map(
            lambda op: op.run_time_ms,
            self._operations,
        ))

        return self._run_time_ms

    def to_device(self, dest_device, predictor=None):
        """Get a predicted trace for the specified device."""
        if dest_device.name == self.device.name:
            return self

        actual_predictor = (
            Trace.DefaultPredictor if predictor is None else predictor
        )

        operations = [
            operation.to_device(dest_device, actual_predictor)
            for operation in self._operations
        ]
        return Trace(dest_device, operations)
