

class Operation:
    """
    Abstract representation of a logical operation in a model.
    """
    def __repr__(self):
        return self.name

    @property
    def run_time_ms(self):
        if self.backward is None:
            return self.forward.run_time_ms
        return self.forward.run_time_ms + self.backward.run_time_ms

    @property
    def ktime_ns(self):
        if self.backward is None:
            return self.forward.ktime_ns
        return self.forward.ktime_ns + self.backward.ktime_ns

    @property
    def arguments(self):
        return None

    @property
    def forward(self):
        raise NotImplementedError

    @property
    def backward(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def to_device(self, dest_device, predictor):
        raise NotImplementedError


class MeasuredOperation(Operation):
    def __init__(
        self,
        name,
        arguments,
        forward,
        backward,
        device,
    ):
        super().__init__()
        self._name = name
        self._arguments = arguments
        self._forward = forward
        self._backward = backward
        self._device = device

    @property
    def name(self):
        return self._name

    @property
    def arguments(self):
        return self._arguments

    @property
    def forward(self):
        return self._forward

    @property
    def backward(self):
        return self._backward

    @property
    def device(self):
        return self._device

    def to_device(self, dest_device, predictor):
        if dest_device.name == self._device.name:
            return self
        return predictor.predict_operation(self, dest_device)


class PredictedOperation(Operation):
    def __init__(
        self,
        measured_operation,
        forward,
        backward,
        device
    ):
        self._measured_operation = measured_operation
        self._forward = forward
        self._backward = backward
        self._device = device

    @property
    def name(self):
        return self._measured_operation.name

    @property
    def arguments(self):
        return self._measured_operation.arguments

    @property
    def forward(self):
        return self._forward

    @property
    def backward(self):
        return self._backward

    @property
    def device(self):
        return self._device

    def to_device(self, dest_device, predictor):
        raise RuntimeError(
            'Cannot make a prediction using a predicted operation.',
        )
