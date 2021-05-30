import functools
import logging
import operator
import os

from habitat.analysis import SPECIAL_OPERATIONS
from habitat.analysis.operation import PredictedOperation
from habitat.analysis.run_time import RunTimePrediction, RunTimePurePrediction
from habitat.analysis.wave_scaling.metadata import MetadataManager
from habitat.analysis.wave_scaling.unified import unified_wave_scaling
from habitat.data import path_to_data
from habitat.utils import ms_to_ns, name_all_arguments

from habitat.analysis.mlp.mlp import RuntimePredictor

logger = logging.getLogger(__name__)

CONV2D_PARAMS = [
    'input', 'weight', 'bias', 'stride', 'padding', 'dilation', 'groups',
]

LINEAR_PARAMS = ['input', 'weight', 'bias']

BMM_PARAMS = ['input', 'mat2', 'out']

LSTM_PARAMS_NO_BATCH_SIZES = [
    'input',
    'hx',
    'flat_weights',
    'bias',
    'num_layers',
    'dropout',
    'training',
    'bidirectional',
    'batch_first',
]

LSTM_PARAMS = [
    'input',
    'batch_sizes',
    'hx',
    'flat_weights',
    'bias',
    'num_layers',
    'dropout',
    'training',
    'bidirectional',
]

MATMUL_PARAMS = ['input', 'other', 'out']


class Predictor:
    def __init__(
        self,
        kernel_metadata_file=None,
        wave_scaling_strategy=unified_wave_scaling
    ):
        self._kernel_metadata = MetadataManager(
            kernel_metadata_file if kernel_metadata_file is not None
            else path_to_data("kernels.sqlite")
        )
        self._wave_scaling_strategy = wave_scaling_strategy

        # Load MLP predictor from saved models
        self.linear_pred = RuntimePredictor(
            "linear", 8, 1024,
            path_to_data("linear/model.pth"),
        )
        self.lstm_pred = RuntimePredictor(
            "lstm", 8, 1024,
            path_to_data("lstm/model.pth"),
        )
        self.conv2d_pred = RuntimePredictor(
            "conv2d", 8, 1024,
            path_to_data("conv2d/model.pth"),
        )
        self.bmm_pred = RuntimePredictor(
            "bmm", 8, 1024,
            path_to_data("bmm/model.pth"),
        )


    def predict_operation(self, operation, dest_device):
        if operation.name not in SPECIAL_OPERATIONS:
            return PredictedOperation(
                operation,
                self._wave_scale(operation.forward, dest_device),
                (self._wave_scale(operation.backward, dest_device)
                 if operation.backward is not None else None),
                dest_device,
            )

        if operation.name == 'conv2d':
            return self._special_scale(operation, dest_device, self._conv2d_scale)
        elif operation.name == 'lstm':
            return self._special_scale(operation, dest_device, self._lstm_scale)
        elif operation.name == 'linear':
            return self._special_scale(operation, dest_device, self._linear_scale)
        elif operation.name == 'bmm':
            return self._special_scale(operation, dest_device, self._bmm_scale)

        logger.warn('Unhandled special operation: %s', operation.name)
        return PredictedOperation(
            operation,
            operation.forward,
            operation.backward,
            dest_device,
        )

    def _wave_scale(self, run_time, dest_device):
        run_time_ns = ms_to_ns(run_time.run_time_ms)
        total_ktime_ns = sum(map(lambda k: k.run_time_ns, run_time.kernels))
        overhead_ns = run_time_ns - total_ktime_ns

        predicted_kernels = list(map(
            lambda kernel: self._wave_scaling_strategy(
                kernel,
                run_time.device,
                dest_device,
                self._kernel_metadata,
            ),
            run_time.kernels,
        ))

        return RunTimePrediction(
            overhead_ns=0 if overhead_ns < 0 else overhead_ns,
            predicted_kernels=predicted_kernels,
            device=dest_device,
        )

    def _special_scale(self, operation, dest_device, scaler):
        predicted_ms = scaler(operation, dest_device)

        if predicted_ms < 0:
            logger.warn(
                'Operation %s predicted run time %.2f ms',
                operation.name,
                predicted_ms,
            )
            predicted_ms = 0.

        return PredictedOperation(
            operation,
            RunTimePurePrediction(predicted_ms, dest_device),
            None,
            dest_device,
        )

    def _conv2d_scale(self, operation, dest_device):
        # 1. Merge arguments (give them all names)
        merged = name_all_arguments(
            CONV2D_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        # 2. Construct arguments that the predictor expects
        arguments = dict(
            batch=merged['input'][0],
            image_size=merged['input'][2],
            in_channels=merged['input'][1],
            out_channels=merged['weight'][0],
            kernel_size=merged['weight'][2],
            stride=(
                merged['stride'][0]
                if isinstance(merged['stride'], tuple) else merged['stride']
            ),
            padding=(
                merged['padding'][0]
                if isinstance(merged['padding'], tuple) else merged['padding']
            ),
            bias=(1 if merged['bias'] is not None else 0),
        )

        # 3. Call model to make prediction
        arguments = [arguments[x] for x in self.conv2d_pred.model.features]

        pred_dest = self.conv2d_pred.predict(arguments, dest_device.name)
        pred_orig = self.conv2d_pred.predict(arguments, operation.device.name)

        return operation.run_time_ms * pred_dest / pred_orig

    def _linear_scale(self, operation, dest_device):
        merged = name_all_arguments(
            LINEAR_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        # The input to the linear function in PyTorch can contain an arbitrary
        # number of dimensions between the batch size and the input feature
        # dimensions.
        #
        #   e.g., The input can have size (32, 50, 512), where 32 is the batch
        #         size and 512 is the input feature dimension.
        #
        # This means that the effective batch size is the product of all the
        # dimensions before the input feature dimension (e.g., 32 * 50 = 1600).
        # We need to take this into account when making a prediction.
        effective_batch = functools.reduce(
            operator.mul,
            merged['input'][:-1],
        )

        arguments = dict(
            batch=effective_batch,
            in_features=merged['weight'][1],
            out_features=merged['weight'][0],
            bias=(1 if merged['bias'] is not None else 0)
        )

        arguments = [arguments[x] for x in self.linear_pred.model.features]

        pred_dest = self.linear_pred.predict(arguments, dest_device.name)
        pred_orig = self.linear_pred.predict(arguments, operation.device.name)

        return operation.run_time_ms * pred_dest / pred_orig

    def _bmm_scale(self, operation, dest_device):
        merged = name_all_arguments(
            BMM_PARAMS,
            operation.arguments.args,
            operation.arguments.kwargs,
        )

        arguments = dict(
            batch=merged['input'][0],
            left=merged['input'][1],
            middle=merged['input'][2],
            right=merged['mat2'][2],
        )
        arguments = [arguments[x] for x in self.bmm_pred.model.features]

        pred_dest = self.bmm_pred.predict(arguments, dest_device.name)
        pred_orig = self.bmm_pred.predict(arguments, operation.device.name)

        return operation.run_time_ms * pred_dest / pred_orig

    def _lstm_scale(self, operation, dest_device):
        # This is hacky, but unfortunately the only way to differentiate these
        # overloaded LSTM calls.
        has_batch_sizes = isinstance(operation.arguments.args[4], bool)

        if not has_batch_sizes:
            merged = name_all_arguments(
                LSTM_PARAMS_NO_BATCH_SIZES,
                operation.arguments.args,
                operation.arguments.kwargs,
            )
            arguments = dict(
                bias=(1 if merged['bias'] is not None else 0),
                bidirectional=(1 if merged['bidirectional'] else 0),
                batch=merged['input'][1],  # We require the batch to be in position 1
                seq_len=merged['input'][0],
                input_size=merged['input'][2],
                hidden_size=merged['hx'][0][2],
                num_layers=merged['num_layers'],
            )

        else:
            merged = name_all_arguments(
                LSTM_PARAMS,
                operation.arguments.args,
                operation.arguments.kwargs,
            )
            max_batch_size = max(operation.arguments.special['batch_sizes'])
            arguments = dict(
                bias=(1 if merged['bias'] is not None else 0),
                bidirectional=(1 if merged['bidirectional'] else 0),
                batch=max_batch_size,
                seq_len=merged['input'][0] // max_batch_size,
                input_size=merged['input'][1],
                hidden_size=merged['hx'][0][2],
                num_layers=merged['num_layers'],
            )

        arguments = [arguments[x] for x in self.lstm_pred.model.features]

        pred_dest = self.lstm_pred.predict(arguments, dest_device.name)
        pred_orig = self.lstm_pred.predict(arguments, operation.device.name)

        return operation.run_time_ms * pred_dest / pred_orig
