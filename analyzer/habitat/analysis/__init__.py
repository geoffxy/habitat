from habitat.analysis.device import _Device

Device = _Device()

SPECIAL_OPERATIONS = {
    # Convolution
    'conv2d',

    # Matrix multiply operations
    'linear',
    'bmm',

    # Recurrent operations
    'lstm',
    'gru',
    'rnn_tanh',
    'rnn_relu',
}
