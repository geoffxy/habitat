

conv2d = [
    'bias',
    'batch',
    'image_size',
    'in_channels',
    'out_channels',
    'kernel_size',
    'stride',
    'padding',
]

bmm = [
    'batch',
    # (batch, left, middle) x (batch, middle, right)
    'left',
    'middle',
    'right',
]

lstm = [
    'bias', # 0 or 1, represents the bias flag
    'bidirectional', # 0 or 1, represents the bidirectional flag
    'batch',
    'seq_len',
    'input_size',
    'hidden_size',
    'num_layers',
]

linear = [
    'bias',
    'batch',
    'in_features',
    'out_features',
]

FEATURES = {
    'bmm': bmm,
    'conv2d': conv2d,
    'linear': linear,
    'lstm': lstm,
}
