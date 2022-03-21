from torch import nn


def conv3d_block(d_in, d_out, kernel_size=3):
    return nn.Sequential(
        nn.Conv3d(d_in, d_out, kernel_size=kernel_size, bias=False),
        nn.BatchNorm3d(d_out),
        nn.ReLU()
    )
