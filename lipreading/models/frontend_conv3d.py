import torch.nn as nn

def _to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

def conv3D(frontend_nout=64, kernel_size=7, pooling_size=3, act_type='relu'):
    frontend_relu = nn.PReLU(num_parameters=frontend_nout) if act_type == 'prelu' else nn.ReLU()
    return nn.Sequential(
        nn.Conv3d(1, frontend_nout, kernel_size=(5, kernel_size, kernel_size), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
        nn.BatchNorm3d(frontend_nout),
        frontend_relu,
        nn.MaxPool3d( kernel_size=(1, pooling_size, pooling_size), stride=(1, 2, 2), padding=(0, 1, 1))
        )

def multi_scale_conv3D(frontend_nout=32, kernel_size=5, stride=(1,2,2), pooling_size=2, act_type='relu'):
    frontend_relu = nn.PReLU(num_parameters=frontend_nout) if act_type == 'prelu' else nn.ReLU()
    branch1 = nn.Sequential(
        nn.Conv3d(1, frontend_nout, kernel_size=(7, kernel_size, kernel_size), stride=stride, padding=(3, 3, 3), bias=False),
        nn.BatchNorm3d(frontend_nout),
        frontend_relu,
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
    )
    branch2 = nn.Sequential(
        nn.Conv3d(1, frontend_nout, kernel_size=(5, kernel_size, kernel_size), stride=stride, padding=(2, 3, 3), bias=False),
        nn.BatchNorm3d(frontend_nout),
        frontend_relu,
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
    )
    branch3 = nn.Sequential(
        nn.Conv3d(1, frontend_nout, kernel_size=(3, kernel_size, kernel_size), stride=stride, padding=(1, 3, 3), bias=False),
        nn.BatchNorm3d(frontend_nout),
        frontend_relu,
        nn.MaxPool3d(kernel_size=(1, pooling_size, pooling_size), stride=stride, padding=(0, 1, 1))
    )

    return branch1, branch2, branch3
