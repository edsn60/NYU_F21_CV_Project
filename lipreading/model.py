import torch
import torch.nn as nn
import math
import numpy as np
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet
from lipreading.models.senet import SEBasicBlock
import lipreading.models.biLSTM as myLSTM
import lipreading.models.vgg_baseline as baseline
import lipreading.models.frontend_conv3d as frontend


# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']
        self.num_kernels = len( self.kernel_sizes )

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        xtrans = x.transpose(1, 2)
        out = self.mb_ms_tcn(xtrans)
        out = self.consensus_func( out, lengths, B )
        return self.tcn_output(out)


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options, relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)


class myLipreading(nn.Module):
    def __init__( self, hidden_dim=256, backbone_type='vgg', backend_type='mstcn', num_classes=50,
                  relu_type='relu', tcn_options={}, ms3d=False, extract_feats=False):
        super(myLipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.backend_type = backend_type
        self.ms3d = ms3d

        if self.backbone_type == 'resnet':
            self.frontend_nout = 64 if not ms3d else 96
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        elif self.backbone_type == 'vgg':
            self.frontend_nout = 64 if not ms3d else 96
            self.backend_out = 512
            # self.trunk = baseline.vgg11(pretrained=True)
            self.trunk = baseline.VGG16(outplain=512)
        elif self.backbone_type == 'seresnet':
            self.frontend_nout = 64 if not ms3d else 96
            self.backend_out = 512
            self.trunk = ResNet(SEBasicBlock, [2, 2, 2, 2], inplanes=self.frontend_nout, num_classes=num_classes, relu_type=relu_type)

        if not ms3d:
            print('3d frontend')
            self.frontend3D = frontend.conv3D(frontend_nout=64, kernel_size=7, pooling_size=3, act_type=relu_type)
        else:
            print('ms3d frontend')
            self.fe_b1, self.fe_b2, self.fe_b3 = frontend.multi_scale_conv3D(frontend_nout=32, kernel_size=5, stride=(1,2,2), pooling_size=2, act_type='relu')


        if backend_type == 'bilstm':
            print("load biLSTM")
            self.backend = myLSTM.LSTM(num_classes, input_size=512, hidden_size=2 * 512, num_layers=2, bidirect=True)
        elif backend_type == 'lstm':
            self.backend = myLSTM.LSTM(num_classes, input_size=512, hidden_size=512, num_layers=2, bidirect=False)
        elif backend_type == 'bgrn':
            self.backend = myLSTM.LSTM(num_classes, input_size=512, hidden_size=512, num_layers=2, bidirect=False)
        elif backend_type == 'fc':
            self.backend = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*72, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes))
        else:
            print('load tcn')
            tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
            self.tcn = tcn_class( input_size=self.backend_out,
                                  num_channels=[hidden_dim*len(tcn_options['kernel_size'])*tcn_options['width_mult']]*tcn_options['num_layers'],
                                  num_classes=num_classes,
                                  tcn_options=tcn_options,
                                  dropout=tcn_options['dropout'],
                                  relu_type=relu_type,
                                  dwpw=tcn_options['dwpw'],
                                )
        # -- initialize
        self._initialize_weights_randomly()


    def forward(self, x, lengths):
        # print(x.shape)
        B, C, T, H, W = x.size()
        if not self.ms3d:
            x = self.frontend3D(x)
        else:
            x1 = self.fe_b1(x)
            x2 = self.fe_b2(x)
            x3 = self.fe_b3(x)
            x = torch.concat((x1, x2, x3), axis=1)
        Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor(x)
        # print(x.shape)
        x = self.trunk(x)
        if self.backbone_type == 'shufflenet':
            x = x.view(-1, self.stage_out_channels)
        x = x.view(B, Tnew, x.size(1))

        # print("cnn ended:")
        # print(x.shape)
        if self.backend_type == 'bilstm' or self.backend_type == 'lstm' or self.backend_type == 'bgrn':
            x = self.backend(x)
        elif self.backend_type == 'fc':
            x = self.backend(x)
        else:
            x = x if self.extract_feats else self.tcn(x, lengths, B)
        # print(x.shape)
        return x


    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))