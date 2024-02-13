import torch
import torch.nn as nn

from models.submodules import dense_submodules

class DilatedConv(nn.Module): #temporal layer
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, batch_norm):
        super(DilatedConv, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalConv(nn.Module): #fully interconnected tcn
    def __init__(self, num_channels, kernel_size, skip_conn, batch_norm, vital_signs, single_tcn=True):
        super(TemporalConv, self).__init__()
        self.skip_conn = skip_conn
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            if not single_tcn:
                vital_signs = 1
            in_channels = vital_signs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers.append(DilatedConv(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size-1)*dilation_size/2), batch_norm=batch_norm))
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.skip_conn:
            o = self.layers[0](x)
            s = o
            for i, layer in enumerate(self.layers[1:]):
                # We apply residuals on every other layer as in Imagenet
                # Skipping first layer
                o = layer(o)
                if (i % 2):
                    o = o + s
                    s = o
            return o
        else:
            return self.network(x)


class TemporalConvNet(nn.Module):
    def __init__(self, data_minutes, vital_signs, num_channels, kernel_size, skip_conn, batch_norm, mid_dense, out_mid_dense, single_tcn):
        super(TemporalConvNet, self).__init__()
        self.seq_lenght = data_minutes
        self.vital_signs = vital_signs
        self.single_tcn = single_tcn
        if self.single_tcn:
            self.tcn = TemporalConv(num_channels, kernel_size, skip_conn, batch_norm, vital_signs, single_tcn)
        else:
            self.tcn = nn.ModuleList([TemporalConv(num_channels, kernel_size, skip_conn, batch_norm) for i in range(vital_signs)])
        # self.mid_dense_layer = nn.ModuleList([dense_submodules.DenseBlock(num_channels[-1]*self.seq_lenght, out_mid_dense, dropout=0, batch_norm=batch_norm) for i in range(vital_signs)])
        self.mid_dense_layer = nn.ModuleList([nn.Linear(num_channels[-1]*self.seq_lenght, out_mid_dense) for i in range(vital_signs)])
        self.mid_dense = mid_dense
    def forward(self, x):
        if self.single_tcn:
            o = self.tcn(x.reshape(-1,self.vital_signs,self.seq_lenght))
        else:
            tcn_out = []
            for i, (tcn_i, dense_i) in enumerate(zip(self.tcn, self.mid_dense_layer)):
                o = tcn_i(x[:,i,:].reshape(-1,1,self.seq_lenght))
                if self.mid_dense:
                    o = dense_i(o.flatten(1))
                tcn_out.append(o)
            o = torch.cat(tcn_out, dim=1)
        if self.mid_dense:
            o = self.mid_dense_layer(o.flatten(1))
        return o