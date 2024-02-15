import torch
import torch.nn as nn

from models.submodules import conv_submodules, dense_submodules

import numpy as np
 
def getPositionEncoding(seq_len: torch.Tensor, d: int, n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Generates positional encoding for the given sequence length.

    :param seq_len: Length of the sequence (tensor of shape (batch_size,))
    :param d: Dimensionality of the positional encoding
    :param n: Number of frequencies used for encoding
    :param dtype: Data type of the positional encoding (e.g., torch.float32)
    :param device: Device on which to create the positional encoding (e.g., CPU or GPU)
    :return: Positional encoding tensor of shape (batch_size, seq_len, d)
    """
    P = torch.zeros((d, seq_len.shape[0]), dtype=dtype, device=device, requires_grad=False)
    for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[2*i, :] = torch.sin(seq_len/denominator)
        P[2*i+1, :] = torch.cos(seq_len/denominator)
    return P.transpose(0, 1)


class TCN(nn.Module):
    def __init__(self, data_minutes, output_size, vital_signs, num_channels, device, dtype, dense_layers=[64], kernel_size=3, dropout=0.2, skip_conn=False, batch_norm=False, mid_dense=False, out_mid_dense=512, pos_enc=False, single_tcn=False):
        super(TCN, self).__init__()
        # All the parallel TCN networks
        self.device = device
        self.dtype = dtype
        self.pos_enc = pos_enc

        self.TemporalConvNet = conv_submodules.TemporalConvNet(data_minutes, vital_signs, num_channels, kernel_size, skip_conn, batch_norm, mid_dense, out_mid_dense, single_tcn)
        if mid_dense:
            self.num_inputs = out_mid_dense*vital_signs
        elif single_tcn:
            self.num_inputs = num_channels[-1]*data_minutes
        else:
            self.num_inputs = num_channels[-1]*data_minutes*vital_signs
        self.DenseNet = dense_submodules.DenseNet(dense_layers, output_size, self.num_inputs, dropout, skip_conn, batch_norm)

    def forward(self, data, ids):
        o = self.TemporalConvNet(data).flatten(1)
        if self.pos_enc:
            pos_enc_mat = getPositionEncoding(seq_len=ids.squeeze(), d=self.num_inputs, n=10000, dtype=self.dtype, device=self.device)
            o = o + pos_enc_mat

        o = self.DenseNet(o)
        return o