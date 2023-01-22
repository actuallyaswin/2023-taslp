import math
import warnings
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import one_hot

from typing import Optional, Tuple


class BSGRUCell(nn.Module):
    """This version uses manual weights+biases plus cross product selection.
    """
    
    def __init__(self, input_size, hidden_size, num_blocks, bias=True):
        super(BSGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        block_size = int(hidden_size / num_blocks)
        if (block_size * num_blocks) != hidden_size:
            raise ValueError("hidden_size must be divisible by num_blocks")
        self.block_size = block_size
        self.weight_ik = Parameter(torch.randn(input_size, num_blocks))
        self.weight_hk = Parameter(torch.randn(hidden_size, num_blocks))
        self.weight_ih = Parameter(torch.randn(
            1, num_blocks, input_size, 3 * block_size))
        self.weight_hh = Parameter(torch.randn(
            1, num_blocks, num_blocks, block_size, 3 * block_size))
        if bias:
            self.bias_ik = Parameter(torch.randn(num_blocks))
            self.bias_ih = Parameter(torch.randn(3 * hidden_size))
            self.bias_hh = Parameter(torch.randn(3 * hidden_size))
        else:
            self.bias_ik = torch.zeros(num_blocks)
            self.bias_ih = torch.zeros(3 * hidden_size)
            self.bias_hh = torch.zeros(3 * hidden_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input, hx, kx):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        # "x" and "y" variable names refer to previous and current timesteps

        # predict the current active block for this time step
        ky = (torch.mm(input, self.weight_ik) +
              torch.mm(hx, self.weight_hk) +
              self.bias_ik)
        if self.training:
            beta = 10.
            ky = self.softmax(beta*ky)
        else:
            ky = one_hot(ky.argmax(-1), num_classes=self.num_blocks).float()

        # use the current block predictions to sparsely activate
        # the input-to-hidden weights (W-matrix)
        mat_W = torch.mul(ky.unsqueeze(-1).unsqueeze(-1), self.weight_ih)
        mat_W = mat_W.view(-1, 3 * self.hidden_size, self.input_size)
        mat_W = torch.transpose(mat_W, 1, 2)

        # use the previous + current block predictions to sparsely activate
        # the hidden-to-hidden weights (U-matrix)
        kk = torch.bmm(kx.unsqueeze(2), ky.unsqueeze(1))
        mat_U = torch.mul(kk.unsqueeze(-1).unsqueeze(-1), self.weight_hh)
        mat_U = mat_U.view(-1, 3 * self.hidden_size, self.hidden_size)
        mat_U = torch.transpose(mat_U, 1, 2)
        
        # repeat
        _kx = kx.repeat_interleave(self.block_size, dim=-1)
        _ky = ky.repeat_interleave(self.block_size, dim=-1)

        # compute all the gate values
        i_gates = torch.bmm(input.unsqueeze(1), mat_W) + self.bias_ih
        h_gates = torch.bmm(hx.unsqueeze(1), mat_U) + self.bias_hh
        i_z, i_r, i_n = i_gates.squeeze(1).chunk(3, -1)
        h_z, h_r, h_n = h_gates.squeeze(1).chunk(3, -1)
        
        updategate = torch.sigmoid(i_z + h_z)
        resetgate = torch.sigmoid(i_r + h_r)
        newgate = torch.tanh(i_n + h_n * resetgate)
        hy = newgate*_ky + updategate * (hx*_kx - newgate*_ky)

        return hy, ky
    

class BSGRUv1(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_blocks: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        dropout: float = 0.,
        device=None,
        dtype=None) -> None:

        super(BSGRUv1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        for l in range(num_layers):
            l_input_size = input_size if l==0 else hidden_size
            self.add_module(
                "layer_{}".format(l), 
                BSGRUCell(l_input_size, hidden_size, num_blocks, bias=bias))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        stdv *= math.sqrt(6)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.num_blocks > 0:
            s += ', num_blocks={num_blocks}'
        return s.format(**self.__dict__)

    def forward(
        self,
        inputs,
        h0: Optional[Tensor] = None,
        k0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        
        # put seq_len as the first axes
        if self.batch_first:
            inputs = torch.transpose(inputs, 0, 1)
        seq_len, batch_size, input_size = inputs.shape
        
        num_directions = 2 if self.bidirectional else 1
        if h0 is None:
            h0 = torch.zeros(self.num_layers * num_directions,
                             batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)
        if k0 is None:
            k0 = torch.zeros(self.num_layers * num_directions,
                             batch_size, self.num_blocks,
                             dtype=inputs.dtype, device=inputs.device)
            k0[:, :, 0] = 1

        in_tensor, out_hid, out_blk = inputs, [], []

        for l in range(self.num_layers):
            
            in_seq = in_tensor.unbind(0)
            lay_hid, lay_blk = [], []
            hx, kx = h0[l], k0[l]
        
            for i in range(len(in_seq)):
                layer = self.get_submodule("layer_{}".format(l))
                hx, kx = layer(in_seq[i], hx, kx)
                lay_hid += [hx]
                lay_blk += [kx]
            
            in_tensor = torch.stack(lay_hid)
            out_hid.append(in_tensor)
            out_blk.append(torch.stack(lay_blk))
        
        out_hid, out_blk = torch.stack(out_hid), torch.stack(out_blk)
        
        if self.batch_first:
            in_tensor = torch.transpose(in_tensor, 0, 1)
            out_hid = torch.transpose(out_hid, 1, 2)
            out_blk = torch.transpose(out_blk, 1, 2)
        
        return in_tensor, out_hid, out_blk


class BSGRUv2(nn.Module):
    """This version uses nn.Linear for weights+biases plus repeat_interleave.
    Also there's no separate cell, so this is only a single layer.
    """

    def __init__(self, input_size, hidden_size, num_blocks):
        super(BSGRUv2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        if hidden_size % num_blocks:
            raise ValueError("hidden_size must be divisible by num_blocks")
        self.block_size = hidden_size//num_blocks
        self.K = nn.Linear(input_size + hidden_size, num_blocks)
        self.W = nn.Linear(input_size, 3 * hidden_size)
        self.U = nn.Linear(hidden_size, 3 * hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        stdv *= math.sqrt(6)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.num_blocks > 0:
            s += ', num_blocks={num_blocks}'
        return s.format(**self.__dict__)

    def forward(self, in_tensor):

        batch_size = in_tensor.size(1)
        h0 = torch.zeros(batch_size, self.hidden_size)
        k0 = torch.zeros(batch_size, self.num_blocks)

        hy, ky = h0, k0
        in_seq = in_tensor.unbind(0)
        out_seq: List[torch.Tensor] = []
        out_blk: List[torch.Tensor] = []

        for i in range(len(in_seq)):

            # gather previous hidden and regime states
            x, hx, kx = in_seq[i], hy, ky

            # predict current regime state
            tau = 5.  # softmax temperature
            ky = self.softmax(tau * self.K(
                torch.cat([x, hx], dim=1)))

            # compute outer product between current and prev regimes
            # kk = torch.bmm(kx.unsqueeze(2), ky.unsqueeze(1))
            _kx = kx.repeat_interleave(self.block_size, dim=-1)
            _ky = ky.repeat_interleave(self.block_size, dim=-1).repeat([1, 3])

            # apply past and current regime estimates to sparsely
            # select model weights
            i_z, i_r, i_n = self.W(x).mul(_ky).chunk(3, -1)
            h_z, h_r, h_n = self.U(hx.mul(_kx)).mul(_ky).chunk(3, -1)

            # compute the gates as usual
            updategate = torch.sigmoid(i_z + h_z)
            resetgate = torch.sigmoid(i_r + h_r)
            newgate = torch.tanh(i_n + h_n * resetgate)
            hy = newgate + updategate * (hx - newgate)
            out_seq += [hy]
            out_blk += [ky]

        return torch.stack(out_seq), torch.stack(out_blk)

