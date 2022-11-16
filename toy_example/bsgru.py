import logging
import math
import warnings
import torch
import torch.nn

from typing import Optional, Tuple, Union
from torch.nn.utils.rnn import PackedSequence
from torch.nn.functional import log_softmax, softmax, one_hot
from torch import sigmoid, tanh, Tensor

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.DEBUG
)

def to_numpy(v):
    return v.detach().cpu().numpy()


def repeat_interleave(v: Tensor, num_repeats: int, dim: int = -1):
    old_shape = list(v.shape)
    old_shape[dim] = -1
    expand_dims = [-1] * (v.ndim + 1)
    expand_dims[dim] = num_repeats
    return v.unsqueeze(dim).expand(*expand_dims).contiguous().view(*old_shape)


class BlockSparseGRUv1(torch.nn.Module): # (torch.nn.RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_blocks: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        beta: int = 10,
        device=None,
        dtype=None) -> None:

        super().__init__()
        # super().__init__('GRU',
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     bias=bias,
        #     batch_first=batch_first,
        #     bidirectional=bidirectional)
        factory_kwargs = {'device': device, 'dtype': dtype}
        if num_blocks < 1:
            raise ValueError(
                'num_blocks must be >= 1')
        if (hidden_size/num_blocks) != (hidden_size//num_blocks):
            raise ValueError(
                'hidden_size must be evenly divisible by num_blocks')
        if bidirectional:
            raise NotImplementedError('no support for bidirectional GRU')

        self.mode = 'GRU'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_layers = num_layers
        self.block_size = hidden_size // num_blocks
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = 0
        self.bidirectional = bidirectional
        self.proj_size = 0
        num_directions = 2 if bidirectional else 1
        self.beta = beta
        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                gate_size = 3 * hidden_size
                l_input_size = input_size
                if layer != 0:
                    l_input_size = hidden_size * num_directions
                w_ik = torch.nn.Parameter(
                    torch.empty((num_blocks, l_input_size), **factory_kwargs))
                w_hk = torch.nn.Parameter(
                    torch.empty((num_blocks, hidden_size), **factory_kwargs))
                w_ih = torch.nn.Parameter(
                    torch.empty((gate_size, l_input_size), **factory_kwargs))
                w_hh = torch.nn.Parameter(
                    torch.empty((gate_size, hidden_size), **factory_kwargs))
                b_ih = torch.nn.Parameter(
                    torch.empty(gate_size, **factory_kwargs))
                b_hh = torch.nn.Parameter(
                    torch.empty(gate_size, **factory_kwargs))
                b_ik = torch.nn.Parameter(
                    torch.empty(num_blocks, **factory_kwargs))
                b_hk = torch.nn.Parameter(
                    torch.empty(num_blocks, **factory_kwargs))
                if bias:
                    layer_params = (w_ik, w_hk, w_ih, w_hh, b_ik, b_hk, b_ih, b_hh)
                else:
                    layer_params = (w_ik, w_hk, w_ih, w_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = [
                    'weight_ik_l{}{}',
                    'weight_hk_l{}{}',
                    'weight_ih_l{}{}',
                    'weight_hh_l{}{}',
                ]
                if bias:
                    param_names += [
                        'bias_ik_l{}{}',
                        'bias_hk_l{}{}',
                        'bias_ih_l{}{}',
                        'bias_hh_l{}{}',
                    ]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                # self._flat_weights_names.extend(param_names)
                # self._all_weights.append(param_names)

        # self._flat_weights = [
        #     (lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn)
        #     for wn in self._flat_weights_names]
        # self.flatten_parameters()
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
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

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.proj_size)
        else:
            expected_hidden_size = (self.num_layers * num_directions,
                                    mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
                          msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]):
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    def flatten_parameters(self) -> None:
            """Resets parameter data pointer so that they can use faster code paths.

            Right now, this works only if the module is on the GPU and cuDNN is enabled.
            Otherwise, it's a no-op.
            """
            # Short-circuits if _flat_weights is only partially instantiated
            if len(self._flat_weights) != len(self._flat_weights_names):
                return

            for w in self._flat_weights:
                if not isinstance(w, Tensor):
                    return
            # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
            # or the tensors in _flat_weights are of different dtypes

            first_fw = self._flat_weights[0]
            dtype = first_fw.dtype
            for fw in self._flat_weights:
                if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                        not fw.data.is_cuda or
                        not torch.backends.cudnn.is_acceptable(fw.data)):
                    return

            # If any parameters alias, we fall back to the slower, copying code path. This is
            # a sufficient check, because overlapping parameter buffers that don't completely
            # alias would break the assumptions of the uniqueness check in
            # Module.named_parameters().
            unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
            if len(unique_data_ptrs) != len(self._flat_weights):
                return

            with torch.cuda.device_of(first_fw):
                import torch.backends.cudnn.rnn as rnn

                # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
                # an inplace operation on self._flat_weights
                with torch.no_grad():
                    if torch._use_cudnn_rnn_flatten_weight():
                        num_weights = 4 if self.bias else 2
                        if self.proj_size > 0:
                            num_weights += 1
                        torch._cudnn_rnn_flatten_weight(
                            self._flat_weights, num_weights,
                            self.input_size, rnn.get_cudnn_mode(self.mode),
                            self.hidden_size, self.proj_size, self.num_layers,
                            self.batch_first, bool(self.bidirectional))

    def _apply(self, fn):
        ret = super()._apply(fn)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        # Flattens params (on CUDA)
        # self.flatten_parameters()

        return ret

    def forward(
        self,
        inputs: Union[torch.Tensor, PackedSequence],
        h_0: Optional[torch.Tensor] = None,
        k_0: Optional[torch.Tensor] = None,
        return_blocks: bool = False):


        # figure out batch sizes and indices
        orig_input = inputs
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            is_batched = inputs.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                inputs = inputs.unsqueeze(batch_dim)
                if h_0 is not None:
                    if h_0.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, h_0 should also be 2-D "
                            f"but got {h_0.dim()}-D tensor")
                    h_0 = h_0.unsqueeze(1)
            else:
                if h_0 is not None and h_0.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, h_0 should also be 3-D "
                        f"but got {h_0.dim()}-D tensor")
            batch_sizes = None
            max_batch_size = inputs.size(batch_dim)
            sorted_indices = None
            unsorted_indices = None


        # ensurses that the hidden state matches the input sequence
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = self.permute_hidden(h_0, sorted_indices)

        if k_0 is None:
            k_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.num_blocks,
                             dtype=inputs.dtype, device=inputs.device)

        # run PyTorch-provided checks
        self.check_forward_args(inputs, h_0, batch_sizes)

        # loop over layers and over time and over blocks
        hiddens = []
        layer_blocks = []
        for l in range(self.num_layers):

            N = int(self.input_size if l == 0 else self.hidden_size)
            M = self.num_blocks
            L = self.block_size

            W_z, W_r, W_h = getattr(
                self, f"weight_ih_l{l}").reshape(3, M, L, N)
            W_k = getattr(self, f"weight_ik_l{l}")
            U_z, U_r, U_h = getattr(
                self, f"weight_hh_l{l}").reshape(3, M, M, L, L)
            U_k = getattr(self, f"weight_hk_l{l}")
            b_z, b_r, b_h = torch.zeros(3, M, L)
            b_k = torch.zeros(M)


            if self.bias:
                b_z, b_r, b_h = getattr(self, f"bias_ih_l{l}").reshape(3, M, L)
                b_k = getattr(self, f"bias_ik_l{l}")

            h_prev = h_0[l, :, :]
            k_prev = k_0[l, :, :]

            outputs = []
            blocks = []

            for t in range(inputs.size(1)):

                x = inputs[:, t, :]

                k_next = x.mm(W_k.T) + h_prev.mm(U_k.T) + b_k
                k_next = softmax(self.beta * k_next, dim=-1)

                # split up h_prev and gates into blocks
                h_prev = h_prev.reshape(max_batch_size, M, L)

                # if model is in eval-mode, swap out the softmax for hardmax
                i_star = torch.argmax(k_next, dim=-1)
                j_star = torch.argmax(k_prev, dim=-1)
                if not self.training:
                    k_next = one_hot(i_star, num_classes=M)
                    k_prev = one_hot(i_star, num_classes=M)

                h_next = []

                for i in range(self.num_blocks):

                    k_i = k_next[:,i].unsqueeze(1)

                    # compute update gate
                    z_i = sigmoid(
                        k_i * x.mm(W_z[i].T)
                        + sum([
                            k_i * k_prev[:,j].unsqueeze(1)
                            * h_prev[:,j].mm(U_z[i,j].T)
                            for j in range(M)
                        ])
                        + b_z[i])

                    # compute reset gate
                    r_i = sigmoid(
                        k_i * x.mm(W_r[i].T)
                        + sum([
                            k_i * k_prev[:,j].unsqueeze(1)
                            * h_prev[:,j].mm(U_r[i,j].T)
                            for j in range(M)
                        ])
                        + b_r[i])

                    # compute new gate
                    n_i = tanh(
                        k_i * x.mm(W_h[i].T)
                        + sum([
                            k_i * k_prev[:,j].unsqueeze(1)
                            * (r_i * h_prev[:,j]).mm(U_h[i,j].T)
                            for j in range(M)
                        ])
                        + b_r[i])

                    # compute output
                    h_i = z_i * (k_i * n_i) + (1 - z_i) * sum([
                        k_prev[:,j].unsqueeze(1) * h_prev[:,j]
                        for j in range(M)
                    ])
                    h_next.append(h_i)

                # stack the block outputs together
                h_next = torch.cat(h_next, dim=-1)

                # feed this time step's output as the next time step's input
                outputs.append(h_next)
                blocks.append(i_star)
                h_prev = h_next
                k_prev = k_next

            # feed this layer's output as the next layer's input
            inputs = torch.stack(outputs, 1)

            # keep track of the final hidden state from this layer
            hiddens.append(h_next)
            layer_blocks.append(torch.stack(blocks, 1))

        # use the same return signature as PyTorch's GRU
        output, hidden = inputs, torch.stack(hiddens)

        # revert indices if using packed sequences
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = hidden.squeeze(1)
        hidden = self.permute_hidden(hidden, unsorted_indices)

        if return_blocks:
            return output, hidden, to_numpy(torch.stack(layer_blocks))
        return output, hidden


class BlockSparseGRUv2(torch.nn.GRU): # (torch.nn.RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_blocks: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        beta: int = 10,
        device=None,
        dtype=None) -> None:

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        if num_blocks < 1:
            raise ValueError(
                'num_blocks must be >= 1')
        if (hidden_size/num_blocks) != (hidden_size//num_blocks):
            raise ValueError(
                'hidden_size must be evenly divisible by num_blocks')
        if bidirectional:
            raise NotImplementedError('no support for bidirectional GRU')
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.beta = beta
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                l_input_size = input_size
                if layer != 0:
                    l_input_size = hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''
                setattr(self, f'weight_ik_l{layer}{suffix}', torch.nn.Parameter(
                    torch.empty((num_blocks, l_input_size), **factory_kwargs)))
                setattr(self, f'weight_hk_l{layer}{suffix}', torch.nn.Parameter(
                    torch.empty((num_blocks, hidden_size), **factory_kwargs)))
                if bias:
                    setattr(self, f'bias_ik_l{layer}{suffix}', torch.nn.Parameter(
                        torch.empty(num_blocks, **factory_kwargs)))
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
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
        inputs: Union[torch.Tensor, PackedSequence],
        h_0: Optional[torch.Tensor] = None,
        k_0: Optional[torch.Tensor] = None,
        return_blocks: bool = False):

        # figure out batch sizes and indices
        orig_input = inputs
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            is_batched = inputs.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                inputs = inputs.unsqueeze(batch_dim)
                if h_0 is not None:
                    if h_0.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, h_0 should also be 2-D "
                            f"but got {h_0.dim()}-D tensor")
                    h_0 = h_0.unsqueeze(1)
            else:
                if h_0 is not None and h_0.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, h_0 should also be 3-D "
                        f"but got {h_0.dim()}-D tensor")
            batch_sizes = None
            max_batch_size = inputs.size(batch_dim)
            sorted_indices = None
            unsorted_indices = None


        # ensures that the hidden state matches the input sequence
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = self.permute_hidden(h_0, sorted_indices)

        if k_0 is None:
            k_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.num_blocks,
                             dtype=inputs.dtype, device=inputs.device)

        # run PyTorch-provided checks
        self.check_forward_args(inputs, h_0, batch_sizes)

        # loop over layers and over time and over blocks
        hiddens = []
        layer_blocks = []
        for l in range(self.num_layers):

            N = int(self.input_size if l == 0 else self.hidden_size)
            M = self.num_blocks
            L = self.block_size

            W_z, W_r, W_h = getattr(
                self, f"weight_ih_l{l}").reshape(3, M, L, N)
            W_k = getattr(self, f"weight_ik_l{l}")
            U_z, U_r, U_h = getattr(
                self, f"weight_hh_l{l}").reshape(3, M, M, L, L)
            U_k = getattr(self, f"weight_hk_l{l}")
            b_z, b_r, b_h = torch.zeros(3, M, L)
            b_k = torch.zeros(M)


            if self.bias:
                b_z, b_r, b_h = getattr(self, f"bias_ih_l{l}").reshape(3, M, L)
                b_k = getattr(self, f"bias_ik_l{l}")

            h_prev = h_0[l, :, :]
            k_prev = k_0[l, :, :]

            outputs = []
            blocks = []

            for t in range(inputs.size(1)):

                x = inputs[:, t, :]

                k_next = x.mm(W_k.T) + h_prev.mm(U_k.T) + b_k
                k_next = softmax(self.beta * k_next, dim=-1)

                # split up h_prev and gates into blocks
                h_prev = h_prev.reshape(max_batch_size, M, L)

                # if model is in eval-mode, swap out the softmax for hardmax
                i_star = torch.argmax(k_next, dim=-1)
                j_star = torch.argmax(k_prev, dim=-1)
                if not self.training:
                    k_next = one_hot(i_star, num_classes=M)
                    k_prev = one_hot(i_star, num_classes=M)

                h_next = []

                for i in range(self.num_blocks):

                    # logging.debug(f'layer {l:<3d} step {t:<3d} block {i:<3d}')

                    k_i = k_next[:,i].unsqueeze(1)

                    # compute update gate
                    z_i = sigmoid(
                        k_i * x.mm(W_z[i].T)
                        + sum([
                            k_i * k_prev[:,j].unsqueeze(1)
                            * h_prev[:,j].mm(U_z[i,j].T)
                            for j in range(M)
                        ])
                        + b_z[i])

                    # compute reset gate
                    r_i = sigmoid(
                        k_i * x.mm(W_r[i].T)
                        + sum([
                            k_i * k_prev[:,j].unsqueeze(1)
                            * h_prev[:,j].mm(U_r[i,j].T)
                            for j in range(M)
                        ])
                        + b_r[i])

                    # compute new gate
                    n_i = tanh(
                        k_i * x.mm(W_h[i].T)
                        + sum([
                            k_i * k_prev[:,j].unsqueeze(1)
                            * (r_i * h_prev[:,j]).mm(U_h[i,j].T)
                            for j in range(M)
                        ])
                        + b_r[i])

                    # compute output
                    h_i = z_i * (k_i * n_i) + (1 - z_i) * sum([
                        k_prev[:,j].unsqueeze(1) * h_prev[:,j]
                        for j in range(M)
                    ])
                    h_next.append(h_i)

                # stack the block outputs together
                h_next = torch.cat(h_next, dim=-1)

                # feed this time step's output as the next time step's input
                outputs.append(h_next)
                blocks.append(i_star)
                h_prev = h_next
                k_prev = k_next

            # feed this layer's output as the next layer's input
            inputs = torch.stack(outputs, 1)

            # keep track of the final hidden state from this layer
            hiddens.append(h_next)
            layer_blocks.append(torch.stack(blocks, 1))

        # use the same return signature as PyTorch's GRU
        output, hidden = inputs, torch.stack(hiddens)

        if return_blocks:
            return output, hidden, to_numpy(torch.stack(layer_blocks))
        return output, hidden


class BlockSparseGRU(torch.nn.GRU): # (torch.nn.RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_blocks: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        beta: float = 10.,
        device=None,
        dtype=None) -> None:

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        if num_blocks < 1:
            raise ValueError(
                'num_blocks must be >= 1')
        if (hidden_size/num_blocks) != (hidden_size//num_blocks):
            raise ValueError(
                'hidden_size must be evenly divisible by num_blocks')
        if bidirectional:
            raise NotImplementedError('no support for bidirectional GRU')
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.beta = beta
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                l_input_size = input_size
                if layer != 0:
                    l_input_size = hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''
                setattr(self, f'weight_ik_l{layer}{suffix}', torch.nn.Parameter(
                    torch.empty((num_blocks, l_input_size), **factory_kwargs)))
                setattr(self, f'weight_hk_l{layer}{suffix}', torch.nn.Parameter(
                    torch.empty((num_blocks, hidden_size), **factory_kwargs)))
                if bias:
                    setattr(self, f'bias_ik_l{layer}{suffix}', torch.nn.Parameter(
                        torch.empty(num_blocks, **factory_kwargs)))
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
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
        inputs: Union[torch.Tensor, PackedSequence],
        h_0: Optional[torch.Tensor] = None,
        k_0: Optional[torch.Tensor] = None,
        return_blocks: bool = False):

        # figure out batch sizes and indices
        orig_input = inputs
        if isinstance(orig_input, PackedSequence):
            inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            is_batched = inputs.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                inputs = inputs.unsqueeze(batch_dim)
                if h_0 is not None:
                    if h_0.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, h_0 should also be 2-D "
                            f"but got {h_0.dim()}-D tensor")
                    h_0 = h_0.unsqueeze(1)
            else:
                if h_0 is not None and h_0.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, h_0 should also be 3-D "
                        f"but got {h_0.dim()}-D tensor")
            batch_sizes = None
            max_batch_size = inputs.size(batch_dim)
            sorted_indices = None
            unsorted_indices = None


        # ensures that the hidden state matches the input sequence
        num_directions = 2 if self.bidirectional else 1
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=inputs.dtype, device=inputs.device)
        else:
            h_0 = self.permute_hidden(h_0, sorted_indices)

        if k_0 is None:
            k_0 = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.num_blocks,
                             dtype=inputs.dtype, device=inputs.device)

        # run PyTorch-provided checks
        self.check_forward_args(inputs, h_0, batch_sizes)

        # loop over layers and over time and over blocks
        hiddens = []
        layer_blocks = []
        for l in range(self.num_layers):

            N = int(self.input_size if l == 0 else self.hidden_size)
            M = self.num_blocks
            L = self.block_size
            H = self.hidden_size

            # retrieve model parameters for this layer
            W_z, W_r, W_h = getattr(self, f"weight_ih_l{l}").view(3, H, N)
            U_z, U_r, U_h = getattr(self, f"weight_hh_l{l}").view(3, 1, H, H)
            W_k = getattr(self, f"weight_ik_l{l}")
            U_k = getattr(self, f"weight_hk_l{l}")
            if self.bias:
                b_z, b_r, b_h = getattr(self, f"bias_ih_l{l}").view(3, H)
                b_k = getattr(self, f"bias_ik_l{l}")
            else:
                b_z, b_r, b_h = torch.zeros(3, M, L)
                b_k = torch.zeros(M)


            h_prev = h_0[l, :, :]
            k_prev = k_0[l, :, :]

            outputs = []
            blocks = []

            for t in range(inputs.size(1)):

                x = inputs[:, t, :]

                # predict the current active block for this time step
                k_next = x.mm(W_k.T) + h_prev.mm(U_k.T) + b_k
                k_next = softmax(self.beta * k_next, dim=-1)

                # if the model is in eval mode, replace softmax for hardmax
                i_star = torch.argmax(k_next, dim=-1)
                j_star = torch.argmax(k_prev, dim=-1)
                if not self.training:
                    k_next = one_hot(i_star, num_classes=M).float()
                    k_prev = one_hot(i_star, num_classes=M).float()

                # use repeat-interleave so that k_next and k_prev can be
                # element-wise multiplied with the hidden-to-hidden weights
                k_n = repeat_interleave(k_next, self.block_size, dim=-1)
                k_p = repeat_interleave(k_prev, self.block_size, dim=-1)
                k_np = torch.matmul(k_n.unsqueeze(2), k_p.unsqueeze(1))
                h_p = h_prev.unsqueeze(1)

                # compute update gate
                z_gate = sigmoid(
                    k_n * x.mm(W_z.T)
                    + h_p.matmul(k_np*U_z).sum(1)
                    + b_z
                )

                # compute reset gate
                r_gate = sigmoid(
                    k_n * x.mm(W_r.T)
                    + h_p.matmul(k_np*U_r).sum(1)
                    + b_r
                )

                # compute new gate
                h_gate = tanh(
                    k_n * x.mm(W_h.T)
                    + (r_gate * (h_p.matmul(k_np*U_h).sum(1)))
                    + b_h
                )

                # compute output
                h_next = (
                    (k_n * (1 - z_gate) * h_gate)
                    + (z_gate * h_p.matmul(k_np).sum(1))
                )

                # use the output from this time step as the input to the next
                outputs.append(h_next)
                blocks.append(i_star)
                h_prev = h_next
                k_prev = k_next

            # use this layer's outputs as the next layer's inputs
            inputs = torch.stack(outputs, 1)

            # keep track of the final hidden state from this layer
            hiddens.append(h_next)
            layer_blocks.append(torch.stack(blocks, 1))

        # use the same return signature as PyTorch's GRU
        output, hidden = inputs, torch.stack(hiddens)

        if return_blocks:
            return output, hidden, to_numpy(torch.stack(layer_blocks))
        return output, hidden
