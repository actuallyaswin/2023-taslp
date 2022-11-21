import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import warnings
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.nn.functional import relu, tanh
from torch.utils.tensorboard import SummaryWriter
from bsgru import BlockSparseGRU
from data import DatasetSineWave


# prepare choices of beta annealing function
min_beta = 1.
max_beta = 10.
beta_functions = [
    lambda i: max_beta - (max_beta-min_beta)/np.exp(0.01*i),
    lambda i: max_beta - (max_beta-min_beta)/np.exp(0.1*i),
    lambda i: max_beta,
    lambda i: min_beta + (max_beta-min_beta)/np.exp(0.1*i),
    lambda i: min_beta + (max_beta-min_beta)/np.exp(0.01*i),
]


def to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def to_clone(t: torch.Tensor):
    return t.detach().clone()


class Network(torch.nn.Module):
    
    def __init__(self, model_name: str, input_size: int, hidden_size: int,
                 num_layers: int, num_blocks: int = 1):
        super().__init__()
        self.model_name = model_name
        self.num_blocks = num_blocks
        if model_name == 'gru':
            self.gru = torch.nn.GRU(
                input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.gru = BlockSparseGRU(
                input_size, hidden_size, num_layers, num_blocks, beta=1)
        self.dnn = torch.nn.Linear(hidden_size, input_size)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        """Processes the input sequence with GRU and Linear layer.

        Args:
            x: input sequence

        Returns:
            Output -> shape=(batch_size, time_steps, output_size)
            Hidden -> shape=(num_layers, batch_size, hidden_size)
            None or Blocks -> shape=(num_layers, batch_size, time_steps)
        """
        if self.model_name == 'gru':
            o, h = self.gru(x)
            m = None
        else:
            o, h, m = self.gru(x)
            h = to_clone(o)
        o = self.tanh(self.dnn(o))
        return o, h, m


def parse_arguments(writer: SummaryWriter) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', choices=['gru', 'bsgru'])
    parser.add_argument('beta_func', type=int,
                        choices=range(len(beta_functions)))
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('num_layers', type=int)
    parser.add_argument('frequencies', type=int, nargs='+')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_rate', type=int, default=20)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--n_fft', type=int)
    parser.add_argument('--hop_length', type=int)
    parser.add_argument('--num_samples_pred', type=int, default=1)
    parser.add_argument('--seed', default=0)
    args = parser.parse_args()
    if args.seed is not None:
        args.seed = int(args.seed)
    args.input_size = 1
    if args.n_fft is not None:
        args.input_size = int(args.n_fft / 2 + 1)
    args.num_blocks = len(args.frequencies)

    # write parsed arguments to tensorboard
    for k, v in vars(args).items():
        if v is not None:
            writer.add_text(k, str(v), 0)

    return args


def check_gradients(module) -> None:
    for name, param in module.named_parameters():
        print(
            name, 
            param.dtype,
            param.grad.data.norm(2).item()
            if param.grad is not None else None
        )
    return


def main():

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO
    )
    writer = SummaryWriter()
    args = parse_arguments(writer)
    torch.manual_seed(args.seed)
    net = Network(args.model_name, args.input_size, args.hidden_size,
                  args.num_layers, args.num_blocks)
    if torch.cuda.is_available():
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    dataset = DatasetSineWave(
        frequencies=args.frequencies,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_samples_pred=args.num_samples_pred,
        seed=args.seed
    )

    
    logging.info(f'Network: {net}')
    logging.info(f'Optimizer: Adam(lr={args.learning_rate:.1e})')
    logging.info(f'Criterion: {criterion}')
    logging.info(f'Dataset: {dataset}')

    # pull out the validation set
    dataset = iter(dataset)
    x_val, y_val, m_val = next(dataset)
    x_val, y_val = to_clone(x_val), to_clone(y_val)


    # model training loop
    best_loss = 1e8
    for n_iter in range(100):

        # validate model
        net.eval()
        with torch.no_grad():
            y_pred, h, m_pred = net(x_val)
            loss_vl = float(criterion(y_pred, y_val))
            writer.add_scalar('loss/validation', loss_vl, n_iter)
            if args.model_name == 'bsgru':
                m_pred = to_numpy(m_pred)
                m_true = to_numpy(m_val)
                y_pred = to_numpy(y_pred)
                y_true = to_numpy(y_val)
                # split the BSGRU hidden state outputs into blocks
                # and compute the L1-norm to represent activity
                h = h.detach().cpu().numpy()
                h = np.stack(np.split(h, args.num_blocks, axis=-1), axis=-1)
                h = np.linalg.norm(h, ord=1, axis=-1)

                for i in range(min(8, args.batch_size)):
                
                    fig, ax = plt.subplots(dpi=80, facecolor='#ccc')
                    ax.set_ylim([-1, 1])
                    ax.plot(y_pred[i], label='pred')
                    ax.plot(y_true[i], label='true', zorder=-1)
                    ax.legend()
                    divider = make_axes_locatable(ax)
                    block_ax = divider.append_axes("top", size=0.05, pad=0.05, sharex=ax)
                    block_ax.imshow(h[i].T,
                                    aspect='auto', cmap='magma', interpolation='none')
                    block_ax.set_axis_off()
                    block_ax = divider.append_axes("top", size=0.05, pad=0.05, sharex=block_ax)
                    block_ax.imshow(m_pred[0, i].reshape(1, -1),
                                    aspect='auto', cmap='rainbow', interpolation='none',
                                    vmin=0, vmax=args.num_blocks)
                    block_ax.set_axis_off()
                    block_ax = divider.append_axes("top", size=0.05, pad=0.05, sharex=block_ax)
                    block_ax.imshow(m_true[i].reshape(1, -1),
                                    aspect='auto', cmap='rainbow', interpolation='none',
                                    vmin=0, vmax=args.num_blocks)
                    block_ax.set_axis_off()
                    writer.add_figure(f'example/{i:02d}', fig, n_iter)

        # write to console
        save_ckpt = ' *' if bool(loss_vl < best_loss) else ''
        logging.info(f'loss/validation={loss_vl:.4f}{save_ckpt}')
        best_loss = min(best_loss, loss_vl)
        
        # anneal beta
        if hasattr(net.gru, 'beta'):
            net.gru.beta = beta_functions[args.beta_func](n_iter)
            writer.add_scalar('beta', net.gru.beta, n_iter)

        # update parameters
        net.train()
        _loss_tr = []
        for _ in range(100):
            x, y, m = next(dataset)
            y_pred = net(x)[0]
            loss_tr = criterion(y_pred, y)
            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()
            _loss_tr.append(float(loss_tr))
        writer.add_scalar('loss/training', float(np.mean(_loss_tr)), n_iter)

        continue
        # checkpointing        
        if save_ckpt and checkpoint_dir is not None:
            torch.save(net.state_dict(), checkpoint_dir.joinpath('weights.pt'))
            with open(checkpoint_dir.joinpath('iterations.txt'), 'w') as fp:
                fp.write(f'{iteration}, {timedelta(seconds=time_taken)}')


if __name__ == '__main__':
    main()
