import argparse
import json
import logging
import torch
import torch.nn as nn

from pathlib import Path
from datetime import timedelta
from time import time
from torch.nn.functional import relu, tanh
from bsgru import BlockSparseGRU
from data import DatasetSineWave

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['gru', 'bsgru'])
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
    parser.add_argument('--seed', default=None)
    args = parser.parse_args()
    if args.seed is not None:
        args.seed = int(args.seed)
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
    args = parse_arguments()
    start_time = int(time())

    input_size = 1
    if args.n_fft:
        input_size = int(args.n_fft / 2 + 1)
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_blocks = len(args.frequencies)
    torch.manual_seed(args.seed)

    # instantiate model
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = (
                nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                if args.model == 'gru' else
                BlockSparseGRU(input_size, hidden_size, num_layers, num_blocks)
            )
            self.dnn = nn.Linear(hidden_size, input_size)
            self.tanh = nn.Tanh()
        def forward(self, x):
            o = self.gru(x)[0]
            o = self.tanh(self.dnn(o))
            return o

    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
    print('\nNetwork:', net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    print('Optimizer: Adam(lr={:.1e})'.format(args.learning_rate),
          '\nCriterion:', criterion)

    # instantiate dataset
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
    print('Dataset:', dataset)

    x_val, y_val, m_val = next(iter(dataset))
    iteration = -1
    
    losses, best_loss = [], 1e8
    print('\nElapsed Time\tIteration\tVal. Loss')

    # checkpoint directory
    checkpoint_dir = None
    # checkpoint_dir = Path(f'ckpt_{start_time}/')
    # checkpoint_dir.mkdir()
    # with open(checkpoint_dir.joinpath('arguments.txt'), 'w') as fp:
    #     json.dump(vars(args), fp, indent=2, sort_keys=True)

    # training loop
    for x, y, m in dataset:
        iteration += 1

        # forward pass
        net.train()
        y_pred = net(x)
        loss_tr = criterion(y_pred, y)
        
        # backpropagation
        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()

        if iteration % 100:
            continue

        # validation
        net.eval()
        with torch.no_grad():
            loss_vl = float(criterion(net(x_val), y_val))
            losses.append(loss_vl)

        time_taken = int(time()) - start_time
        save_ckpt = '*' if bool(loss_vl < best_loss) else ''
        print(f'{timedelta(seconds=time_taken)}\t\t{iteration:<10}'
              f'\t{loss_vl:.4f}\t{save_ckpt}')
        best_loss = min(best_loss, loss_vl)

        # checkpointing        
        if save_ckpt and checkpoint_dir is not None:
            torch.save(net.state_dict(), checkpoint_dir.joinpath('weights.pt'))
            with open(checkpoint_dir.joinpath('iterations.txt'), 'w') as fp:
                fp.write(f'{iteration}, {timedelta(seconds=time_taken)}')


if __name__ == '__main__':
    main()
