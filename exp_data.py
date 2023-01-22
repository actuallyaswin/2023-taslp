import numpy as np
import torch
from typing import List, Optional, Sequence

def make_sine_waves(
        amount: int,
        seed: int = 0,
        frequencies: List[int] = [5, 7, 9],
        sample_rate: int = 20,
        duration: int = 5,
        apply_random_amplitude: bool = False,
        window_size: float = 1,
        num_samples_pred: int = 1,
        to_tensor: bool = True
    ):
    rng = np.random.default_rng(seed)
    
    x_batch, m_batch = [], []
    t = np.arange(0, int(duration * sample_rate)) / sample_rate

    for _ in range(amount):

        # prepare masking variable that selects one
        # of the sources at each second
        m = rng.choice(
            range(len(frequencies)),
            size=duration)
        m = np.repeat(m, sample_rate)

        # construct mixture
        x = []
        for f in frequencies:
            a = 1 if not apply_random_amplitude else rng.random()
            p = 0 # ignore phase shifts
            s = a * np.cos(2*np.pi*f*t)
            x.append(s)

        # apply mask
        x = np.choose(m, x)

        # normalize the signal to have a max amplitude of 1
        x = x / (np.max(x) + 1e-8)

        # multiply hann window for transitions
        if window_size > 0:
            transition_points = np.abs(np.diff(m, append=1))
            transition_scaling = 1 - np.convolve(
                np.hanning(round(sample_rate*window_size)),
                transition_points, 'same')
            x *= transition_scaling
        x_batch.append(x)
        
        # make `m` one-hot
        m = [np.eye(len(frequencies))[i] for i in m]
        m_batch.append(m)
    
    x_batch = np.array(x_batch)[..., np.newaxis]
    m_batch = np.array(m_batch)
    if to_tensor:
        x_batch = torch.FloatTensor(x_batch).cuda()
        m_batch = torch.FloatTensor(m_batch).cuda()
    batch = (
        x_batch[:, :-num_samples_pred],
        x_batch[:, num_samples_pred:],
        m_batch[:, :-num_samples_pred]
    )
    return batch