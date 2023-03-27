import numpy as np
import torch

def compute_psnr(a, b):
    """
    Compute Peak-Signal-to-Noise-Ratio across two numpy.ndarray objects
    """
    if torch.is_tensor(a):
        a = a.numpy()
    if torch.is_tensor(b):
        b = b.numpy()

    max_b = np.abs(b).max()
    sumdeltasq = 0.0

    sumdeltasq = ((a - b) * (a - b)).sum()

    sumdeltasq /= b.size
    sumdeltasq = np.sqrt(sumdeltasq)

    eps = 1e-5
    eps2 = 1e-10
    psnr = 20 * np.log10((max_b + eps) / (sumdeltasq + eps2))

    return psnr