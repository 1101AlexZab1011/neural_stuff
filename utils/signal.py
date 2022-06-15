import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from typing import Optional, Union, Generator, Any


def compute_morlet_cwt(
    sig: np.ndarray,
    t: np.ndarray,
    freqs: np.ndarray,
    omega_0: Optional[float] = 5,
    phase: Optional[bool] = False
) -> np.ndarray:
    dt = t[1] - t[0]
    widths = omega_0 / (2 * np.pi * freqs * dt)
    cwtmatr = signal.cwt(
        sig,
        lambda M, s: signal.morlet2(M, s, w=omega_0),
        widths
    )
    if phase:
        return cwtmatr
    else:
        return np.real(cwtmatr)**2 + np.imag(cwtmatr)**2


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplespacing=1):
    freqs = np.abs(np.fft.fftfreq(samples, samplespacing))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


def random_walk(x):
    out = []
    start = 0
    for _ in x:
        step = np.random.uniform(-1, 1)
        start = start + step
        out.append(start)

    return np.array(out)


def running_mean(x, n_samples):
    return np.convolve(x, np.ones((n_samples,)) / n_samples)[(n_samples - 1):]
