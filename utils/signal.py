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


def make_spikes(tc, peak_coords, spike_tc):
    if not isinstance(peak_coords, (list, tuple, np.ndarray)):
        peak_coords = [peak_coords]

    spike_samples_before_peak = np.where(spike_tc == np.max(spike_tc))[0][0]

    for peak in peak_coords:
        spike_startsample = peak - spike_samples_before_peak
        spike_endsample = spike_startsample + len(spike_tc)

        if spike_startsample >= 0:
            startsample = 0
        else:
            startsample = -spike_startsample
            spike_startsample = 0
        if spike_endsample <= len(tc):
            endsample = 0
        else:
            endsample = len(spike_tc) - (spike_endsample - len(tc))
            spike_endsample = 0

        tc[spike_startsample:spike_endsample - 1] += spike_tc[startsample:endsample - 1]
    return tc


def make_artifacts(tc, center_coords, ratio, artifact_len, max_freq, additional_noise=0):
    if not isinstance(center_coords, (list, tuple, np.ndarray)):
        center_coords = [center_coords]

    artifacts_bounds = []

    for center in center_coords:
        artifact_startsample = int(max(center - artifact_len // 2, 0))
        artifact_endsample = artifact_startsample + artifact_len

        if artifact_endsample >= len(tc):
            artifact_endsample = 0

        noise = band_limited_noise(np.random.randint(0, max_freq // 2), max_freq, len(tc), 1 / 500)
        changed_len = len(tc[artifact_startsample:artifact_endsample - 1])
        tc[artifact_startsample:artifact_endsample - 1] += ratio * noise[
            artifact_startsample:artifact_endsample - 1
        ] + additional_noise[:changed_len]
        artifacts_bounds.append((artifact_startsample, artifact_endsample - 1))

    return tc, artifacts_bounds
