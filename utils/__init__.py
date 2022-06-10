import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from typing import Optional, Union, Generator, Any
import mne
from abc import ABC, abstractmethod
import pickle
import tensorflow as tf


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


def normal_rows(matr):
    matr_n = matr.copy()

    for i, row in enumerate(matr):
        matr_n[i] /= row.mean()

    return matr_n


def random_walk(x):
    out = []
    start = 0
    for _ in x:
        step = np.random.uniform(-1, 1)
        # print(step)
        start = start + step
        out.append(start)

    return np.array(out)


def read_pkl(path: str) -> Any:
    with open(
        path,
        'rb'
    ) as file:
        content = pickle.load(
            file
        )
    return content


def save_pkl(content: Any, path: str) -> None:
    if path[-4:] != '.pkl':
        raise OSError(f'Pickle file must have extension ".pkl", but it has "{path[-4:]}"')

    pickle.dump(content, open(path, 'wb'))


def plot_data_sample(X, P, Y, i, j):
    sig = X[i, 0, j, :]
    filtered_data = Y[i, 0, j, :]
    predicted_data = P[i, 0, j, :]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(sig)
    ax2.plot(filtered_data)
    ax3.plot(predicted_data)
    fig.set_size_inches(15, 5)
    plt.show()


class DatasetGenerator(ABC):
    @abstractmethod
    def __call__(self, n_datasamples: int) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        pass


class SignalGenerator(DatasetGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
    ):
        if tmax <= tmin:
            raise ValueError(f'Time to start must be lesser than time to end, {tmin = }, {tmax = }')

        self._tmin = tmin
        self._tmax = tmax
        self._sfreq = sfreq
        self._n_samples = int(np.rint((self._tmax - self._tmin) * self._sfreq))
        self._x = np.arange(self._n_samples) / self._sfreq
        self._times = np.linspace(self._tmin, self._tmax, len(self._x))

    def __call__(self, n_datasamples: int):
        for _ in range(n_datasamples):
            sig = sp.stats.zscore(random_walk(self._x)) +\
                sp.stats.zscore(
                    band_limited_noise(
                        0,
                        self._sfreq // 2,
                        self._n_samples,
                        samplespacing=1 / self._sfreq
                    )
            )
            yield sig

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        raise AttributeError('Can not set times')


class FIRDatasetGenerator(SignalGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
        lfreq: Union[int, float],
        hfreq: Union[int, float],
        **kwargs
    ):
        super().__init__(tmin, tmax, sfreq)
        self._lfreq = lfreq
        self._hfreq = hfreq
        self._fir_kwargs = kwargs

    def __call__(self, n_datasamples: int):
        for _ in range(n_datasamples):
            sig = sp.stats.zscore(random_walk(self._x)) +\
                sp.stats.zscore(
                    band_limited_noise(
                        0,
                        self._sfreq // 2,
                        self._n_samples,
                        samplespacing=1 / self._sfreq
                    )
            )
            filtered_sig = mne.filter.filter_data(
                sig,
                self._sfreq,
                self._lfreq,
                self._hfreq,
                verbose=False,
                **self._fir_kwargs
            )
            yield sig, filtered_sig


class CWTDatasetGenerator(SignalGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
    ):
        super().__init__(tmin, tmax, sfreq)

    def __call__(self, n_datasamples: int):
        for _ in range(n_datasamples):
            sig = sp.stats.zscore(random_walk(self._x)) +\
                sp.stats.zscore(
                    band_limited_noise(
                        0,
                        self._sfreq // 2,
                        self._n_samples,
                        samplespacing=1 / self._sfreq
                    )
            )
            freqs = np.arange(1, self._sfreq // 2)
            cwtmatr = compute_morlet_cwt(sig, self._x, freqs)
            yield sig, cwtmatr


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def plot_predicted_CWT(Y_true, Y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    mapable = ax1.imshow(
        Y_true,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
    )
    plt.colorbar(mapable, ax=ax1)
    mapable = ax2.imshow(
        Y_pred,
        aspect='auto',
        origin='lower',
        cmap='coolwarm',
    )
    plt.colorbar(mapable, ax=ax2)
    plt.show()
