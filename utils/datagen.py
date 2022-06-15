import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Generator
import mne
from abc import ABC, abstractmethod
from utils.signal import random_walk, band_limited_noise, compute_morlet_cwt


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
