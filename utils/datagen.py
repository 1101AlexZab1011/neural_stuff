import scipy as sp
import numpy as np
from typing import Union, Generator, Optional, Callable
import random
import mne
from abc import ABC, abstractmethod
from utils import sample_with_minimum_distance
from utils.signal import random_walk, band_limited_noise, compute_morlet_cwt, make_spikes, make_artifacts
from alltools.console.colored import warn


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


class BetaAlphaBoundGenerator(SignalGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
    ):
        if sfreq <= 65:
            raise ValueError(
                'To make beta oscillations at least 65 Hz '
                'sampling frequency is needed'
            )
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
            sig[:sig.shape[0] // 4] += sp.stats.zscore(
                band_limited_noise(
                    15,
                    30,
                    self._n_samples,
                    samplespacing=1 / self._sfreq
                )[:sig.shape[0] // 4]
            )
            sig[-sig.shape[0] // 4:] += sp.stats.zscore(
                band_limited_noise(
                    15,
                    30,
                    self._n_samples,
                    samplespacing=1 / self._sfreq
                )[:sig.shape[0] // 4]
            )
            sig[sig.shape[0] // 4:-sig.shape[0] // 4] += sp.stats.zscore(
                band_limited_noise(
                    8,
                    12,
                    self._n_samples,
                    samplespacing=1 / self._sfreq
                )[sig.shape[0] // 4:-sig.shape[0] // 4]
            )
            # sig = sp.stats.zscore(sig)
            freqs = np.arange(1, self._sfreq // 2)
            cwtmatr = compute_morlet_cwt(sig, self._x, freqs)
            yield sig, cwtmatr


class NoisySignal(SignalGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
        **kwargs
    ):
        super().__init__(tmin, tmax, sfreq)
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
            yield sig


class SpikesDatasetGenerator(SignalGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
        n_spikes: Union[int, tuple[int, int]],
        n_channels: int,
        spike_chance: Optional[float] = 1.,
        spike_negativity_chance: Optional[float] = 0.,
        spike_deviation: Optional[float] = 0.,
        spike_amplitude_deviation: Optional[Union[float, tuple[float, float]]] = 1,
        spike_fun: Optional[Callable] = None,
        *args,
        **kwargs
    ):
        super().__init__(tmin, tmax, sfreq)
        self._n_spikes = n_spikes
        self._n_channels = n_channels
        self._spike_chance = spike_chance
        self._spike_fun = spike_fun if spike_fun is not None else lambda: sp.signal.daub(10)
        dev = int(np.rint(spike_deviation * self._sfreq))
        self._spike_deviation = dev if dev > 0 else 1
        self._spike_fun_args = args
        self._spike_fun_kwargs = kwargs
        self._spike_amplitude_deviation = spike_amplitude_deviation
        self._spike_negativity_chance = spike_negativity_chance
        self._spike = sp.stats.zscore(
            self._spike_fun(
                *self._spike_fun_args,
                **self._spike_fun_kwargs
            )
        )

    def __call__(self, n_datasamples: int):
        for _ in range(n_datasamples):
            n_spikes = self._n_spikes \
                if isinstance(self._n_spikes, int) \
                else np.random.randint(*self._n_spikes)
            peak_times = sample_with_minimum_distance(len(self._x), n_spikes, len(self._spike))
            # print(peak_times)
            data = list()
            for _ in range(self._n_channels):
                sig = sp.stats.zscore(random_walk(self._x)) +\
                    sp.stats.zscore(
                        band_limited_noise(
                            0,
                            self._sfreq // 2,
                            self._n_samples,
                            samplespacing=1 / self._sfreq
                        )
                )
                peaks = np.array([
                    # to get rid of situation when peak goes out of signal
                    min(
                        peak_time +\
                        np.random.randint(2 * self._spike_deviation) -\
                        self._spike_deviation,
                        len(sig) - 1
                    )
                    for peak_time in peak_times
                ])
                for peak in peaks:
                    amplitude_ratio = self._spike_amplitude_deviation\
                        if isinstance(self._spike_amplitude_deviation, (int, float))\
                        else random.uniform(*self._spike_amplitude_deviation)
                    if np.random.random() <= self._spike_chance:
                        pos = -1 if np.random.random() <= self._spike_negativity_chance else 1
                        sig = make_spikes(sig, peak, pos * amplitude_ratio * self._spike)
                data.append(sig)
            yield np.array(data), peak_times


class ArtifactsDatasetGenerator(SignalGenerator):
    def __init__(
        self,
        tmin: Union[int, float],
        tmax: Union[int, float],
        sfreq: int,
        n_artifacts: Union[int, tuple[int, int]],
        n_channels: int,
        hfreq: Optional[int] = None,
        artifact_chance: Optional[float] = 1.,
        minimum_artifact_len: Optional[float] = .01,
        maximum_artifact_len: Optional[float] = 1.,
        artifact_negativity_chance: Optional[float] = 0.,
        artifact_exploding_chance: Optional[float] = 0.,
        artifact_exploding_amplitude_deviation: Optional[Union[float, tuple[float, float]]] = 0.,
        artifact_deviation: Optional[float] = 0.,
        artifact_amplitude_deviation: Optional[Union[float, tuple[float, float]]] = 1,
    ):
        super().__init__(tmin, tmax, sfreq)
        self._hfreq = self._sfreq // 2 if hfreq is None else hfreq
        assert self._hfreq <= self._sfreq // 2, 'Nyquist criterion not met'
        self._n_artifacts = n_artifacts
        self._n_channels = n_channels
        self._artifact_chance = artifact_chance
        dev = int(np.rint(artifact_deviation * self._sfreq))
        self._artifact_deviation = dev if dev > 0 else 1
        self._artifact_amplitude_deviation = artifact_amplitude_deviation
        self._artifact_negativity_chance = artifact_negativity_chance
        self._minimum_artifact_len = int(minimum_artifact_len * self._sfreq)
        self._maximum_artifact_len = int(maximum_artifact_len * self._sfreq)
        self._artifact_exploding_chance = artifact_exploding_chance
        self._artifact_exploding_amplitude_deviation = artifact_exploding_amplitude_deviation

    def __call__(self, n_datasamples: int):
        for i in range(n_datasamples):
            n_artifacts = self._n_artifacts \
                if isinstance(self._n_artifacts, int) \
                else np.random.randint(*self._n_artifacts)
            peak_times = sample_with_minimum_distance(
                len(self._x),
                n_artifacts,
                self._maximum_artifact_len
            )
            peaks = np.array([
                # to get rid of situation when peak goes out of signal
                min(
                    peak_time +\
                    np.random.randint(2 * self._artifact_deviation) -\
                    self._artifact_deviation,
                    len(self._x) - 1
                )
                for peak_time in peak_times
            ])
            all_bounds = {peak: list() for peak in peaks}
            # similar artifacts for the same peak
            data = np.array([
                sp.stats.zscore(random_walk(self._x)) +
                sp.stats.zscore(
                    band_limited_noise(
                        0,
                        self._hfreq,
                        self._n_samples,
                        samplespacing=1 / self._sfreq
                    )
                )
                for _ in range(self._n_channels)
            ])
            for peak in peaks:
                amplitude_ratio = self._artifact_amplitude_deviation\
                    if isinstance(self._artifact_amplitude_deviation, (int, float))\
                    else random.uniform(*self._artifact_amplitude_deviation)
                artifact_len = np.random.randint(
                    self._minimum_artifact_len,
                    self._maximum_artifact_len
                )
                exploding_amplitude_ratio = self._artifact_exploding_amplitude_deviation\
                    if isinstance(self._artifact_exploding_amplitude_deviation, (int, float))\
                    else random.uniform(*self._artifact_exploding_amplitude_deviation)
                exploding_len = np.random.randint(artifact_len // 2, artifact_len)
                exploding = np.random.random() <= self._artifact_exploding_chance
                for i, sig in enumerate(data):
                    if np.random.random() <= self._artifact_chance:
                        current_amplitude_ratio = random.uniform(
                            .75 * amplitude_ratio,
                            1.25 * amplitude_ratio
                        )
                        current_artifact_len = np.random.randint(
                            max(self._minimum_artifact_len, (.75 * artifact_len)),
                            min(int(1.25 * artifact_len), self._maximum_artifact_len))
                        pos = -1 if np.random.random() <= self._artifact_negativity_chance else 1
                        artifact_exploding = np.zeros(current_artifact_len - 1)
                        if exploding:
                            current_exploding_amplitude_ratio = random.uniform(
                                .75 * exploding_amplitude_ratio,
                                1.25 * exploding_amplitude_ratio
                            )
                            current_exploding_len = np.random.randint(
                                min(int(.75 * exploding_len), .75 * current_artifact_len),
                                min(int(1.25 * exploding_len), current_artifact_len)
                            )
                            start = np.random.randint(
                                0,
                                current_artifact_len - current_exploding_len
                            )
                            end = start + current_exploding_len
                            artifact_exploding[start:end] += pos *\
                                current_exploding_amplitude_ratio *\
                                sp.signal.gaussian(
                                current_exploding_len,
                                np.random.randint(
                                    current_exploding_len // 2,
                                    current_exploding_len
                                ),
                            )
                        sig, bounds = make_artifacts(
                            sig,
                            peak,
                            current_amplitude_ratio,
                            current_artifact_len,
                            self._sfreq // 2,
                            artifact_exploding
                        )
                        data[i] = sig
                        all_bounds[peak].append(np.array(bounds))

            out_bounds = list()
            for center, bounds in all_bounds.items():
                if bounds:
                    out_bounds.append(
                        [
                            np.squeeze(np.array(bounds), axis=1)[:, 0].min(),
                            np.squeeze(np.array(bounds), axis=1)[:, 1].max()
                        ]
                    )
                else:
                    warn(f'Empty bounds at {center} for {i}th datasample')

            yield np.array(data), np.array(out_bounds)
