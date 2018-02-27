import numpy as np


class Signal:
    def __init__(self,
                 timestamps,
                 offset=0,
                 offset_range=None,
                 amplitude=1,
                 amplitude_range=None,
                 period=1,
                 period_range=None,
                 phase=0,
                 phase_range=None):

        self.timestamps = timestamps
        self._sample = None

        if offset_range is not None:
            assert len(offset_range) == 2
        else:
            assert offset is not None
        self.offset = offset
        self.offset_range = offset_range

        if amplitude_range is not None:
            assert len(amplitude_range) == 2
        else:
            assert amplitude is not None
        self.amplitude = amplitude
        self.amplitude_range = amplitude_range

        if period_range is not None:
            assert len(period_range) == 2
        else:
            assert period is not None
        self.period = period
        self.period_range = period_range

        if phase_range is not None:
            assert len(phase_range) == 2
        else:
            assert phase is not None
        self.phase = phase
        self.phase_range = phase_range

    def __call__(self):
        return self.sample

    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.generate()
        return self._sample

    def _sample_random(self, r, r_range):
        return r if r_range is None else (r_range[1] - r_range[0]) * np.random.random() + r_range[0]

    def generate(self):

        self.offset = self._sample_random(self.offset, self.offset_range)
        self.amplitude = self._sample_random(self.amplitude, self.amplitude_range)
        self.period = self._sample_random(self.period, self.period_range)
        self.phase = self._sample_random(self.phase, self.phase_range)

        self._sample = self.offset + self.amplitude * np.cos(2.0 * np.pi * self.timestamps / self.period - self.phase)
        return self._sample


class MixedSignal:
    def __init__(self, start_time, stop_time, n_timesteps, sig_coeffs):
        self._inputs = None
        self._labels = None

        self.n_signals = len(sig_coeffs)
        self.n_timesteps = n_timesteps

        self.timestamps = np.linspace(start_time, stop_time, n_timesteps)

        self.signals = np.empty((self.n_signals, n_timesteps))
        for i, coeffs in enumerate(sig_coeffs):
            self.signals[i, :] = Signal(self.timestamps, **coeffs)

        self.indices = np.zeros(self.n_timesteps, dtype=int)

    def __call__(self):
        return self.X, self.labels

    def __len__(self):
        return len(self.signals)

    @property
    def inputs(self):
        if self._inputs is None:
            self._inputs = 0
        return self._inputs

    @property
    def labels(self):
        if self._labels is None:
            self._labels = 0
        return self._labels

    def generate(self):
        shuff = np.arange(self.n_timesteps)
        np.random.shuffle(shuff)
        self.indices = np.zeros(self.n_timesteps, dtype=int)
        for s in range(len(self.signals)):
            self.indices[np.where(shuff < s * self.n_timesteps // self.n_signals)] += 1

        self._labels = np.zeros((self.n_signals, self.n_timesteps), dtype=float)
        indices_tup = (self.indices, np.arange(self.n_timesteps))
        self._labels[indices_tup] = 1

        x = np.sum(self._labels * self.signals, axis=0)
        x = np.vstack((self.timestamps, x)).T
        self._inputs = x.reshape(len(x), 2, 1)
