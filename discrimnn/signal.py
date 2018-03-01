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

    @staticmethod
    def _sample_random(r, r_range):
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
        self._signals = None
        self._inputs = None
        self._labels = None
        self._one_hots = None
        self.mixed_signal = None
        self.n_signals = len(sig_coeffs)
        self.n_timesteps = n_timesteps

        self.timestamps = np.linspace(start_time, stop_time, n_timesteps)

        self.signal_objects = []
        for coeffs in sig_coeffs:
            self.signal_objects.append(Signal(self.timestamps, **coeffs))

    def __call__(self):
        return self.inputs, self.one_hots

    def __len__(self):
        return len(self.signals)

    @property
    def signals(self):
        if self._signals is None:
            self._signals = np.empty((self.n_signals, self.n_timesteps))
            for i, signal in enumerate(self.signal_objects):
                self._signals[i, :] = signal()
        return self._signals

    @property
    def inputs(self):
        if self._inputs is None:
            self.generate()
        return self._inputs

    @property
    def labels(self):
        if self._labels is None:
            self.generate()
        return self._labels

    @property
    def one_hots(self):
        if self._one_hots is None:
            self.generate()
        return self._one_hots

    def generate(self):
        shuffled_indexes = np.arange(self.n_timesteps)
        np.random.shuffle(shuffled_indexes)
        self._labels = np.zeros(self.n_timesteps, dtype=int)
        for s in range(self.n_signals):
            self._labels[np.where(shuffled_indexes < s * self.n_timesteps // self.n_signals)] += 1

        self._one_hots = np.zeros((self.n_timesteps, self.n_signals), dtype=float)
        labels_tup = (np.arange(self.n_timesteps), self._labels)
        self._one_hots[labels_tup] = 1

        self._signals = None
        for signal in self.signal_objects:
            signal.generate()
        self.mixed_signal = np.sum(self._one_hots.T * self.signals, axis=0)
        # self._inputs = np.vstack((self.timestamps, self.mixed_signal)).T

        # self._inputs = x.reshape(self.n_timesteps, 2, 1)
        self._inputs = self.mixed_signal.reshape(self.n_timesteps, 1, 1)
        # self._inputs = x.reshape(1, self.n_timesteps, 2)
        # self._inputs = x.reshape(self.n_timesteps, 2)

        return self._inputs, self._one_hots

    def generate_batch(self, batch_size):
        x_batch = np.empty((batch_size, *self.inputs.shape))
        y_batch = np.empty((batch_size, *self.one_hots.shape))
        for i in range(batch_size):
            x, y = self.generate()
            x_batch[i] = x
            y_batch[i] = y
        return x_batch, y_batch
