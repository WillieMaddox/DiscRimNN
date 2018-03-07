import numpy as np


class BaseNoise(object):
    def reset(self):
        pass


class UniformNoise(BaseNoise):
    def __init__(self, mu=0.0, sigma=0.01):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'UniformNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NormalNoise(BaseNoise):
    def __init__(self, mu=0.0, sigma=0.01):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OUNoise(BaseNoise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        if isinstance(mu, np.ndarray):
            assert mu.shape[0] == size
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class WaveProperty:
    def __init__(self, kwargs):
        assert 'mean' in kwargs
        self.mean = kwargs['mean']
        self.delta = kwargs['delta'] if 'delta' in kwargs else 0
        self._value = None

    @property
    def value(self):
        if self._value is None:
            self._generate()
        return self._value

    def generate(self):
        self._generate()
        return self._value

    def _generate(self):
        self._value = self.mean if self.delta == 0 else (2 * np.random.random() - 1) * self.delta + self.mean


class Signal:
    def __init__(self, timestamps, offset=None, amplitude=None, period=None, phase=None):

        self.timestamps = timestamps
        self._sample = None

        offset = {} if offset is None else offset
        if 'mean' not in offset:
            offset['mean'] = 0
        self.offset = WaveProperty(offset)

        amplitude = {} if amplitude is None else amplitude
        if 'mean' not in amplitude:
            amplitude['mean'] = 1
        self.amplitude = WaveProperty(amplitude)

        period = {} if period is None else period
        if 'mean' not in period:
            period['mean'] = 1
        self.period = WaveProperty(period)

        phase = {} if phase is None else phase
        if 'mean' not in phase:
            phase['mean'] = 0
        self.phase = WaveProperty(phase)

    def __call__(self):
        return self.sample

    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.generate()
        return self._sample

    def generate(self, offset=None, amplitude=None, period=None, phase=None):

        offset = self.offset.generate() if offset is None else offset
        amplitude = self.amplitude.generate() if amplitude is None else amplitude
        period = self.period.generate() if period is None else period
        phase = self.phase.generate() if phase is None else phase

        self._sample = offset + amplitude * np.cos(2.0 * np.pi * self.timestamps / period - phase)
        return self._sample


class MixedSignal:
    def __init__(self, time_coeffs, sig_coeffs, msig_coeffs=None):
        self.signals = None
        self.inputs = None
        self.labels = None
        self.classes = None
        self.one_hots = None
        self.mixed_signal = None
        self.n_timestamps = time_coeffs['n_timestamps']
        self.n_timesteps = time_coeffs['n_timesteps']
        self.n_signals = len(sig_coeffs)
        self.n_samples = self.n_timestamps - (self.n_timesteps - 1)
        self.timestamps = np.linspace(time_coeffs['start'], time_coeffs['stop'], self.n_timestamps)

        self.mixed_signal_props = {}
        for name, coeffs in msig_coeffs.items():
            self.mixed_signal_props[name] = WaveProperty(coeffs)

        self.signal_objects = []
        for coeffs in sig_coeffs:
            self.signal_objects.append(Signal(self.timestamps, **coeffs))

    def __call__(self):
        if self.inputs is None or self.labels is None:
            self.generate()
        return self.inputs, self.labels

    def __len__(self):
        return self.n_signals

    def generate_property_values(self):
        prop_vals = {}
        for name, prop in self.mixed_signal_props.items():
            prop_vals[name] = prop.generate()
        return prop_vals

    def generate(self):
        shuffled_indexes = np.arange(self.n_timestamps)
        np.random.shuffle(shuffled_indexes)
        self.classes = np.zeros(self.n_timestamps, dtype=int)
        for s in range(self.n_signals):
            self.classes[np.where(shuffled_indexes < s * self.n_timestamps // self.n_signals)] += 1

        self.one_hots = np.zeros((self.n_timestamps, self.n_signals), dtype=float)
        classes_tup = (np.arange(self.n_timestamps), self.classes)
        self.one_hots[classes_tup] = 1

        mixed_prop_vals = self.generate_property_values()
        self.signals = np.empty((self.n_signals, self.n_timestamps))
        for i, signal in enumerate(self.signal_objects):
            signal.generate(**mixed_prop_vals)
            self.signals[i, :] = signal()

        self.mixed_signal = np.sum(self.one_hots.T * self.signals, axis=0)

        # self.inputs = np.vstack((self.timestamps, self.mixed_signal)).T
        # self.inputs = self.inputs.reshape(self.n_timesteps, 2, 1)
        # self.inputs = self.inputs.reshape(self.n_timesteps, 1, 2)
        # self.inputs = self.inputs.reshape(1, self.n_timesteps, 2)
        # self.inputs = self.inputs.reshape(self.n_timesteps, 2)

        # self.inputs = self.mixed_signal.reshape(self.n_timestamps, 1, 1)
        # self.inputs = self.mixed_signal.reshape(1, self.n_timesteps, 1)

        mixed_signal = np.zeros((self.n_samples, self.n_timesteps))
        for i in range(self.n_timesteps):
            mixed_signal[:, i] = self.mixed_signal[i:i+self.n_samples]
        self.inputs = mixed_signal.reshape(self.n_samples, self.n_timesteps, 1)
        # self.labels = self.one_hots.reshape(n_samples, self.n_timesteps, self.n_signals)
        self.labels = self.one_hots[(self.n_timesteps - 1):]
        return self.inputs, self.labels

    def generate_batch(self, batch_size):
        x_batch = np.empty((batch_size, *self.inputs.shape))
        y_batch = np.empty((batch_size, *self.one_hots.shape))
        for i in range(batch_size):
            x, y = self.generate()
            x_batch[i] = x
            y_batch[i] = y
        return x_batch, y_batch
