import types
from math import isclose
import numpy as np
from .utils import name_generator
from .utils import color_generator
from .utils import normal_noise_generator
from .utils import uniform_noise_generator
from .utils import no_noise
from .utils import timesequence_generator


class WaveProperty(float):
    def __new__(cls, mean=None, delta=None):
        mean = 0.0 if mean is None else float(mean)
        return super().__new__(cls, mean)

    def __init__(self, mean=None, delta=None):
        super().__init__()
        mean = 0.0 if mean is None else float(mean)
        delta = 0.0 if delta is None else float(delta)
        self.mean = mean
        self.delta = delta

        if isclose(delta, 0):
            self.generate = lambda: self
        else:
            self.generate = self._generator(delta)

    def __call__(self, **kwargs) -> float:
        return self.generate()

    def _generator(self, delta):
        def inner():
            return self + (2 * np.random.random() - 1) * delta  # i.e. np.random.uniform(self - delta, self + delta)
        return inner


class Amplitude(WaveProperty):

    def __new__(cls, mean=None, delta=None):
        mean = 1.0 if mean is None else float(mean)
        return super().__new__(cls, mean)

    def __init__(self, mean=None, delta=None):
        mean = 1.0 if mean is None else float(mean)
        super().__init__(mean, delta)

    def __call__(self, amplitude=1, **kwargs) -> float:
        return amplitude * self.generate()

    def _generator(self, delta):
        a_min, a_max = self - delta, self + delta

        def inner():
            return np.random.uniform(a_min, a_max)
        return inner


class Frequency(WaveProperty):

    def __new__(cls, mean=None, delta=None):
        mean = 1.0 if mean is None else float(mean)
        return super().__new__(cls, mean)

    def __init__(self, mean=None, delta=None):
        mean = 1.0 if mean is None else float(mean)
        super().__init__(mean, delta)

    def __call__(self, frequency=1, **kwargs) -> float:
        return frequency * self.generate()

    def _generator(self, delta):
        f_min, f_max = self - delta, self + delta

        def inner():
            # return 1. / np.random.uniform(1. / f_max, 1. / f_min)  # This distribution breaks when self == delta.
            return np.random.uniform(f_max, f_min)
        return inner


class Offset(WaveProperty):

    def __call__(self, offset=0, **kwargs) -> float:
        return offset + self.generate()

    def _generator(self, delta):
        b_min, b_max = self - delta, self + delta

        def inner():
            return np.random.uniform(b_min, b_max)
        return inner


class Phase(WaveProperty):

    def __call__(self, phase=0, **kwargs) -> float:
        return phase + self.generate()

    def _generator(self, delta):
        def inner():
            return np.random.random()  # later on this will be scaled by 2*pi
        return inner


class Wave:
    def __init__(self,
                 time=None,
                 amplitude=None,
                 frequency=None,
                 offset=None,
                 phase=None,
                 noise=None,
                 color=None,
                 name=None):

        self._sample = None
        self.timestamps = None

        self._timestamp_generator = timesequence_generator(**time) if time is not None else lambda: None
        self.is_independent = time is not None

        amplitude = {} if amplitude is None else amplitude
        self.amplitude = Amplitude(**amplitude)

        frequency = {} if frequency is None else frequency
        self.frequency = Frequency(**frequency)

        offset = {} if offset is None else offset
        self.offset = Offset(**offset)

        phase = {} if phase is None else phase
        self.phase = Phase(**phase)

        self.noise = None
        noise = {} if noise is None else noise
        if 'uniform' in noise:
            self._noise_generator = uniform_noise_generator(**noise['uniform'])
        elif 'normal' in noise:
            self._noise_generator = normal_noise_generator(**noise['normal'])
        else:
            self._noise_generator = no_noise()

        self.color = color_generator() if color is None else color
        self.name = name_generator() if name is None else name

    @property
    def sample(self):
        return self._sample

    @property
    def n_timestamps(self):
        return len(self.timestamps)

    def __call__(self, timestamps):
        if self.sample is None:
            self.generate(timestamps)
        return self.sample

    # @property
    # def timestamps(self):
    #     return self._timestamps()

    def generate(self, timestamps, **kwargs):

        t = self._timestamp_generator() if self.is_independent else timestamps
        a = self.amplitude(**kwargs)
        f = self.frequency(**kwargs)
        o = self.offset(**kwargs)
        p = self.phase(**kwargs)

        n = self._noise_generator(len(t))
        self._sample = a * np.sin(2.0 * np.pi * (f * t - p)) + o + n
        self.timestamps = t
        self.noise = n
        # return self._sample

    def __repr__(self):
        return 'Wave(amplitude={}, frequency={}, offset={}, phase={})'.format(self.amplitude, self.frequency, self.offset, self.phase)
