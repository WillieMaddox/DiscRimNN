import numbers
from math import isclose
import numpy as np
from .utils import name_generator
from .utils import color_generator
from .utils import normal_noise_generator
from .utils import uniform_noise_generator
from .utils import timesequence_generator


class WavePropertyOld(float):
    def __new__(cls, mean=None, delta=None):
        mean = 0.0 if mean is None else float(mean)
        return super().__new__(cls, mean)

    def __init__(self, mean=None, delta=None):
        super().__init__()
        mean = 0.0 if mean is None else float(mean)
        delta = 0.0 if delta is None else float(delta)
        self.mean = mean
        self.delta = delta

        if isclose(delta, 0, abs_tol=1e-9):
            self.generate = lambda: self
        else:
            self.generate = self._generator(delta)

    def __call__(self, **kwargs) -> float:
        return self.generate()

    def _generator(self, delta):
        def inner():
            return self + (2 * np.random.random() - 1) * delta  # i.e. np.random.uniform(self - delta, self + delta)
        return inner


class AmplitudeOld(WavePropertyOld):

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


class FrequencyOld(WavePropertyOld):

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


class OffsetOld(WavePropertyOld):

    def __call__(self, offset=0, **kwargs) -> float:
        return offset + self.generate()

    def _generator(self, delta):
        b_min, b_max = self - delta, self + delta

        def inner():
            return np.random.uniform(b_min, b_max)
        return inner


class PhaseOld(WavePropertyOld):

    def __call__(self, phase=0, **kwargs) -> float:
        return phase + self.generate()

    def _generator(self, delta):
        def inner():
            return np.random.random()  # later on this will be scaled by 2*pi
        return inner


class WaveProperty:

    def __init__(self, mean=None, delta=None):
        self.value = 0.0 if mean is None else float(mean)
        self.delta = 0.0 if delta is None else float(delta)

        if isclose(self.delta, 0, abs_tol=1e-9):
            self.generate = lambda: self.value
        else:
            self.generate = self._generator(self.delta)

    def __call__(self, **kwargs) -> float:
        self.value = self.generate()
        return self.value

    def _generator(self, delta):
        def inner():
            return self + (2 * np.random.random() - 1) * delta  # i.e. np.random.uniform(self - delta, self + delta)
        return inner

    # def __repr__(self):
    #     return f'WaveProperty(mean={self.value}, delta={self.delta})'

    def __repr__(self):
        return f'{self.value}'

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.value + other.value
        elif isinstance(other, numbers.Real):
            return self.value + other
        else:
            raise NotImplemented

    def __radd__(self, other):
        if isinstance(other, numbers.Real):
            return other + self.value
        else:
            raise TypeError(f'Can only add values of type {numbers.Real}, not of type {type(other)}')

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.value * other.value
        elif isinstance(other, numbers.Real):
            return self.value * other
        else:
            raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Real):
            return other * self.value
        else:
            raise TypeError(f'Can only multiply values of type {numbers.Real}, not of type {type(other)}')

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        elif isinstance(other, numbers.Real):
            return self.value == other
        else:
            raise NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return self.value != other.value
        elif isinstance(other, numbers.Real):
            return self.value != other
        else:
            raise NotImplemented

    def __gt__(self, other):
        if isinstance(other, self.__class__):
            return self.value > other.value
        elif isinstance(other, numbers.Real):
            return self.value > other
        else:
            raise NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.value < other.value
        elif isinstance(other, numbers.Real):
            return self.value < other
        else:
            raise NotImplemented


class Amplitude(WaveProperty):

    def __init__(self, mean=None, delta=None):
        mean = 1.0 if mean is None else float(mean)
        super().__init__(mean, delta)

    def __call__(self, amplitude=1, **kwargs) -> float:
        self.value = self.generate()
        return self.value * amplitude

    def _generator(self, delta):
        a_min, a_max = self.value - delta, self.value + delta

        def inner():
            return np.random.uniform(a_min, a_max)
        return inner

    # def __repr__(self):
    #     return f'Amplitude(mean={self.value}, delta={self.delta})'


class Frequency(WaveProperty):

    def __init__(self, mean=None, delta=None):
        mean = 1.0 if mean is None else float(mean)
        super().__init__(mean, delta)

    def __call__(self, frequency=1, **kwargs) -> float:
        self.value = self.generate()
        return self.value * frequency

    def _generator(self, delta):
        f_min, f_max = self.value - delta, self.value + delta

        def inner():
            # This distribution breaks when self.value == delta.
            # return 1. / np.random.uniform(1. / f_max, 1. / f_min)
            return np.random.uniform(f_max, f_min)
        return inner


class Offset(WaveProperty):

    def __call__(self, offset=0, **kwargs) -> float:
        self.value = self.generate()
        return self.value + offset

    def _generator(self, delta):
        b_min, b_max = self.value - delta, self.value + delta

        def inner():
            return np.random.uniform(b_min, b_max)
        return inner


class Phase(WaveProperty):

    def __call__(self, phase=0, **kwargs) -> float:
        self.value = self.generate()
        return self.value + phase

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

        self.timestamps = None

        self._timestamp_generator = timesequence_generator(**time) if time is not None else lambda: None
        self.is_independent = time is not None

        amplitude = amplitude or {}
        self.amplitude = Amplitude(**amplitude)

        frequency = frequency or {}
        self.frequency = Frequency(**frequency)

        offset = offset or {}
        self.offset = Offset(**offset)

        phase = phase or {}
        self.phase = Phase(**phase)

        self.noise = None
        noise = noise or {}
        if 'uniform' in noise:
            self._noise_generator = uniform_noise_generator(**noise['uniform'])
        elif 'normal' in noise:
            self._noise_generator = normal_noise_generator(**noise['normal'])
        else:
            self._noise_generator = lambda *args, **kwargs: 0

        self.color = color or color_generator()
        self.name = name or name_generator()

        self._sample = None
    @property
    def sample(self):
        return self._sample

    @property
    def n_timestamps(self):
        return len(self.timestamps)

    def __len__(self):
        return len(self.timestamps)

    # @property
    # def timestamps(self):
    #     return self._timestamps()

    def generate(self, timestamps, **kwargs):

        t = self._timestamp_generator() if self.is_independent else timestamps
        n = self._noise_generator(len(t))
        a = self.amplitude(**kwargs)
        f = self.frequency(**kwargs)
        o = self.offset(**kwargs)
        p = self.phase(**kwargs)

        self._sample = a * np.sin(2.0 * np.pi * (f * t - p)) + o + n
        self.timestamps = t
        self.noise = n
        # return self._sample

    def __repr__(self):
        return f'Wave(amplitude={self.amplitude}, frequency={self.frequency}, offset={self.offset}, phase={self.phase})'
