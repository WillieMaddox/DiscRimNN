import types
from math import isclose
import numpy as np
from .utils import name_generator
from .utils import color_generator
from .utils import normal_noise_generator
from .utils import uniform_noise_generator
from .utils import no_noise
from .utils import timesequence_generator


class WaveProperty:
    def __init__(self, mean=None, delta=None):
        self.mean = 0.0 if mean is None else float(mean)
        self.delta = 0.0 if delta is None else float(delta)
        self._value = None

    def __call__(self):
        return self.value

    @property
    def value(self):
        if self._value is None:
            self._generate()
        return self._value

    def generate(self):
        self._generate()
        return self._value

    def _generate(self):
        self._value = (
            self.mean
            if self.delta == 0
            else (2 * np.random.random() - 1) * self.delta + self.mean
        )


class Wave:
    def __init__(self,
                 timestamps,
                 amplitude=None,
                 frequency=None,
                 offset=None,
                 phase=None,
                 noise=None,
                 color=None,
                 name=None):

        if isinstance(timestamps, types.FunctionType):
            self._timestamps = timestamps
        else:
            self._timestamps = lambda: timestamps

        self._sample = None

        amplitude = {} if amplitude is None else amplitude
        self._amplitude = WaveProperty(**amplitude)

        frequency = {} if frequency is None else frequency
        self._frequency = WaveProperty(**frequency)

        offset = {} if offset is None else offset
        self._offset = WaveProperty(**offset)

        phase = {} if phase is None else phase
        self._phase = WaveProperty(**phase)

        self.signal_noise = None
        noise = {} if noise is None else noise
        if 'uniform' in noise:
            self.signal_noise_generator = uniform_noise_generator(**noise['uniform'])
        elif 'normal' in noise:
            self.signal_noise_generator = normal_noise_generator(**noise['normal'])
        else:
            self.signal_noise_generator = no_noise()

        self.name = name_generator() if name is None else name
        self.color = color_generator() if color is None else color

    @property
    def timestamps(self):
        return self._timestamps()

    @property
    def amplitude(self):
        return self._amplitude()

    @property
    def frequency(self):
        return self._frequency()

    @property
    def offset(self):
        return self._offset()

    @property
    def phase(self):
        return self._phase()

    def __call__(self):
        return self.sample

    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.generate()
        return self._sample

    def generate(self, **kwargs):

        amplitude = self._amplitude.generate()
        if 'amplitude' in kwargs:
            amplitude *= kwargs['amplitude']
        frequency = self._frequency.generate()
        if 'frequency' in kwargs:
            frequency *= kwargs['frequency']
        offset = self._offset.generate()
        if 'offset' in kwargs:
            offset += kwargs['offset']
        phase = self._phase.generate()
        if 'phase' in kwargs:
            phase += kwargs['phase']

        self.signal_noise = self.signal_noise_generator(len(self.timestamps))
        self._sample = amplitude * np.sin(2.0 * np.pi * (self.timestamps * frequency - phase)) + offset + self.signal_noise
        return self._sample

    def __repr__(self):
        return 'Wave(amplitude={}, frequency={}, offset={}, phase={})'.format(self.amplitude, self.frequency, self.offset, self.phase)

