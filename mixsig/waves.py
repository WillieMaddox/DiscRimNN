import string
import numpy as np


def name_generator():
    alphabet = np.array(list(string.ascii_uppercase))
    return ''.join(alphabet[np.random.randint(0, len(alphabet), 3)])


def color_generator():
    r, g, b = np.random.randint(0, 255, 3)
    return '#%02X%02X%02X' % (r, g, b)


class WaveProperty:
    def __init__(self, mean=None, delta=None):
        self.mean = 0.0 if mean is None else float(mean)
        self.delta = 0.0 if delta is None else float(delta)
        self._value = None

    # def __init__(self, kwargs):
    #     kwargs = {} if kwargs is None else kwargs
    #     self.mean = kwargs['mean'] if 'mean' in kwargs else 0
    #     self.delta = kwargs['delta'] if 'delta' in kwargs else 0

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
        self._value = self.mean if self.delta == 0 else (2 * np.random.random() - 1) * self.delta + self.mean


class Wave:
    def __init__(self, timestamps=None, amplitude=None, period=None, offset=None, phase=None, name=None, color=None):

        self.timestamps = timestamps
        self._sample = None

        amplitude = {} if amplitude is None else amplitude
        self._amplitude = WaveProperty(**amplitude)
        period = {} if period is None else period
        self._period = WaveProperty(**period)
        offset = {} if offset is None else offset
        self._offset = WaveProperty(**offset)
        phase = {} if phase is None else phase
        self._phase = WaveProperty(**phase)

        self.name = name_generator() if name is None else name
        self.color = color_generator() if color is None else color

    def __call__(self):
        return self.sample

    @property
    def amplitude(self):
        return self._amplitude()

    @property
    def period(self):
        return self._period()

    @property
    def offset(self):
        return self._offset()

    @property
    def phase(self):
        return self._phase()

    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.generate()
        return self._sample

    def generate(self, offset=0, amplitude=0, period=0, phase=0):
        amplitude = self.amplitude.generate() + amplitude
        period = self.period.generate() + period
        offset = self.offset.generate() + offset
        phase = self.phase.generate() + phase

        self._sample = offset + amplitude * np.cos(2.0 * np.pi * self.timestamps / period - phase)
        return self._sample


