import numbers
from collections import namedtuple
from math import isclose
import numpy as np
from .utils import name_generator
from .utils import color_generator
from .utils import normal_noise_generator
from .utils import uniform_noise_generator
from .utils import timesequence_generator
from .utils import create_label_distribution
from .utils import create_one_hots_from_labels

WaveProps = namedtuple('WaveProps', 'a w o p')


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
                 *features,
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

        self.features = features or ('x',)
        self._wp = None
        self._sample = None
        self._inputs = None

    @property
    def n_classes(self):
        return 1

    @property
    def sample(self):
        if self._sample is None:
            self._sample = self.d0xdt0()
        return self._sample

    @property
    def n_timestamps(self):
        return len(self.timestamps)

    def __len__(self):
        return len(self.timestamps)

    def generate(self, ts, **kwargs):
        self.timestamps = self._timestamp_generator() if self.is_independent else ts
        self.noise = self._noise_generator(len(self.timestamps))
        a = self.amplitude(**kwargs)
        w = self.frequency(**kwargs) * 2.0 * np.pi
        o = self.offset(**kwargs)
        p = self.phase(**kwargs) * 2.0 * np.pi
        self._wp = WaveProps(a, w, o, p)
        self._sample = None
        self._inputs = None

    @property
    def inputs(self):
        if self._inputs is None:
            self._inputs = np.zeros((len(self.timestamps), len(self.features)))
            for i, feat in enumerate(self.features):
                if feat in ('x', 'd0xdt0'):
                    feature = self.d0xdt0()
                elif feat in ('dxdt', 'd1xdt1'):
                    feature = self.d1xdt1()
                elif feat == 'd2xdt2':
                    feature = self.d2xdt2()
                elif feat == 'd3xdt3':
                    feature = self.d3xdt3()
                elif feat == 'time':
                    feature = self.timestamps
                else:
                    raise ValueError(f'Unknown feature {feat}')
                self._inputs[:, i] = feature
        return self._inputs

    def d0xdt0(self):
        a, w, o, p = self._wp
        return a * np.sin(w * self.timestamps - p) + o + self.noise

    def d1xdt1(self):
        a, w, o, p = self._wp
        return a * w * np.cos(w * self.timestamps - p)

    def d2xdt2(self):
        a, w, o, p = self._wp
        return -1 * a * w ** 2 * np.sin(w * self.timestamps - p)

    def d3xdt3(self):
        a, w, o, p = self._wp
        return -1 * a * w ** 3 * np.cos(w * self.timestamps - p)

    def __repr__(self):
        return f'Wave(amplitude={self.amplitude}, frequency={self.frequency}, offset={self.offset}, phase={self.phase})'


class MixedWave:
    def __init__(self,
                 sigs_coeffs,
                 msig_coeffs=None,
                 features=('x',),
                 name='Mixed'):

        self.name = name
        self.labels = None
        self.one_hots = None
        self.signals = None
        self.mixed_signal = None
        self.inputs = None
        self.mixed_inputs = None
        self.timestamps = None
        self.n_timestamps = None

        if 'time' in msig_coeffs:
            self.sequence_generator = timesequence_generator(**msig_coeffs['time'])

        mixed_signal_prop_defaults = {
            'amplitude': {'mean': 1, 'delta': 0},
            'frequency': {'mean': 1, 'delta': 0},
            'offset': {'mean': 0, 'delta': 0},
            'phase': {'mean': 0, 'delta': 0},
        }
        self.mixed_signal_props = {}
        for prop_name, default_coeffs in mixed_signal_prop_defaults.items():
            coeffs = msig_coeffs[prop_name] if prop_name in msig_coeffs else default_coeffs
            if prop_name == 'amplitude':
                self.mixed_signal_props[prop_name] = Amplitude(**coeffs)
            elif prop_name == 'frequency':
                self.mixed_signal_props[prop_name] = Frequency(**coeffs)
            elif prop_name == 'offset':
                self.mixed_signal_props[prop_name] = Offset(**coeffs)
            elif prop_name == 'phase':
                self.mixed_signal_props[prop_name] = Phase(**coeffs)

        self.features = features
        self.n_features = len(self.features)

        self.waves = [Wave(*features, **coeffs) for coeffs in sigs_coeffs]
        self.n_classes = len(self.waves)

    def generate(self):
        """ Generate waves from property values."""
        # First process the timestamp dependent waves.  (i.e. make a mixed signal wave.)
        # generate new timestamps
        self.timestamps = self.sequence_generator()
        self.n_timestamps = len(self.timestamps)
        # generate new mixed signal properties.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}

        # generate new individual waves.
        for wave in self.waves:
            wave.generate(self.timestamps, **props)

        # create a uniform distribution of class labels -> np.array([2,1,3, ... ,1])
        # (500,), (t,), (n_timestamps,)
        self.labels = create_label_distribution(self.n_timestamps, self.n_classes)

        # create one-hots from labels -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
        # (500, 4), (t, c), (n_timestamps, n_classes)
        self.one_hots = create_one_hots_from_labels(self.labels, self.n_classes)

        # (4, 500), (c, t), (n_classes, n_timestamps)
        self.signals = np.vstack([wave.sample for wave in self.waves if not wave.is_independent])
        # (500,), (t,), (n_timestamps,)
        self.mixed_signal = np.sum(self.one_hots.T * self.signals, axis=0)

        # (4, 500, 2), (c, t, f), (n_classes, n_timestamps, n_features)
        self.inputs = np.stack([wave.inputs for wave in self.waves if not wave.is_independent])
        # (500, 2), (t, f), (n_timestamps, n_features)
        self.mixed_inputs = np.sum(self.one_hots.T[..., None] * self.inputs, axis=0)
