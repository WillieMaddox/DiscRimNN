import string
from functools import reduce
from datetime import datetime
import numpy as np
from typing import Text


def get_datetime_now(t=None, fmt='%Y_%m%d_%H%M_%S'):
    """Return timestamp as a string; default: current time, format: YYYY_DDMM_hhmm_ss."""
    if t is None:
        t = datetime.now()
    return t.strftime(fmt)


def factors(n):
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0)))


def bits2shape(bitstr, seq_bits):
    shp = None
    for bit in bitstr:
        if bit in seq_bits:
            shp = (seq_bits[bit], ) if shp is None else shp + (seq_bits[bit], )
    return shp


def name_generator() -> Text:
    alphabet = np.array(list(string.ascii_uppercase))
    return ''.join(alphabet[np.random.randint(0, len(alphabet), 3)])


def color_generator() -> Text:
    r, g, b = np.random.randint(0, 255, 3)
    return '#%02X%02X%02X' % (r, g, b)


def uniform_noise_generator(mu=0, delta=0.5):
    lo, hi = mu - delta, mu + delta

    def gen_noise(n_timestamps):
        return np.random.uniform(lo, hi, (n_timestamps,))
    return gen_noise


def normal_noise_generator(mu=0, sigma=0.01):
    def gen_noise(n_timestamps):
        return np.random.normal(mu, sigma, (n_timestamps,))
    return gen_noise


def timesequence_generator(t_min=None, t_max=None, n_max=None, n_min=None, noise_type=None, **kwargs):
    """
    If self.delta == 0 and self.dt == 1
    the time spacing will look like...
    _timestamps = [0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...]

    If self.delta != 0
    each time point will be perturbed...
    uniform_timestamps 0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...
    noise              +.10, -.35, +.12, -.14, -.02, -.13 ...
    _timestamps        0.10, 0.65, 2.12, 2.86, 3.98, 5.13 ...
    """

    if t_min is None or t_max is None:
        raise ValueError("Both t_min and t_max are required.")
    elif t_max <= t_min:
        raise ValueError("t_min must be less than t_max.")

    n_timestamps = kwargs.get('n_timestamps', None)
    if n_max is None and n_timestamps is None:
        raise ValueError("n_max or n_timestamps is required.")
    elif n_max is not None and n_timestamps is not None:
        raise ValueError("Please specify either n_max or n_timestamps, not both.")
    elif n_max is None:
        n_max = n_timestamps

    if n_min is None:
        n_min = n_max
    assert 2 < n_min <= n_max

    noise_type = noise_type or ''
    assert noise_type.lower() in ('pareto', 'large', 'jitter', 'small', '')
    endpoint = kwargs.get('endpoint', False)

    if n_min != n_max:
        def gen_n_timestamps():
            return np.random.randint(n_min, n_max + 1)
    else:
        def gen_n_timestamps():
            return n_max

    if noise_type in ('pareto', 'large'):
        # large gaps in the timesequence.  The smaller pareto_shape, the larger the gaps.
        pareto_shape = kwargs.get('pareto_shape', None)
        pareto_shape = 2 if pareto_shape is None else pareto_shape
        assert pareto_shape is not None and pareto_shape > 0, ValueError('shape should be greater than 0.')

        def gen_timesequence():
            times = np.cumsum(np.random.pareto(pareto_shape, size=gen_n_timestamps()))
            slope = (t_max - t_min) / (times[-1] - times[0])
            intercept = t_max - slope * times[-1]
            return slope * times + intercept

    elif noise_type in ('jitter', 'small'):
        # slight perturbations no greater than dt / 2
        delta = kwargs.get('delta', None)
        delta = 1 if delta is None else delta
        assert delta is not None and 0 < delta <= 1, ValueError('delta should be between 0 and 1.')

        if n_min != n_max:
            def gen_timesequence():
                n_timestamps = gen_n_timestamps()
                uniform_timestamps = np.linspace(t_min, t_max, n_timestamps, endpoint=endpoint)
                dt = (t_max - t_min) / (n_timestamps - 1)
                noise = (dt / 2.0) * delta * (2.0 * np.random.uniform(size=n_timestamps) - 1)
                return uniform_timestamps + noise
        else:
            n_timestamps = gen_n_timestamps()
            uniform_timestamps = np.linspace(t_min, t_max, n_timestamps, endpoint=endpoint)
            dt = (t_max - t_min) / (n_timestamps - 1)

            def gen_timesequence():
                noise = (dt / 2.0) * delta * (2.0 * np.random.uniform(size=n_timestamps) - 1)
                return uniform_timestamps + noise

    else:
        # timestamps are all evenly spaced. dt is constant.
        if n_min != n_max:
            def gen_timesequence():
                n_timestamps = gen_n_timestamps()
                uniform_timestamps = np.linspace(t_min, t_max, n_timestamps, endpoint=endpoint)
                return uniform_timestamps
        else:
            n_timestamps = gen_n_timestamps()
            uniform_timestamps = np.linspace(t_min, t_max, n_timestamps, endpoint=endpoint)

            def gen_timesequence():
                return uniform_timestamps

    return gen_timesequence


def create_label_distribution(n_timestamps, n_classes):
    """
    Create a distribution of ints which represent class labels.
    :return np.array([2,1,3, ... ,1])
    """
    shuffled_indexes = np.arange(n_timestamps)
    np.random.shuffle(shuffled_indexes)
    labels = np.zeros(n_timestamps, dtype=int)
    for c in range(n_classes):
        labels[np.where(shuffled_indexes < c * n_timestamps // n_classes)] += 1
    return labels


def create_one_hots_from_labels(labels, n_classes):
    """
    Create one-hot vector from the class label distribution.
    self.one_hots -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
    """
    length = len(labels)
    one_hots = np.zeros((length, n_classes), dtype=float)
    one_hots[(np.arange(length), labels)] = 1
    return one_hots


# generate a sequence of random numbers in [0, n_classes)
def generate_sequence(length, n_classes, labels=None):
    if labels is None:
        return np.random.randint(0, n_classes, length)
    else:
        assert n_classes == len(labels)
        seq = np.random.randint(0, n_classes, length)
        return labels[seq]


# one hot encode sequence
def one_hot_encode(sequence, n_classes):
    return np.identity(n_classes)[sequence]


# decode a one hot encoded sequence
def one_hot_decode(encoded_sequence):
    return np.argmax(encoded_sequence, axis=-1)
