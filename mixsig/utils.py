import string
import numpy as np
from typing import Text


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


def no_noise():
    def gen_noise(*args, **kwargs):
        return 0
    return gen_noise


def timesequence_generator(start, stop, n_timestamps, delta=None):
    """
    If self.delta == 0 and self.dt == 1
    time spacing is like...
    _timestamps 0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...

    If self.delta != 0
    each time point will be perturbed...
    uniform_timestamps 0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...
    noise              +.10, -.35, +.12, -.14, -.02, -.13 ...
    _timestamps        0.10, 0.65, 2.12, 2.86, 3.98, 5.13 ...
    """

    dt = (stop - start) / (n_timestamps - 1)
    uniform_timestamps = np.linspace(start, stop, n_timestamps)
    delta = 0 if delta is None else delta
    if delta == 0:
        def gen_noise():
            return uniform_timestamps
    else:
        def gen_noise():
            noise = (dt / 2.0) * delta * (2.0 * np.random.uniform(size=n_timestamps) - 1)
            return uniform_timestamps + noise
    return gen_noise
