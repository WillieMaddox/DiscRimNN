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
    def gen_noise(n_timestamps):
        return np.random.uniform(mu - delta, mu + delta, (n_timestamps,))
    return gen_noise


def normal_noise_generator(mu=0, sigma=0.01):
    def gen_noise(n_timestamps):
        return np.random.normal(mu, sigma, (n_timestamps,))
    return gen_noise


def no_noise():
    def gen_noise(*args, **kwargs):
        return 0
    return gen_noise
