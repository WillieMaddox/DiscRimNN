import string
import numpy as np


def name_generator():
    alphabet = np.array(list(string.ascii_uppercase))
    return ''.join(alphabet[np.random.randint(0, len(alphabet), 3)])


def color_generator():
    r, g, b = np.random.randint(0, 255, 3)
    return '#%02X%02X%02X' % (r, g, b)


def noise_generator(noise_type, **kwargs):
    pass


