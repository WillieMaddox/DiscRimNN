import string
import numpy as np
from typing import Text


def name_generator() -> Text:
    alphabet = np.array(list(string.ascii_uppercase))
    return ''.join(alphabet[np.random.randint(0, len(alphabet), 3)])


def color_generator() -> Text:
    r, g, b = np.random.randint(0, 255, 3)
    return '#%02X%02X%02X' % (r, g, b)

