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


def timesequence_generator(t_min=None, t_max=None, n_max=None, n_min=None, noise_type=None, **kwargs):
    """
    If self.delta == 0 and self.dt == 1
    the time spacing will look like...
    _timestamps 0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...

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


