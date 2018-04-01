import numpy as np


class TimeSequence:
    """
    dt_delta - optional
    subsequence_length - MixedSignal only
    """
    def __init__(self, start, stop, n_timestamps, delta=None):

        self.start = start
        self.stop = stop
        self.n_timestamps = n_timestamps
        self.dt = (self.stop - self.start) / (self.n_timestamps - 1)
        self.uniform_timestamps = np.linspace(self.start, self.stop, self.n_timestamps)
        self.delta = 0 if delta is None else delta
        self._timestamps = None

    def __len__(self):
        return len(self.timestamps)

    def __call__(self):
        return self.timestamps

    @property
    def timestamps(self):
        if self._timestamps is None:
            self.generate()
        return self._timestamps

    def generate(self):
        """
        If self.delta == 0 and self.dt == 1
        time spacing is like...
        _timestamps 0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...

        If self.delta != 0
        each time point will be perturbed...
        uniform_timestamps 0.00, 1.00, 2.00, 3.00, 4.00, 5.00 ...
        deltas             +.10, -.35, +.12, -.14, -.02, -.13 ...
        _timestamps        0.10, 0.65, 2.12, 2.86, 3.98, 5.13 ...
        """
        if self.delta == 0:
            deltas = 0
        else:
            deltas = (self.dt / 2.0) * self.delta * (2.0 * np.random.uniform(size=self.n_timestamps) - 1)
        self._timestamps = self.uniform_timestamps + deltas
