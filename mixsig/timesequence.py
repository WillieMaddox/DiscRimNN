import numpy as np


class TimeSequence:
    """
    start
    stop
    n_timestamps
    dt

    dt_delta - optional
    subsequence_length - MixedSignal only
    """
    def __init__(self, start, stop, n_timestamps, delta=None):

        # nones_counter = 0
        # for kw in (start, stop, n_timestamps, dt):
        #     if kw is None:
        #         nones_counter += 1
        # assert nones_counter == 1

        self.start = start
        self.stop = stop
        self.n_timestamps = n_timestamps
        self.dt = (self.stop - self.start) / (self.n_timestamps - 1)
        self.uniform_timestamps = np.linspace(self.start, self.stop, self.n_timestamps)
        self.delta = 0 if delta is None else delta
        self._timestamps = None

    def __call__(self):
        return self.timestamps

    # @property
    # def start(self):
    #     if self._start is None:
    #         self._start = self._stop - (self._n_timestamps - 1) * self._dt
    #     return self._start

    # @property
    # def stop(self):
    #     if self._stop is None:
    #         self._stop = self._start + (self._n_timestamps - 1) * self._dt
    #     return self._stop

    # @property
    # def n_timestamps(self):
    #     if self._n_timestamps is None:
    #         n = (self._stop - self._start) / self._dt
    #         assert np.abs(n - np.rint(n)) <= np.finfo(float).eps
    #         self._n_timestamps = 1 + n
    #     return self._n_timestamps

    # @property
    # def dt(self):
    #     if self._dt is None:
    #         self._dt = (self.stop - self.start) / (self.n_timestamps - 1)
    #     return self._dt

    # @property
    # def uniform_timestamps(self):
    #     if self._uniform_timestamps is None:
    #         self._uniform_timestamps = np.linspace(self.start, self.stop, self.n_timestamps)
    #     return self._uniform_timestamps

    @property
    def timestamps(self):
        if self._timestamps is None:
            self.generate()
        return self._timestamps

    def generate(self):
        if self.delta == 0:
            deltas = 0
        else:
            deltas = (self.dt / 2.0) * self.delta * (2.0 * np.random.uniform(size=self.n_timestamps) - 1)
        self._timestamps = self.uniform_timestamps + deltas
