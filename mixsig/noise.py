import numpy as np
from .utils import name_generator
from .utils import color_generator


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self,
                 n_timestamps,
                 mu=0.0,
                 theta=0.15,
                 sigma=0.2,
                 color=None,
                 name=None):

        """Initialize parameters and noise process."""
        self.n_timestamps = n_timestamps

        if isinstance(mu, (tuple, list, np.ndarray)):
            self.n_signals = len(mu)
        else:
            self.n_signals = 1

        self.mu = np.ones(self.n_signals) * mu

        self.theta = theta
        self.sigma = sigma

        self.state = self.mu

        self.name = name_generator() if name is None else name
        self.color = color_generator() if color is None else color

        self._signals = None

    def __len__(self):
        return len(self.signals)

    def __call__(self):
        return self.signals

    @property
    def signals(self):
        if self._signals is None:
            self._signals = self.generate()
        return self._signals

    def generate(self, **kwargs):
        if 'reset' in kwargs and kwargs['reset']:
            self.reset()

        self._signals = np.empty((self.n_signals, self.n_timestamps))
        for i in range(self.n_timestamps):
            self._signals[:, i] = self.sample()
        return self._signals

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def __repr__(self):
        return 'OUNoise(mu={}, theta={}, sigma={})'.format(self.mu, self.theta, self.sigma)
