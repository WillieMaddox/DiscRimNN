import numpy as np


class UniformNoise:
    def __init__(self, mu=0.0, delta=0.5):
        self.mu = mu
        self.delta = delta
        self.low = self.mu - self.delta
        self.hi = self.mu + self.delta

    def __call__(self):
        return np.random.uniform(self.low, self.hi)

    def __repr__(self):
        return 'UniformNoise(low={}, hi={})'.format(self.low, self.hi)


class NormalNoise:
    def __init__(self, mu=0.0, sigma=0.01):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, n_timestamps, n_signals=None, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.n_timestamps = n_timestamps
        self.n_signals = n_signals
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = mu
        self._signals = None

    def __call__(self):
        return self.signals

    @property
    def signals(self):
        if self._signals is None:
            self._signals = self.generate()
        return self._signals

    def generate(self, n_signals=1):
        signals = np.empty((n_signals, self.n_timestamps))
        for i in range(self.n_timestamps):
            signals[:, i] = self.sample()
        self._signals = signals
        return self._signals

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn()
        self.state = x + dx
        return self.state

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

