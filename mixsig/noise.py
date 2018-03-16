import numpy as np

class BaseNoise(object):
    def reset(self):
        pass


class UniformNoise(BaseNoise):
    def __init__(self, mu=0.0, sigma=0.01):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'UniformNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NormalNoise(BaseNoise):
    def __init__(self, mu=0.0, sigma=0.01):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OUNoise(BaseNoise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        if isinstance(mu, np.ndarray):
            assert mu.shape[0] == size
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
