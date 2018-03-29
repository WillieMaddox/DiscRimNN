import pytest
from hypothesis import given
from hypothesis import example
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from mixsig.noise import NormalNoise
from mixsig.noise import UniformNoise
from mixsig.noise import NoNoise
from mixsig.noise import OUNoise

def test_normal_noise():
    noise = NormalNoise(n_timestamps=201, sigma=0.5)
    assert len(noise) == 201


def test_normal_noise_with_delayed_size():
    noise = NormalNoise(sigma=0.5)
    noise.generate(n_timestamps=201)
    assert len(noise) == 201


def test_normal_noise_with_no_size():
    noise = NormalNoise(sigma=0.5)
    with pytest.raises(AttributeError):
        noise()


def test_normal_noise_repr():
    noise = NormalNoise(sigma=0.5)
    assert noise.__repr__() == 'NormalNoise(mu=0, sigma=0.5)'


def test_uniform_noise():
    noise = UniformNoise(n_timestamps=201, delta=0.5)
    assert len(noise) == 201


def test_uniform_noise_with_delayed_size():
    noise = UniformNoise(delta=0.5)
    noise.generate(n_timestamps=201)
    assert len(noise) == 201


def test_uniform_noise_with_no_size():
    noise = UniformNoise(delta=0.5)
    with pytest.raises(AttributeError):
        noise()


def test_uniform_noise_repr():
    noise = UniformNoise(mu=2, delta=1)
    assert noise.__repr__() == 'UniformNoise(lo=1, hi=3)'


def test_no_noise_repr():
    noise = NoNoise()
    assert noise.__repr__() == 'NoNoise()'


st_ounoise_kwargs = st.fixed_dictionaries(
    dict(
        mu=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        theta=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        sigma=st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
    )
)


@given(st.integers(min_value=2, max_value=20000), st_ounoise_kwargs)
@example(n_timestamps=30, kwargs={'mu': 0.0})
@example(n_timestamps=30, kwargs={'mu': [0.0, 0.0]})
def test_ou_noise_inputs(n_timestamps, kwargs):
    noise = OUNoise(n_timestamps, **kwargs)
    assert noise().shape == (noise.n_signals, noise.n_timestamps)
    assert noise.signals.shape == (noise.n_signals, noise.n_timestamps)


@given(st.integers(min_value=2, max_value=20000), st_ounoise_kwargs)
@example(n_timestamps=30, kwargs={'mu': 5})
@example(n_timestamps=30, kwargs={'mu': [0, 0]})
@example(n_timestamps=30, kwargs={'mu': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]})
@example(n_timestamps=30, kwargs={'mu': np.random.randn(1)})
@example(n_timestamps=30, kwargs={'mu': np.random.randn(2)})
@example(n_timestamps=30, kwargs={'mu': np.random.randn(10)})
def test_ou_noise_mu_as_numpy_array(n_timestamps, kwargs):
    noise = OUNoise(n_timestamps, **kwargs)
    assert np.all(noise.state == noise.mu)
    noise.generate()
    if noise.sigma > 0:
        assert np.all(noise.state != noise.mu)
    else:
        assert np.all(noise.state == noise.mu)
    noise.reset()
    assert np.all(noise.state == noise.mu)


ounoise = st.builds(
    OUNoise,
    n_timestamps=st.integers(min_value=2, max_value=20000),
    mu=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    theta=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    sigma=st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
)


@given(ounoise)
@example(OUNoise(30, mu=0.0))
@example(OUNoise(30, mu=np.random.randn(2)))
@example(OUNoise(30, mu=[0.0, 0.0, 0.0, 0.1, 0.5]))
def test_ou_noise_generator(noise):
    assert noise.signals.shape == (noise.n_signals, noise.n_timestamps)
    noise.generate()
    assert noise.signals.shape == (noise.n_signals, noise.n_timestamps)


def test_ou_noise_length():
    noise = OUNoise(30, mu=0.5)
    assert len(noise) == 1
    noise = OUNoise(30, mu=[0.5, 1.3])
    assert len(noise) == 2
    noise = OUNoise(30, mu=[0.5, 1.3, 5.6])
    assert len(noise) == 3


@given(st.integers(min_value=2, max_value=20000), st_ounoise_kwargs)
@example(n_timestamps=30, kwargs={'mu': 0.0})
@example(n_timestamps=30, kwargs={'mu': [0.0, 0.5]})
def test_ou_noise_reset(n_timestamps, kwargs):
    noise = OUNoise(n_timestamps, **kwargs)
    assert np.all(noise.state == noise.mu)
    noise.generate(reset=True)
    if noise.sigma > 0:
        assert np.all(noise.state != noise.mu)
    else:
        assert np.all(noise.state == noise.mu)
    noise.reset()
    assert np.all(noise.state == noise.mu)


def test_ou_noise_repr():
    noise = OUNoise(30, mu=0.5, theta=0.15, sigma=0.5)
    assert noise.__repr__() == 'OUNoise(mu=[ 0.5], theta=0.15, sigma=0.5)'



