import pytest
from hypothesis import given
from hypothesis import example
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from mixsig.noise import OUNoise


st_ounoise_kwargs = st.fixed_dictionaries(
    dict(
        n_signals=st.integers(min_value=1, max_value=20),
        # n_timestamps=st.one_of(
        #     st.none(),
        #     st.integers(min_value=2, max_value=20000)
        # ),
        n_timestamps=st.integers(min_value=2, max_value=20000),
        mu=st.one_of(
            st.none(),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            # arrays(float, ())
        ),
        theta=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        sigma=st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
    )
)


@given(st_ounoise_kwargs)
@example(kwargs={'n_signals': 1, 'n_timestamps': 30, 'mu': 0.0})
@example(kwargs={'n_signals': 2, 'n_timestamps': 30, 'mu': 0.0})
def test_ou_noise_inputs(kwargs):
    noise = OUNoise(**kwargs)
    assert noise().shape == (noise.n_signals, noise.n_timestamps)
    assert noise.signals.shape == (noise.n_signals, noise.n_timestamps)


@given(st_ounoise_kwargs)
@example(kwargs={'n_signals': 1, 'n_timestamps': 30, 'mu': None})
@example(kwargs={'n_signals': 2, 'n_timestamps': 30, 'mu': 0})
@example(kwargs={'n_signals': 10, 'n_timestamps': 30, 'mu': 5})
@example(kwargs={'n_signals': 1, 'n_timestamps': 30, 'mu': np.random.randn(1)})
@example(kwargs={'n_signals': 2, 'n_timestamps': 30, 'mu': np.random.randn(2)})
@example(kwargs={'n_signals': 10, 'n_timestamps': 30, 'mu': np.random.randn(10)})
def test_ou_noise_mu_as_numpy_array(kwargs):
    noise = OUNoise(**kwargs)
    assert np.all(noise.state == noise.mu)
    noise.generate(n_timestamps=50)
    if noise.sigma > 0:
        assert np.all(noise.state != noise.mu)
    else:
        assert np.all(noise.state == noise.mu)
    noise.reset()
    assert np.all(noise.state == noise.mu)


ounoise = st.builds(
    OUNoise,
    n_signals=st.integers(min_value=1, max_value=20),
    n_timestamps=st.integers(min_value=2, max_value=20000),
    mu=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        # arrays(float, ())
    ),
    theta=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    sigma=st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
)


@given(ounoise)
@example(OUNoise(n_signals=1, n_timestamps=30, mu=0.0))
@example(OUNoise(n_signals=2, n_timestamps=30, mu=np.random.randn(2)))
@example(OUNoise(n_signals=5, n_timestamps=30, mu=0.0))
def test_ou_noise_generator(noise):
    assert noise.signals.shape == (noise.n_signals, noise.n_timestamps)
    noise.generate(n_timestamps=50)
    assert noise.signals.shape == (noise.n_signals, noise.n_timestamps)


@given(st_ounoise_kwargs)
@example(kwargs={'n_signals': 1, 'n_timestamps': 30, 'mu': 0.0})
@example(kwargs={'n_signals': 2, 'n_timestamps': 30, 'mu': 0.0})
def test_ou_noise_reset(kwargs):
    noise = OUNoise(**kwargs)
    assert np.all(noise.state == noise.mu)
    noise.generate(n_timestamps=50, reset=True)
    if noise.sigma > 0:
        assert np.all(noise.state != noise.mu)
    else:
        assert np.all(noise.state == noise.mu)
    noise.reset()
    assert np.all(noise.state == noise.mu)
