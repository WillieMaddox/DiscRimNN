import pytest
from math import isclose
import numpy as np
from hypothesis import given
from hypothesis import example
from hypothesis import strategies as st
from mixsig.utils import timesequence_generator
from mixsig.waves import WaveProperty
from mixsig.waves import Amplitude
from mixsig.waves import Frequency
from mixsig.waves import Offset
from mixsig.waves import Phase
from mixsig.waves import Wave

waveproperty = st.builds(
    WaveProperty,
    mean=st.one_of(
        st.none(),
        st.integers(min_value=1e-13, max_value=1e13),
        st.floats(min_value=1e-13, max_value=1e13, allow_infinity=False, allow_nan=False)
    ),
    delta=st.one_of(
        st.none(),
        st.integers(min_value=0.0, max_value=1e5),
        st.floats(min_value=0.0, max_value=1e5, allow_infinity=False, allow_nan=False)
    )
)


@given(waveproperty)
def test_waveproperty_output_is_float(wp):
    assert isinstance(wp, float)
    assert isinstance(wp(), float)
    # assert isinstance(wp.generate(), float)


@given(waveproperty)
def test_waveproperty_generator_a(wp):
    if isclose(wp.delta, 0):
        assert wp == wp(), f'{wp.mean} {wp.delta}'
    else:
        assert wp != wp(), f'{wp.mean} {wp.delta}'


@pytest.mark.parametrize('mean,delta', [
    (5, None),
    (5, 0.0),
    (5, 0),
    (5, 1),
    (5, 1.0),
    (5.0, None),
    (5.0, 0.0),
    (5.0, 0),
    (5.0, 1),
    (5.0, 1.0),
    (None, None),
    (None, 0.0),
    (None, 0),
    (None, 1),
    (None, 1.0),
], ids=repr)
@pytest.mark.parametrize('Wp', [WaveProperty, Amplitude, Frequency, Offset, Phase], ids=repr)
def test_waveproperty_generator_b(Wp, mean, delta):
    wp = Wp(mean, delta)
    assert wp == wp.mean
    if isclose(wp.delta, 0):
        assert wp == wp(), f'{wp.mean} {wp.delta}'
    else:
        assert wp != wp(), f'{wp.mean} {wp.delta}'


def test_wave_default_kwargs():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    wave = Wave(ts)
    assert wave.amplitude == 1
    assert wave.frequency == 1
    assert wave.offset == 0
    assert wave.phase == 0
    assert wave.__repr__() == 'Wave(amplitude=1.0, frequency=1.0, offset=0.0, phase=0.0)'


def test_wave_mean():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1, 'delta': 0},
        'frequency': {'mean': 5, 'delta': 0},
        'offset': {'mean': -3, 'delta': 0},
        'phase': {'mean': 2, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    wave = Wave(ts, **params)
    assert np.all(wave.sample == wave())
    assert wave.amplitude == 1
    assert wave.frequency == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.name == 'A'
    assert wave.color == '#ff0000'
    wave.generate(amplitude=1, frequency=1, offset=1, phase=1)
    assert wave.amplitude == 1
    assert wave.frequency == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.signal_noise == 0


def test_wave_with_delayed_size():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 2},
        'frequency': {'mean': 2},
    }
    wave = Wave(ts, **params)
    w0 = wave.generate()
    assert len(w0) == len(ts)


def test_wave_with_no_timesequence_arg():
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    with pytest.raises(TypeError):
        Wave(**params)


def test_wave_with_uniform_noise():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
        'noise': {'uniform': {'mu': 0.0, 'delta': 0.5}},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert len(wave.signal_noise) == len(ts)


def test_wave_with_normal_noise():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
        'noise': {'normal': {'mu': 0.0, 'sigma': 0.5}},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert len(wave.signal_noise) == len(ts)


def test_wave_with_no_noise():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert wave.signal_noise == 0


@pytest.mark.parametrize('n_min,n_max,noise_type,delta,pareto_shape', [
    (None, 200, None, 0.5, None),
    (None, 200, None, None, None),
    (100, 200, None, None, None),
    (None, 200, 'jitter', 0.5, None),
    (None, 200, 'jitter', None, None),
    (100, 200, 'jitter', None, None),
    (None, 200, 'pareto', None, 3),
    (None, 200, 'pareto', None, None),
    (100, 200, 'pareto', None, None),
], ids=repr)
def test_wave_with_timestamp_noise(n_min, n_max, noise_type, delta, pareto_shape):
    sequence_generator = timesequence_generator(
        t_min=0.0,
        t_max=50.0,
        n_max=n_max,
        n_min=n_min,
        noise_type=noise_type,
        delta=delta,
        pareto_shape=pareto_shape)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert len(wave.sample) == len(ts)
