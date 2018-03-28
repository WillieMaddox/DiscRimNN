import numpy as np
import pytest
from hypothesis import given
from hypothesis import example
from hypothesis import strategies as st
from mixsig.timesequence import TimeSequence
from mixsig.noise import NormalNoise
from mixsig.noise import UniformNoise
from mixsig.waves import WaveProperty
from mixsig.waves import Wave

waveproperty = st.builds(
    WaveProperty,
    mean=st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_infinity=False, allow_nan=False)
    ),
    delta=st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_infinity=False, allow_nan=False)
    )
)


@given(waveproperty)
def test_waveproperty_output_is_float(wp):
    assert isinstance(wp(), float)
    assert isinstance(wp.generate(), float)


@given(waveproperty)
@example(WaveProperty(mean=5.0, delta=None))
@example(WaveProperty(mean=5, delta=0.0))
@example(WaveProperty(mean=5, delta=0))
def test_waveproperty_generator(wp):
    value1 = wp()
    value2 = wp.generate()
    if wp.delta == 0.0 or wp.delta == 0:
        assert value1 == value2, f'{wp.mean} {wp.delta}'


def test_wave_default_kwargs():
    ts = TimeSequence(0.0, 50.0, 201)
    wave = Wave(ts)
    assert wave.amplitude == 0
    assert wave.period == 0
    assert wave.offset == 0
    assert wave.phase == 0


def test_wave_mean():
    ts = TimeSequence(0.0, 50.0, 201)
    params = {
        'amplitude': {'mean': 1, 'delta': 0},
        'period': {'mean': 5, 'delta': 0},
        'offset': {'mean': -3, 'delta': 0},
        'phase': {'mean': 2, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    wave = Wave(ts, **params)
    assert np.all(wave.sample == wave())
    assert wave.amplitude == 1
    assert wave.period == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.name == 'A'
    assert wave.color == '#ff0000'
    wave.generate(amplitude=1, period=1, offset=1, phase=1)
    assert wave.amplitude == 1
    assert wave.period == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.noise == 0


def test_wave_with_delayed_size():
    ts = TimeSequence(0.0, 50.0, 201)
    params = {
        'amplitude': {'mean': 1},
        'period': {'mean': 1},
    }
    wave = Wave(ts, **params)
    w0 = wave.generate()
    assert len(w0) == len(ts)


def test_wave_with_no_timesequence_arg():
    params = {
        'amplitude': {'mean': 1},
        'period': {'mean': 1},
    }
    with pytest.raises(TypeError):
        Wave(**params)


def test_wave_with_uniform_noise():
    ts = TimeSequence(0.0, 50.0, 201)
    params = {
        'amplitude': {'mean': 1},
        'period': {'mean': 1},
        'noise': {'uniform': {'n_timestamps': len(ts), 'mu': 0.0, 'delta': 0.5}},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert len(wave.noise) == len(ts)


def test_wave_with_normal_noise():
    ts = TimeSequence(0.0, 50.0, 201)
    params = {
        'amplitude': {'mean': 1},
        'period': {'mean': 1},
        'noise': {'normal': {'n_timestamps': len(ts), 'mu': 0.0, 'sigma': 0.5}},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert len(wave.noise) == len(ts)


def test_wave_with_no_noise():
    ts = TimeSequence(0.0, 50.0, 201)
    params = {
        'amplitude': {'mean': 1},
        'period': {'mean': 1},
    }
    wave = Wave(ts, **params)
    wave.generate()
    assert wave.noise == 0
