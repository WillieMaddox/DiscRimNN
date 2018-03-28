import pytest
from hypothesis import given
from hypothesis import example
from hypothesis import strategies as st

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
    wave = Wave()
    assert wave.amplitude == 0
    assert wave.period == 0
    assert wave.offset == 0
    assert wave.phase == 0


def test_wave_mean():
    params = {
        'amplitude': {'mean': 1, 'delta': 0},
        'period': {'mean': 5, 'delta': 0},
        'offset': {'mean': -3, 'delta': 0},
        'phase': {'mean': 2, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    wave = Wave(**params)
    assert wave.amplitude == 1
    assert wave.period == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.name == 'A'
    assert wave.color == '#ff0000'


