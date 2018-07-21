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
from mixsig.waves import MixedWave

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
def test_waveproperty_generator_a(wp):
    wp0 = wp()
    wp1 = wp()
    if isclose(wp.delta, 0, abs_tol=1e-9):
        assert isclose(wp0, wp1), f'{wp.value} {wp.delta}'
        assert isclose(wp1, wp0), f'{wp.value} {wp.delta}'
    else:
        assert wp0 != wp1, f'{wp.value} {wp.delta}'
        assert wp1 != wp0, f'{wp.value} {wp.delta}'


@pytest.mark.parametrize('delta', [None, 0.0, 0, 1, 1.0], ids=repr)
@pytest.mark.parametrize('value', [5, 5.0, None], ids=repr)
@pytest.mark.parametrize('Wp', [WaveProperty, Amplitude, Frequency, Offset, Phase], ids=repr)
def test_waveproperty_generator_b(Wp, value, delta):
    wp = Wp(value, delta)
    assert isinstance(wp, Wp)
    assert isinstance(wp(), float)
    assert wp == wp.value
    wp0 = wp()
    wp1 = wp()

    if isclose(wp.delta, 0):
        assert wp0 == wp1
        assert wp1 == wp0
    else:
        assert id(wp0) != id(wp1)
        assert wp0 != wp1
        assert wp1 != wp0


@pytest.mark.parametrize('Wp,key', [
    (WaveProperty, 'dummy'),
    (Amplitude, 'amplitude'),
    (Frequency, 'frequency'),
    (Offset, 'offset'),
    (Phase, 'phase'),
], ids=repr)
def test_waveproperty_call(Wp, key):
    wp = Wp()
    kwargs = {'dummy': 0, 'amplitude': 6.0, 'frequency': 7.0, 'offset': 8.0, 'phase': 9.0}
    wp0 = wp()
    assert wp0 == wp

    wp1 = wp(**kwargs)
    if key in ('amplitude', 'frequency'):
        assert isclose(wp * kwargs[key], wp1)
    else:
        assert isclose(wp + kwargs[key], wp1)


@pytest.mark.parametrize('Wp1', [WaveProperty, Amplitude, Frequency, Offset, Phase], ids=repr)
@pytest.mark.parametrize('Wp2', [WaveProperty, Amplitude, Frequency, Offset, Phase], ids=repr)
def test_waveproperty_dunders(Wp1, Wp2):
    mean1 = 4
    mean2 = 8
    wp1 = Wp1(mean1)
    wp2 = Wp2(mean2)
    if not isinstance(wp1, Wp2) and not isinstance(wp2, Wp1):
        with pytest.raises(TypeError):
            _ = wp1 + wp2
        with pytest.raises(TypeError):
            _ = wp1 * wp2
        with pytest.raises(TypeError):
            _ = wp1 == wp2
        with pytest.raises(TypeError):
            _ = wp1 != wp2
        with pytest.raises(TypeError):
            _ = wp1 < wp2
        with pytest.raises(TypeError):
            _ = wp1 > wp2
    elif isinstance(wp1, Wp2) and isinstance(wp2, Wp1):
        assert wp1 + wp2 == wp1.value + wp2.value
        assert wp1 * wp2 == wp1.value * wp2.value
        assert 4 + wp1 == wp2
        assert 2 * wp1 == wp2
        assert wp1 == Wp2(mean1)
        assert wp1 != wp2
        assert wp1 != mean2
        assert 2 < wp1 < wp2 < 10 and 10 > wp2 > wp1 > 2
        with pytest.raises(TypeError):
            _ = wp1 + 'abc'
        with pytest.raises(TypeError):
            _ = wp1 * 'abc'
        with pytest.raises(TypeError):
            _ = 'abc' + wp1
        with pytest.raises(TypeError):
            _ = 'abc' * wp1
    else:
        assert mean2 + wp1 == wp2 + mean1
        assert mean2 * wp1 == wp2 * mean1


def test_wave_default_kwargs():
    wave = Wave()
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
    wave = Wave(**params)
    assert wave.amplitude == 1
    assert wave.frequency == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.name == 'A'
    assert wave.color == '#ff0000'
    wave.generate(ts, amplitude=1, frequency=1, offset=1, phase=1)
    assert wave.amplitude == 1
    assert wave.frequency == 5
    assert wave.offset == -3
    assert wave.phase == 2
    assert wave.noise == 0


def test_wave_with_delayed_size():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 2},
        'frequency': {'mean': 2},
    }
    wave = Wave(**params)
    wave.generate(ts)
    assert len(wave.sample) == len(ts)


def test_wave_generate_with_no_timesequence_arg():
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    with pytest.raises(TypeError):
        wave = Wave(**params)
        wave.generate()


def test_wave_with_uniform_noise():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
        'noise': {'uniform': {'mu': 0.0, 'delta': 0.5}},
    }
    wave = Wave(**params)
    wave.generate(ts)
    assert len(wave.noise) == len(ts)


def test_wave_with_normal_noise():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
        'noise': {'normal': {'mu': 0.0, 'sigma': 0.5}},
    }
    wave = Wave(**params)
    wave.generate(ts)
    assert len(wave.noise) == len(ts)


def test_wave_with_no_noise():
    sequence_generator = timesequence_generator(t_min=0.0, t_max=50.0, n_max=201)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    wave = Wave(**params)
    wave.generate(ts)
    assert wave.noise == 0


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
    wave = Wave(**params)
    wave.generate(ts)
    assert len(wave.sample) == len(ts)


def test_wave_with_1_feature():
    n_timestamps = 201
    features = ('x',)
    sequence_generator = timesequence_generator(t_min=0.0, t_max=2.0, n_max=n_timestamps)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    wave = Wave(*features, **params)
    wave.generate(ts)
    assert wave.inputs.shape == (n_timestamps, len(features))


def test_wave_with_2_features():
    n_timestamps = 201
    features = ('x', 'dxdt')
    sequence_generator = timesequence_generator(t_min=0.0, t_max=2.0, n_max=n_timestamps)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    wave = Wave(*features, **params)
    wave.generate(ts)
    assert wave.inputs.shape == (n_timestamps, len(features))


def test_wave_with_3_features():
    n_timestamps = 201
    features = ('d0xdt0', 'd3xdt3', 'd2xdt2')
    sequence_generator = timesequence_generator(t_min=0.0, t_max=2.0, n_max=n_timestamps)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    wave = Wave(*features, **params)
    wave.generate(ts)
    assert wave.inputs.shape == (n_timestamps, len(features))


def test_wave_raises_invalid_feature():
    n_timestamps = 201
    features = ('dydz2',)
    sequence_generator = timesequence_generator(t_min=0.0, t_max=2.0, n_max=n_timestamps)
    ts = sequence_generator()
    params = {
        'amplitude': {'mean': 1},
        'frequency': {'mean': 1},
    }
    with pytest.raises(ValueError):
        wave = Wave(*features, **params)
        wave.generate(ts)
        _ = wave.inputs


@pytest.mark.parametrize('n_classes', [1, 2, 3], ids=repr)
@pytest.mark.parametrize('n_features', [1, 2, 3], ids=repr)
def test_mixedwave_features_and_classes(n_features, n_classes):
    n_timestamps = 301
    features = [('x',), ('x', 'dxdt'), ('x', 'dxdt', 'd2xdt2')][n_features - 1]
    # waves_coeffs = [{'frequency': {'mean': 1, 'delta': 0.5}}] * n_classes
    mwave_coeffs = {
        # 'waves_coeffs': waves_coeffs,
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}
    }
    # sigs_coeffs = [mwave_coeffs, *waves_coeffs]
    mwave = MixedWave(
        classes=list(range(n_classes)),
        mwave_coeffs=mwave_coeffs,
    )

    # assert len(mwave.waves) == len(waves_coeffs)
    mwave.generate()
    assert mwave.labels.shape == (n_timestamps,)
    # assert mwave.one_hots.shape == (n_timestamps, n_classes)

    # i_timestamp = np.random.randint(n_timestamps)
    # i_label = mwave.labels[i_timestamp]
    # assert mwave.one_hots[i_timestamp][i_label] == 1
