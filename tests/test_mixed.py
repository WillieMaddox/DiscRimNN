import os
import json
import pytest
from distutils.dir_util import copy_tree
from pytest import fixture
import numpy as np
from mixsig.mixed import MixedSignal

# class TestMixedSignal:
#     def test___init__(self):
#         msig = MixedSignal()


def test_create_from_3_waves_0_noise():
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#ff0000',
        'name': 'A',
    }
    wave2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#00ff00',
        'name': 'B',
    }
    wave3_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#0000ff',
        'name': 'C',
    }
    sigs_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    batch_size = 64
    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        batch_size=batch_size,
        window_size=window_size,
        window_type='sliding',
        sequence_code='xw1_xc',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.waves) == len(sigs_coeffs)
    X, y = msig.generate()
    assert len(X) % batch_size == 0
    assert len(y) % batch_size == 0
    assert len(msig.X) % batch_size == 0
    assert len(msig.y) % batch_size == 0
    assert np.all(msig.X == X)
    assert np.all(msig.y == y)
    assert msig.X.shape == (msig.n_timestamps - window_size + 1, window_size, 1)
    assert msig.y.shape == (msig.n_timestamps - window_size + 1, len(sigs_coeffs))
    assert np.all([sig.name for sig in msig.waves] == ['A', 'B', 'C'])
    assert np.all([sig.color for sig in msig.waves] == ['#ff0000', '#00ff00', '#0000ff'])


def test_create_from_2_waves_1_noise():
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#ff0000',
        'name': 'A',
    }
    wave2_coeffs = {
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301, 'noise_type': 'pareto', 'pareto_shape': 1.5},
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#00ff00',
        'name': 'B',
    }
    wave3_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#0000ff',
        'name': 'C',
    }
    sigs_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    batch_size = 16
    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        batch_size=batch_size,
        window_size=window_size,
        window_type='sliding',
        sequence_code='xw1_xc',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.waves) == len(sigs_coeffs)
    X, y = msig.generate()
    assert len(X) % batch_size == 0
    assert len(y) % batch_size == 0
    assert len(msig.X) % batch_size == 0
    assert len(msig.y) % batch_size == 0
    assert np.all(msig.X == X)
    assert np.all(msig.y == y)
    assert msig.X.shape == (msig.n_timestamps - window_size + 1, window_size, 1)
    assert msig.y.shape == (msig.n_timestamps - window_size + 1, len(sigs_coeffs))
    assert np.all([sig.name for sig in msig.waves] == ['A', 'B', 'C'])
    assert np.all([sig.color for sig in msig.waves] == ['#ff0000', '#00ff00', '#0000ff'])


def test_create_from_1_waves_2_noise():
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#ff0000',
        'name': 'A',
    }
    wave2_coeffs = {
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301, 'noise_type': 'pareto', 'pareto_shape': 1.5},
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#00ff00',
        'name': 'B',
    }
    wave3_coeffs = {
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301, 'noise_type': 'pareto', 'pareto_shape': 1.2},
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#0000ff',
        'name': 'C',
    }
    sigs_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    batch_size = 128
    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        batch_size=batch_size,
        window_size=window_size,
        window_type='sliding',
        sequence_code='xw1_xc',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.waves) == len(sigs_coeffs)
    X, y = msig.generate()
    assert len(X) % batch_size == 0
    assert len(y) % batch_size == 0
    assert len(msig.X) % batch_size == 0
    assert len(msig.y) % batch_size == 0
    assert np.all(msig.X == X)
    assert np.all(msig.y == y)
    assert msig.X.shape == (msig.n_timestamps - window_size + 1, window_size, 1)
    assert msig.y.shape == (msig.n_timestamps - window_size + 1, len(sigs_coeffs))
    assert np.all([sig.name for sig in msig.waves] == ['A', 'B', 'C'])
    assert np.all([sig.color for sig in msig.waves] == ['#ff0000', '#00ff00', '#0000ff'])


def test_invalid_window_type():
    with pytest.raises(AssertionError):
        MixedSignal(
            [{'offset': {'mean': -0.1}}],
            window_type='boxcart',
            run_label='test'
        )


def test_invalid_sequence_code1():
    with pytest.raises(AssertionError):
        MixedSignal(
            [{'offset': {'mean': -0.1}}],
            window_type='sliding',
            sequence_code='xwh_xwc',
            run_label='test'
        )


def test_invalid_sequence_code2():
    with pytest.raises(AssertionError):
        MixedSignal(
            [{'offset': {'mean': -0.1}}],
            window_type='sliding',
            sequence_code='xw1_xwg',
            run_label='test'
        )


def test_generate_sliding():
    wave1_coeffs = {'offset': {'mean': -0.1}}
    wave2_coeffs = {'offset': {'mean': 0.0}}
    wave3_coeffs = {'offset': {'mean': 0.1}}
    sigs_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    batch_size = 1
    window_size = 0
    n_timestamps = 500

    msig_coeffs = {'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}

    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        batch_size=batch_size,
        window_size=window_size,
        window_type='sliding',
        sequence_code='xw1_xwc',
        run_label='test'
    )

    X, y = msig.generate()
    n_timestamps = msig.n_timestamps
    n_labels = len(msig.labels)
    n_samples = msig.n_samples
    n_classes = msig.n_classes
    # code_map = {'x': n_samples, 'w': window_size, 't': n_timestamps, 'c': n_classes}
    assert msig.n_classes == len(msig.waves) == len(sigs_coeffs)
    assert len(X) % batch_size == 0
    assert len(y) % batch_size == 0

    if window_size < 1 or window_size > n_timestamps:
        window_size = n_timestamps

    sequence_codes = {
        't_t': {
            'x': (n_timestamps, ),
            'y': (n_labels, )},
        't_tc': {
            'x': (n_timestamps, ),
            'y': (n_timestamps, n_classes)},
        't1_tc': {
            'x': (n_timestamps, 1),
            'y': (n_timestamps, n_classes)},
        'xw_xc': {
            'x': (n_samples, window_size),
            'y': (n_samples, n_classes)},
        'xw1_xc': {
            'x': (n_samples, window_size, 1),
            'y': (n_samples, n_classes)},
        'xw_xwc': {
            'x': (n_samples, window_size),
            'y': (n_samples, window_size, n_classes)},
        'xw1_xwc': {
            'x': (n_samples, window_size, 1),
            'y': (n_samples, window_size, n_classes)}}

    for sequence_code, shapes in sequence_codes.items():
        X, y = msig.generate_sliding(sequence_code)
        assert len(X.shape) == len(shapes['x']), print(sequence_code)
        assert len(y.shape) == len(shapes['y']), print(sequence_code)
        assert X.shape == shapes['x'], print(sequence_code)
        assert y.shape == shapes['y'], print(sequence_code)


def test_generate_boxcar():
    wave1_coeffs = {'offset': {'mean': -0.1}}
    wave2_coeffs = {'offset': {'mean': 0.0}}
    wave3_coeffs = {'offset': {'mean': 0.1}}
    sigs_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    batch_size = 1
    window_size = 0
    n_timestamps = 500

    msig_coeffs = {'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}

    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        batch_size=batch_size,
        window_size=window_size,
        window_type='boxcar',
        sequence_code='xw1_xc',
        run_label='test'
    )
    # msig.save_config()
    X, y = msig.generate()
    n_timestamps = msig.n_timestamps
    n_labels = len(msig.labels)
    n_samples = msig.n_samples
    n_classes = msig.n_classes
    assert msig.n_classes == len(msig.waves) == len(sigs_coeffs)

    assert len(X) % batch_size == 0
    assert len(y) % batch_size == 0
    assert np.all(msig.X == X)
    assert np.all(msig.y == y)

    if window_size < 1 or window_size > n_timestamps:
        window_size = n_timestamps

    assert msig.n_timestamps % window_size == 0
    assert msig.X.shape == (msig.n_timestamps // window_size, window_size, 1)
    assert msig.y.shape == (msig.n_timestamps // window_size, len(sigs_coeffs))

    sequence_codes = {
        't_t': {
            'x': (n_timestamps, ),
            'y': (n_labels, )},
        't_tc': {
            'x': (n_timestamps, ),
            'y': (n_timestamps, n_classes)},
        't1_tc': {
            'x': (n_timestamps, 1),
            'y': (n_timestamps, n_classes)},
        'xw_xc': {
            'x': (n_samples, window_size),
            'y': (n_samples, n_classes)},
        'xw1_xc': {
            'x': (n_samples, window_size, 1),
            'y': (n_samples, n_classes)},
        'xw_xwc': {
            'x': (n_samples, window_size),
            'y': (n_samples, window_size, n_classes)},
        'xw1_xwc': {
            'x': (n_samples, window_size, 1),
            'y': (n_samples, window_size, n_classes)}}

    for sequence_code, shapes in sequence_codes.items():
        X, y = msig.generate_boxcar(sequence_code)
        assert len(X.shape) == len(shapes['x']), print(sequence_code)
        assert len(y.shape) == len(shapes['y']), print(sequence_code)
        assert X.shape == shapes['x'], print(sequence_code)
        assert y.shape == shapes['y'], print(sequence_code)



@fixture
def datadir(tmpdir, request):
    """
    REF: http://www.camillescott.org/2016/07/15/travis-pytest-scipyconf/
    Fixture responsible for locating the test data directory and copying it
    into a temporary directory.
    """
    filename = request.module.__file__
    test_dir = os.path.dirname(filename)
    data_dir = os.path.join(test_dir, 'data')
    copy_tree(data_dir, str(tmpdir))

    def getter(filename, as_str=True):
        filepath = tmpdir.join(filename)
        if as_str:
            return str(filepath)
        return filepath

    return getter


def test_generate_config(datadir):
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    wave2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'B',
        'color': '#00ff00'
    }
    wave3_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'C',
        'color': '#0000ff'
    }
    sig_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    window_size = 10
    msig = MixedSignal(
        sig_coeffs,
        msig_coeffs=msig_coeffs,
        window_size=window_size,
        window_type='sliding',
        sequence_code='xw1_xc',
        run_label='test'
    )
    truth_filename = datadir('mixed_signal_config.json')
    with open(truth_filename, 'rb') as ifs:
        signal_config_truth = json.load(ifs)

    msig.save_config()
    with open(msig.config_filename, 'rb') as ifs:
        signal_config_test = json.load(ifs)

    assert signal_config_truth == signal_config_test


