import os
import json
import pytest
from distutils.dir_util import copy_tree
from pytest import fixture
import numpy as np
from mixsig.mixed import MixedSignal
from mixsig.utils import string2shape
# class TestMixedSignal:
#     def test___init__(self):
#         msig = MixedSignal()


def test_create_from_1_input_2_outputs():
    n_timestamps = 301
    features = ('x',)
    n_features = len(features)
    window_size = 10
    waves_coeffs = [{'name': 'A'}, {'name': 'B'}]

    mwave_coeffs = {
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}
    }

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    n_classes = 2
    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=window_size,
        window_type='sliding',
        sequence_type='many2one',
        run_label='test'
    )

    assert msig.n_classes == n_classes
    X, y = msig.generate()
    assert X.shape == (n_timestamps - window_size + 1, window_size, n_features)
    assert y.shape == (n_timestamps - window_size + 1, n_classes)


def test_create_from_2_input_2_outputs():
    n_timestamps = 301
    features = ('x', 'dxdt')
    n_features = len(features)
    window_size = 10
    waves_coeffs = [{'name': 'A'}, {'name': 'B'}]

    mwave_coeffs = {
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}
    }

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    n_classes = 2
    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=window_size,
        window_type='sliding',
        sequence_type='many2one',
        run_label='test'
    )

    assert msig.n_classes == n_classes
    X, y = msig.generate()
    assert X.shape == (n_timestamps - window_size + 1, window_size, n_features)
    assert y.shape == (n_timestamps - window_size + 1, n_classes)


def test_create_from_mixed_wave_123():
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#ff0000',
        'name': 'A',
    }
    wave2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#00ff00',
        'name': 'B',
    }
    wave3_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#0000ff',
        'name': 'C',
    }
    waves_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    n_timestamps = 301
    mwave_coeffs = {
        'name': 'mixed_wave',
        'amplitude': {'mean': 1, 'delta': 2},
        'frequency': {'mean': 2, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}
    }

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    n_classes = 3
    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        window_size=window_size,
        window_type='sliding',
        sequence_type='many2many',
        run_label='test'
    )

    if window_size < 1 or window_size > n_timestamps:
        window_size = n_timestamps

    assert msig.n_classes == n_classes
    X, y = msig.generate()
    assert X.shape == (msig.n_timestamps - window_size + 1, window_size, 1)
    assert y.shape == (msig.n_timestamps - window_size + 1, window_size, n_classes)
    assert np.all([sig.name for sig in msig.waves] == ['A', 'B', 'C'])
    assert np.all([sig.color for sig in msig.waves] == ['#ff0000', '#00ff00', '#0000ff'])


def test_create_from_mixed_wave_13_wave_2():
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#ff0000',
        'name': 'A',
    }
    wave2_coeffs = {
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 301, 'noise_type': 'pareto', 'pareto_shape': 1.5},
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
    waves_coeffs = [wave1_coeffs, wave3_coeffs]

    n_timestamps = 602
    mwave_coeffs = {
        'name': 'mixed_wave',
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 301}
    }

    sigs_coeffs = [mwave_coeffs, wave2_coeffs, *waves_coeffs]

    n_classes = 3
    window_size = 0
    msig = MixedSignal(
        sigs_coeffs,
        window_size=window_size,
        window_type='sliding',
        sequence_type='many2many',
        run_label='test'
    )

    if window_size < 1 or window_size > n_timestamps:
        window_size = n_timestamps

    assert msig.n_classes == n_classes
    X, y = msig.generate()
    assert X.shape == (window_size, 1)
    assert y.shape == (window_size, n_classes)
    assert np.all([sig.name for sig in msig.waves] == ['B', 'A', 'C'])
    assert np.all([sig.color for sig in msig.waves] == ['#00ff00', '#ff0000', '#0000ff'])


def test_create_from_mixed_wave_12_wave_34():
    wave1_coeffs = {
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 101, 'noise_type': 'pareto', 'pareto_shape': 1.5},
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#00ff00',
        'name': 'B',
    }
    wave2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#ff0000',
        'name': 'A',
    }
    wave3_coeffs = {
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 101, 'noise_type': 'pareto', 'pareto_shape': 1.2},
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#0000ff',
        'name': 'C',
    }
    wave4_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#0000ff',
        'name': 'D',
    }
    mwave_coeffs = {
        'amplitude': {'mean': 1, 'delta': 2},
        'frequency': {'mean': 1, 'delta': 0.5},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 101},
        'name': 'mixed_wave',
    }

    n_timestamps = 303
    sigs_coeffs = [wave1_coeffs, mwave_coeffs, wave2_coeffs, wave3_coeffs, wave4_coeffs]

    features = ('x', 'dxdt', 'd2xdt2')
    n_features = len(features)
    n_classes = 4
    window_size = 0
    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=window_size,
        window_type='sliding',
        sequence_type='many2many',
        run_label='test'
    )

    if window_size < 1 or window_size > n_timestamps:
        window_size = n_timestamps

    assert msig.n_classes == n_classes
    X, y = msig.generate()
    assert X.shape == (window_size, n_features)
    assert y.shape == (window_size, n_classes)
    assert np.all([sig.name for sig in msig.waves] == ['B', 'A', 'C', 'D'])
    assert np.all([sig.color for sig in msig.waves] == ['#00ff00', '#ff0000', '#0000ff', '#0000ff'])


def test_invalid_window_type():
    with pytest.raises(AssertionError):
        MixedSignal(
            [{'offset': {'mean': -0.1}}],
            window_type='boxcart',
            run_label='test'
        )


@pytest.mark.parametrize('n_classes', [2, 3], ids=repr)
@pytest.mark.parametrize('n_features', [1, 2], ids=repr)
@pytest.mark.parametrize('window_type', ['sliding', 'boxcar', 'random'], ids=repr)
@pytest.mark.parametrize('sequence_type', ['one2one', 'one2many', 'many2one'], ids=repr)
def test_generate_with_window_size_0_error(sequence_type, window_type, n_features, n_classes):
    n_timestamps = 301
    features = [('x',), ('x', 'dxdt'), ('x', 'dxdt', 'd2xdt2')][n_features - 1]
    waves_coeffs = [{'frequency': {'mean': 1, 'delta': 0.5}}] * n_classes

    mwave_coeffs = {
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=0,
        window_type=window_type,
        sequence_type=sequence_type,
        run_label='test'
    )

    with pytest.raises(ValueError):
        _ = msig.sequence_code


@pytest.mark.parametrize('n_classes', [2, 3], ids=repr)
@pytest.mark.parametrize('n_features', [1, 2], ids=repr)
@pytest.mark.parametrize('window_type', ['sliding', 'boxcar', 'random'], ids=repr)
@pytest.mark.parametrize('sequence_type', ['one2many', 'many2one', 'many2many'], ids=repr)
def test_generate_with_window_size_1_error(sequence_type, window_type, n_features, n_classes):
    n_timestamps = 301
    features = [('x',), ('x', 'dxdt'), ('x', 'dxdt', 'd2xdt2')][n_features - 1]
    waves_coeffs = [{'frequency': {'mean': 1, 'delta': 0.5}}] * n_classes

    mwave_coeffs = {
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=1,
        window_type=window_type,
        sequence_type=sequence_type,
        run_label='test'
    )

    with pytest.raises(ValueError):
        _ = msig.sequence_code


@pytest.mark.parametrize('sequence_type,window_size,n_features,n_classes,sequence_code', [
    ('one2one', 1, 1, 1, 't1_t1'),
    ('one2one', 1, 1, 2, 't1_tc'),
    ('one2one', 1, 2, 1, 'tf_t1'),
    ('one2one', 1, 2, 2, 'tf_tc'),
    ('many2many', 0, 1, 1, 't1_t1'),
    ('many2many', 0, 1, 2, 't1_tc'),
    ('many2many', 0, 2, 1, 'tf_t1'),
    ('many2many', 0, 2, 2, 'tf_tc'),
], ids=repr)
@pytest.mark.parametrize('window_type', ['sliding', 'boxcar', 'random'], ids=repr)
def test_generate_with_window_size_0_1(window_type, sequence_type, window_size, n_features, n_classes, sequence_code):
    n_timestamps_per_wave = 500
    n_timestamps = n_timestamps_per_wave * n_classes
    features = [('x',), ('x', 'dxdt'), ('x', 'dxdt', 'd2xdt2')][n_features - 1]
    waves_coeffs = [{'frequency': {'mean': 1, 'delta': 0.5},
                     'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps_per_wave, 'noise_type': 'pareto',
                              'pareto_shape': 1.5}}] * n_classes

    sigs_coeffs = [*waves_coeffs]

    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=window_size,
        window_type=window_type,
        sequence_type=sequence_type,
        run_label='test'
    )

    X, y = msig.generate()
    assert msig.n_classes == n_classes
    assert msig.n_timestamps == n_timestamps
    n_samples = msig.n_samples

    if window_size < 1 or window_size > n_timestamps:
        window_size = n_timestamps

    seq_bits = {
        '1': 1,
        't': n_timestamps,
        'x': n_samples,
        'w': window_size,
        'f': n_features,
        'c': n_classes,
    }

    ic, oc = sequence_code.split('_')
    in_shape = string2shape(ic, seq_bits)
    out_shape = string2shape(oc, seq_bits)
    X, y = msig.generate_sliding(sequence_code)
    assert X.shape == in_shape, print(sequence_code)
    assert y.shape == out_shape, print(sequence_code)


# @pytest.mark.parametrize('oc', [
#     't1', 'tc', 'x1', 'xc', 'xw1', 'xwc'
#     # 't', 't1', 'tc', '1t', 't11', 't1c', '1t1', '1tc',
#     # 'w', 'w1', 'wc', '1w',               '1w1', '1wc',
#     # 'x', 'x1', 'xc', 'xw', 'x11', 'x1c', 'xw1', 'xwc',
# ], ids=repr)
# @pytest.mark.parametrize('ic', [
#     't1', 'tf', 'x1', 'xf', 'xw1', 'xwf'
#     # 't', 't1', 'tf', '1t', 't11', 't1f', '1t1', '1tf',
#     # 'w', 'w1', 'wf', '1w',               '1w1', '1wf',
#     # 'x', 'x1', 'xf', 'xw', 'x11', 'x1f', 'xw1', 'xwf',
# ], ids=repr)
# def test_generate_sliding_old(ic, oc):
#     wave1_coeffs = {'offset': {'mean': -0.1}}
#     wave2_coeffs = {'offset': {'mean': 0.0}}
#     wave3_coeffs = {
#         'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 301, 'noise_type': 'pareto', 'pareto_shape': 1.5},
#         'offset': {'mean': 0.1}
#     }
#     waves_coeffs = [wave1_coeffs, wave2_coeffs]
#
#     n_classes = 3
#     features = ('x', 'dxdt')
#     window_size = 0
#     n_timestamps = 500
#
#     mwave_coeffs = {
#         'name': 'mixed_wave',
#         'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}
#
#     sigs_coeffs = [mwave_coeffs, wave3_coeffs, *waves_coeffs]
#
#     msig = MixedSignal(
#         sigs_coeffs,
#         *features,
#         window_size=window_size,
#         window_type='sliding',
#         run_label='test'
#     )
#
#     X, y = msig.generate()
#     assert msig.n_classes == n_classes
#     n_timestamps = msig.n_timestamps
#     n_samples = msig.n_samples
#     n_classes = msig.n_classes
#     n_features = len(features)
#
#     if window_size < 1 or window_size > n_timestamps:
#         window_size = n_timestamps
#
#     seq_bits = {
#         '1': 1,
#         't': n_timestamps,
#         'x': n_samples,
#         'w': window_size,
#         'f': n_features,
#         'c': n_classes,
#     }
#
#     in_shape = string2shape(ic, seq_bits)
#     out_shape = string2shape(oc, seq_bits)
#     sequence_code = '_'.join([ic, oc])
#     X, y = msig.generate_sliding(sequence_code)
#     assert X.shape == in_shape, print(sequence_code)
#     assert y.shape == out_shape, print(sequence_code)


@pytest.mark.parametrize('n_classes', [2, 3], ids=repr)
@pytest.mark.parametrize('n_features', [1, 2], ids=repr)
@pytest.mark.parametrize('window_size', [2, 5], ids=repr)
@pytest.mark.parametrize('sequence_type', ['one2one', 'one2many', 'many2one', 'many2many'], ids=repr)
def test_generate_sliding(sequence_type, window_size, n_features, n_classes):
    n_timestamps = 301
    features = [('x',), ('x', 'dxdt'), ('x', 'dxdt', 'd2xdt2')][n_features - 1]
    waves_coeffs = [{'frequency': {'mean': 1, 'delta': 0.5}}] * n_classes

    mwave_coeffs = {
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    msig = MixedSignal(
        sigs_coeffs,
        *features,
        window_size=window_size,
        window_type='sliding',
        sequence_type=sequence_type,
        run_label='test'
    )

    n_samples = n_timestamps - window_size + 1

    X, y = msig.generate()
    assert msig.sequence_type == sequence_type
    assert msig.window_size == window_size
    assert msig.n_features == n_features
    assert msig.n_classes == n_classes

    seq_bits = {
        '1': 1,
        't': n_timestamps,
        'x': n_samples,
        'w': window_size,
        'f': n_features,
        'c': n_classes,
    }

    ic, oc = msig.sequence_code.split('_')
    in_shape = string2shape(ic, seq_bits)
    out_shape = string2shape(oc, seq_bits)
    assert X.shape == in_shape, print(msig.sequence_code)
    assert y.shape == out_shape, print(msig.sequence_code)


@pytest.mark.parametrize('n_classes', [2, 3], ids=repr)
@pytest.mark.parametrize('n_features', [1, 2], ids=repr)
@pytest.mark.parametrize('batch_size', [1, 5], ids=repr)
@pytest.mark.parametrize('window_size', [2, 5], ids=repr)
@pytest.mark.parametrize('sequence_type', ['one2one', 'one2many', 'many2one', 'many2many'], ids=repr)
def test_generate_boxcar(sequence_type, window_size, batch_size, n_features, n_classes):
    features = [('x',), ('x', 'dxdt'), ('x', 'dxdt', 'd2xdt2')][n_features - 1]
    waves_coeffs = [{'frequency': {'mean': 1, 'delta': 0.5}}] * n_classes

    n_timestamps = 250

    mwave_coeffs = {
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps}}

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    msig = MixedSignal(
        sigs_coeffs,
        *features,
        batch_size=batch_size,
        window_size=window_size,
        window_type='boxcar',
        sequence_type=sequence_type,
        run_label='test'
    )
    X, y = msig.generate()
    assert msig.n_classes == n_classes
    # TODO: test separately for statefullness
    assert len(X) % batch_size == 0
    assert len(y) % batch_size == 0
    assert n_timestamps == msig.n_timestamps
    # n_labels = len(msig.labels)
    n_samples = msig.n_samples

    assert msig.n_timestamps % msig.window_size == 0
    # assert X.shape == (n_timestamps // window_size, window_size, n_features)
    # assert y.shape == (n_timestamps // window_size, window_size, n_classes)

    seq_bits = {
        '1': 1,
        't': n_timestamps,
        'x': n_samples,
        'w': window_size,
        'f': n_features,
        'c': n_classes,
    }

    ic, oc = msig.sequence_code.split('_')
    in_shape = string2shape(ic, seq_bits)
    out_shape = string2shape(oc, seq_bits)
    assert X.shape == in_shape, print(msig.sequence_code)
    assert y.shape == out_shape, print(msig.sequence_code)


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
    waves_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    mwave_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'name': 'mixed_wave',
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': 301}
    }

    sigs_coeffs = [mwave_coeffs, *waves_coeffs]

    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        window_size=window_size,
        window_type='sliding',
        run_label='test'
    )
    truth_filename = datadir('data_config.json')
    with open(truth_filename, 'rb') as ifs:
        data_config_truth = json.load(ifs)

    msig.save_config()
    with open(msig.data_config_filename, 'rb') as ifs:
        data_config_test = json.load(ifs)

    assert data_config_truth == data_config_test


def test_sequence_code_errors():

    n_timestamps = 101

    wave1_coeffs = {
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps, 'noise_type': 'pareto', 'pareto_shape': 1.5},
        'name': 'B',
    }

    sigs_coeffs = [wave1_coeffs]

    msig = MixedSignal(sigs_coeffs, window_size=1, sequence_type='many2one')
    with pytest.raises(ValueError):
        msig.generate()

    msig = MixedSignal(sigs_coeffs, window_size=1, sequence_type='many2many')
    with pytest.raises(ValueError):
        msig.generate()

    msig = MixedSignal(sigs_coeffs, window_size=1, sequence_type='one2many')
    with pytest.raises(ValueError):
        msig.generate()

    msig = MixedSignal(sigs_coeffs, ('',))
    with pytest.raises(ValueError):
        msig.generate()

    msig = MixedSignal([])
    with pytest.raises(ValueError):
        msig.generate()


def test_generate_samples():

    # examples:
    # if n_samples == 0
    # (1200, 1) -> (1200, 1)
    # if n_samples == 1
    # (1200, 1) -> (1, 1200, 1)
    # if n_samples == 32
    # (1200, 1) -> (32, 1200, 1)

    n_timestamps = 101

    wave1_coeffs = {
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps, 'noise_type': 'pareto', 'pareto_shape': 1.5},
        'phase': {'mean': 0, 'delta': 1},
        'name': 'A',
    }

    wave2_coeffs = {
        'time': {'t_min': 0, 't_max': 2, 'n_timestamps': n_timestamps, 'noise_type': 'pareto', 'pareto_shape': 1.5},
        'phase': {'mean': 0, 'delta': 1},
        'color': '#00ff00',
        'name': 'B',
    }

    sigs_coeffs = [wave1_coeffs, wave2_coeffs]

    msig = MixedSignal(
        sigs_coeffs,
        window_size=0,
        window_type='sliding',
        run_label='test'
    )
    n_features = msig.n_features
    n_classes = msig.n_classes

    with pytest.raises(ValueError):
        msig.generate_samples(0)

    n_samples = 2
    X, y = msig.generate_samples(n_samples)
    assert X.shape == (n_samples, 2*n_timestamps, n_features)
    assert y.shape == (n_samples, 2*n_timestamps, n_classes)

    n_samples = 10
    X, y = msig.generate_samples(n_samples)
    assert X.shape == (n_samples, 2*n_timestamps, n_features)
    assert y.shape == (n_samples, 2*n_timestamps, n_classes)

