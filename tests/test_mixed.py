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

    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        window_size=window_size,
        window_method='sliding',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.signals) == len(sigs_coeffs)
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig.inputs.shape == (msig_coeffs['time']['n_timestamps'] - window_size + 1, window_size, 1)
    assert msig.labels.shape == (msig_coeffs['time']['n_timestamps'] - window_size + 1, len(sigs_coeffs))
    assert np.all(msig.signal_names == ['A', 'B', 'C'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00', '#0000ff'])


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
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#00ff00',
        'name': 'B',
    }
    sigs_coeffs = [wave1_coeffs, wave2_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        window_size=window_size,
        window_method='sliding',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.signals) == len(sigs_coeffs)
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig.inputs.shape == (msig_coeffs['time']['n_timestamps'] - window_size + 1, window_size, 1)
    assert msig.labels.shape == (msig_coeffs['time']['n_timestamps'] - window_size + 1, len(sigs_coeffs))
    assert np.all(msig.signal_names == ['A', 'B'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00'])


def test_create_from_1_waves_2_noise():
    wave1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'frequency': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'color': '#ff0000',
        'name': 'A',
    }
    sigs_coeffs = [wave1_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    window_size = 10
    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        window_size=window_size,
        window_method='sliding',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.signals) == len(sigs_coeffs)
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig.inputs.shape == (msig_coeffs['time']['n_timestamps'] - window_size + 1, window_size, 1)
    assert msig.labels.shape == (msig_coeffs['time']['n_timestamps'] - window_size + 1, len(sigs_coeffs))
    assert np.all(msig.signal_names == ['A'])
    assert np.all(msig.signal_colors == ['#ff0000'])


def test_create_from_3_waves_boxcar():
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
    sigs_coeffs = [wave1_coeffs, wave2_coeffs, wave3_coeffs]

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 75, 'n_timestamps': 301}
    }

    window_size = 10
    with pytest.raises(AssertionError):
        MixedSignal(
            sigs_coeffs,
            msig_coeffs=msig_coeffs,
            window_size=window_size,
            window_method='boxcar',
            run_label='test'
        )

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'frequency': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': 1},
        'time': {'t_min': 0, 't_max': 150, 'n_timestamps': 801}
    }
    window_size = 9

    with pytest.raises(AssertionError):
        MixedSignal(
            sigs_coeffs,
            msig_coeffs=msig_coeffs,
            window_size=window_size,
            window_method='boxcrap',
            run_label='test'
        )

    msig = MixedSignal(
        sigs_coeffs,
        msig_coeffs=msig_coeffs,
        window_size=window_size,
        window_method='boxcar',
        run_label='test'
    )
    # msig.save_config()
    assert len(msig.signals) == len(sigs_coeffs)
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig_coeffs['time']['n_timestamps'] % window_size == 0
    assert msig.inputs.shape == (msig_coeffs['time']['n_timestamps'] / window_size, window_size, 1)
    assert msig.labels.shape == (msig_coeffs['time']['n_timestamps'] / window_size, len(sigs_coeffs))
    assert np.all(msig.signal_names == ['A', 'B', 'C'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00', '#0000ff'])


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
        window_method='sliding',
        run_label='test'
    )
    truth_filename = datadir('mixed_signal_config.json')
    with open(truth_filename, 'rb') as ifs:
        signal_config_truth = json.load(ifs)

    msig.save_config()
    with open(msig.config_filename, 'rb') as ifs:
        signal_config_test = json.load(ifs)

    assert signal_config_truth == signal_config_test


def test_generate_sliding():
    assert True
