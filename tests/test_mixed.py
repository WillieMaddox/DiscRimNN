import pytest
import numpy as np
from mixsig.mixed import MixedSignal

# class TestMixedSignal:
#     def test___init__(self):
#         msig = MixedSignal()


def test_create_from_3_waves_0_noise():
    sig1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    sig2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'B',
        'color': '#00ff00'
    }
    sig3_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'C',
        'color': '#0000ff'
    }
    sig_coeffs = {'waves': [sig1_coeffs, sig2_coeffs, sig3_coeffs]}

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'period': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': np.pi},
    }

    time_coeffs = {'start': 0, 'stop': 75, 'n_timestamps': 301, 'delta': 0}
    n_timesteps = 10
    msig = MixedSignal(
        time_coeffs,
        sig_coeffs,
        msig_coeffs=msig_coeffs,
        n_timesteps=n_timesteps,
        method='sliding',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.signals) == len(sig_coeffs['waves'])
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig.inputs.shape == (time_coeffs['n_timestamps'] - n_timesteps + 1, n_timesteps, 1)
    assert msig.labels.shape == (time_coeffs['n_timestamps'] - n_timesteps + 1, len(sig_coeffs['waves']))
    assert np.all(msig.signal_names == ['A', 'B', 'C'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00', '#0000ff'])


def test_create_from_2_waves_1_noise():
    sig1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    sig2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'B',
        'color': '#00ff00'
    }
    sig3_coeffs = {
        'mu': 1.0,
        'theta': 0.5,
        'sigma': 0.01,
        'name': 'C',
        'color': '#0000ff'
    }
    sig_coeffs = {
        'waves': [sig1_coeffs, sig2_coeffs],
        'noise': sig3_coeffs
    }

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'period': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': np.pi},
    }

    time_coeffs = {'start': 0, 'stop': 75, 'n_timestamps': 301, 'delta': 0}
    n_timesteps = 10
    msig = MixedSignal(
        time_coeffs,
        sig_coeffs,
        msig_coeffs=msig_coeffs,
        n_timesteps=n_timesteps,
        method='sliding',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.signals) == len(sig_coeffs['waves']) + 1
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig.inputs.shape == (time_coeffs['n_timestamps'] - n_timesteps + 1, n_timesteps, 1)
    assert msig.labels.shape == (time_coeffs['n_timestamps'] - n_timesteps + 1, len(sig_coeffs['waves']) + 1)
    assert np.all(msig.signal_names == ['A', 'B', 'C'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00', '#0000ff'])


def test_create_from_1_waves_2_noise():
    sig1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    sig2_coeffs = {
        'mu': 0.0,
        'theta': 0.25,
        'sigma': 0.1,
        'name': 'B',
        'color': '#00ff00'
    }
    sig3_coeffs = {
        'mu': [0.0, 1.0],
        'theta': [0.25, 0.5],
        'sigma': [0.1, 0.01],
        'name': ['B', 'C'],
        'color': ['#00ff00', '#0000ff']
    }
    sig_coeffs = {
        'waves': [sig1_coeffs],
        'noise': sig3_coeffs
    }

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'period': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': np.pi},
    }

    time_coeffs = {'start': 0, 'stop': 75, 'n_timestamps': 301, 'delta': 0}
    n_timesteps = 10
    msig = MixedSignal(
        time_coeffs,
        sig_coeffs,
        msig_coeffs=msig_coeffs,
        n_timesteps=n_timesteps,
        method='sliding',
        run_label='test'
    )

    # msig.save_config()
    assert len(msig.signals) == len(sig_coeffs['waves']) + 2
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert msig.inputs.shape == (time_coeffs['n_timestamps'] - n_timesteps + 1, n_timesteps, 1)
    assert msig.labels.shape == (time_coeffs['n_timestamps'] - n_timesteps + 1, len(sig_coeffs['waves']) + 2)
    assert np.all(msig.signal_names == ['A', 'B', 'C'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00', '#0000ff'])


def test_create_from_2_waves_boxcar():
    sig1_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': -0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'A',
        'color': '#ff0000'
    }
    sig2_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.0, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'B',
        'color': '#00ff00'
    }
    sig3_coeffs = {
        'amplitude': {'mean': 1.0, 'delta': 0},
        'period': {'mean': 1, 'delta': 0},
        'offset': {'mean': 0.1, 'delta': 0},
        'phase': {'mean': 0, 'delta': 0},
        'name': 'C',
        'color': '#0000ff'
    }
    sig_coeffs = {'waves': [sig1_coeffs, sig2_coeffs, sig3_coeffs]}

    msig_coeffs = {
        'amplitude': {'mean': 10, 'delta': 2},
        'period': {'mean': 25, 'delta': 0},
        'offset': {'mean': 1, 'delta': 5},
        'phase': {'mean': 0, 'delta': np.pi},
    }

    time_coeffs = {'start': 0, 'stop': 75, 'n_timestamps': 301, 'delta': 0}
    n_timesteps = 10
    with pytest.raises(AssertionError):
        MixedSignal(
            time_coeffs,
            sig_coeffs,
            msig_coeffs=msig_coeffs,
            n_timesteps=n_timesteps,
            method='boxcar',
            run_label='test'
        )

    time_coeffs = {'start': 0, 'stop': 150, 'n_timestamps': 801, 'delta': 0}
    n_timesteps = 9

    with pytest.raises(ValueError):
        msig = MixedSignal(
            time_coeffs,
            sig_coeffs,
            msig_coeffs=msig_coeffs,
            n_timesteps=n_timesteps,
            method='boxcrap',
            run_label='test'
        )
        msig.generate()

    msig = MixedSignal(
        time_coeffs,
        sig_coeffs,
        msig_coeffs=msig_coeffs,
        n_timesteps=n_timesteps,
        method='boxcar',
        run_label='test'
    )
    # msig.save_config()
    assert len(msig.signals) == len(sig_coeffs['waves'])
    X, y = msig.generate()
    assert np.all(msig.inputs == X)
    assert np.all(msig.labels == y)
    assert time_coeffs['n_timestamps'] % n_timesteps == 0
    assert msig.inputs.shape == (time_coeffs['n_timestamps'] / n_timesteps, n_timesteps, 1)
    assert msig.labels.shape == (time_coeffs['n_timestamps'] / n_timesteps, len(sig_coeffs['waves']))
    assert np.all(msig.signal_names == ['A', 'B', 'C'])
    assert np.all(msig.signal_colors == ['#ff0000', '#00ff00', '#0000ff'])
