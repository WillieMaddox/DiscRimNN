import pytest
# import numpy as np
# from hypothesis import given
# from hypothesis import example
# from hypothesis import strategies as st
from mixsig.utils import timesequence_generator


@pytest.mark.parametrize('t_min,t_max,n_max,n_timestamps', [
    (None, None, 200, None),
    (None, 50, 200, None),
    (0, None, 200, None),
    (50, 50, 200, None),
    (51, 50, 200, None),
    (0, 50, None, None),
    (0, 50, 200, 200),
], ids=repr)
def test_sequence_generator1(t_min, t_max, n_max, n_timestamps):
    with pytest.raises(ValueError):
        timesequence_generator(
            t_min=t_min,
            t_max=t_max,
            n_max=n_max,
            n_timestamps=n_timestamps)


@pytest.mark.parametrize('n_min,n_max', [
    (0, 200),
    (1, 200),
    (201, 200),
], ids=repr)
def test_sequence_generator2(n_min, n_max):
    with pytest.raises(AssertionError):
        timesequence_generator(
            t_min=0,
            t_max=50,
            n_max=n_max,
            n_min=n_min)


def test_sequence_generator_invalid_noise_type():
    with pytest.raises(AssertionError):
        timesequence_generator(
            t_min=0,
            t_max=50,
            n_max=200,
            noise_type='badstring')


@pytest.mark.parametrize('delta', [-1, 0, 1.01], ids=repr)
def test_sequence_generator_jitter(delta):
    with pytest.raises(AssertionError):
        timesequence_generator(
            t_min=0,
            t_max=50,
            n_max=200,
            noise_type='jitter',
            delta=delta)


@pytest.mark.parametrize('pareto_shape', [-1, 0], ids=repr)
def test_sequence_generator_pareto(pareto_shape):
    with pytest.raises(AssertionError):
        timesequence_generator(
            t_min=0,
            t_max=50,
            n_max=200,
            noise_type='pareto',
            pareto_shape=pareto_shape)
