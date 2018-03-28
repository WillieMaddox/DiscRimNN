import pytest
import numpy as np
from hypothesis import given
from hypothesis import assume
from hypothesis import example
from hypothesis import strategies as st
from mixsig.timesequence import TimeSequence

st_timesequence_args = st.tuples(
    st.one_of(
        st.integers(min_value=0),
        st.floats(min_value=0.0, allow_infinity=False, allow_nan=False)
    ),
    st.one_of(
        st.integers(min_value=0),
        st.floats(min_value=0.0, allow_infinity=False, allow_nan=False)
    ),
    st.one_of(
        st.integers(min_value=2, max_value=20000)
    )
)


st_timesequence_kwargs = st.fixed_dictionaries(
    dict(
        delta=st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=1),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        )
    )
)


# @st.composite
# def timesequence_inputs(draw):
#     args = draw(st_timesequence_args)
#     kwargs = draw(st_timesequence_kwargs)
#     assume(args[0] < args[1])
#     return args, kwargs


# @st.composite
# def timesequence_kwargs(draw):
#     kwargs = draw(st_timesequence_kwargs)
#     assume(kwargs['start'] < kwargs['stop'])
#     return kwargs


# @given(timesequence_kwargs())
# def test_timesequence_inputs2(kwargs):
#     ts = TimeSequence(**kwargs)
#     assert isinstance(ts(), np.ndarray)
#     assert len(ts()) == ts.n_timestamps


# @given(timesequence_inputs())
# def test_timesequence_inputs2(args, kwargs):
#     ts = TimeSequence(*args, **kwargs)
#     assert isinstance(ts(), np.ndarray)
#     assert len(ts()) == ts.n_timestamps


@given(st_timesequence_args, st_timesequence_kwargs)
def test_timesequence_inputs(args, kwargs):
    assume(args[0] < args[1])
    ts = TimeSequence(*args, **kwargs)
    assert isinstance(ts(), np.ndarray)
    assert len(ts()) == ts.n_timestamps
    assert len(ts) == ts.n_timestamps


@given(st_timesequence_args, st_timesequence_kwargs)
@example(args=(0.0, 50.0, 201), kwargs={'delta': None})
@example(args=(0.0, 50.0, 201), kwargs={'delta': 0})
@example(args=(0.0, 50.0, 201), kwargs={'delta': 0.5})
@example(args=(0.0, 50.0, 201), kwargs={'delta': 1.0})
def test_timesequence_generator(args, kwargs):
    assume(args[0] < args[1])
    ts = TimeSequence(*args, **kwargs)
    ts.generate()
    dt = ts.timestamps[1:] - ts.timestamps[:-1]
    assert np.all(dt > 0), f'{ts.start} {ts.stop} {ts.n_timestamps} {ts.dt} {ts.delta}'
