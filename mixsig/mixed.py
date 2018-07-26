import os
import json
import numpy as np
from keras.utils import Sequence
from .utils import get_datetime_now
from .utils import string2shape
from .utils import shape2string
from .utils import timesequence_generator
from .utils import create_one_hots_from_labels
from .utils import one_hot_encode
from .waves import Wave
from .waves import MixedWave


class MixedSignal:
    def __init__(self,
                 sigs_coeffs,
                 *features,
                 batch_size=1,
                 window_size=0,
                 window_type='sliding',
                 network_type='TCN',
                 sequence_type='many2many',
                 classification_type='categorical',
                 stateful=False,
                 run_label=None,
                 n_groups=5,
                 name='Mixed'):

        self.group_index = 0
        self.group_indices = None
        self.n_groups = n_groups

        self.features = features or ('x',)
        self.n_features = len(self.features)

        if stateful:
            assert batch_size > 0
        else:
            batch_size = 1

        self.stateful = stateful
        self.batch_size = batch_size

        # Should these be properties?
        self.name = name
        self.inputs = None
        self.labels = None
        self.one_hots = None
        self.mixed_signal = None
        self.n_timestamps = None
        self._n_samples = None
        self._window_size = None if window_size < 1 else window_size

        self.window_type = window_type.lower()
        assert self.window_type in ('sliding', 'boxcar', 'random')

        self.network_type = network_type
        assert self.network_type in ('MLP', 'RNN', 'LSTM', 'PLSTM', 'TCN')

        self.sequence_type = sequence_type
        self._sequence_code = None
        assert self.sequence_type in ('one2one', 'one2many', 'many2one', 'many2many')

        if 'time' in sigs_coeffs:
            self.sequence_generator = timesequence_generator(**sigs_coeffs['time'])

        mwave_indexes = []
        n_mixed_waves = 0
        mwave_idx = None
        has_time = []
        for i, coeffs in enumerate(sigs_coeffs):
            if 'time' in coeffs:
                has_time.append(i)
            else:
                mwave_indexes.append(i)
            if 'name' in coeffs and coeffs['name'].lower() == 'mixed_wave':
                n_mixed_waves += 1
                mwave_idx = i

        assert n_mixed_waves in (0, 1), print(f'only zero or one mixed waves allowed, found {n_mixed_waves}')

        if n_mixed_waves:
            assert mwave_idx in has_time, print('mixed-wave must have keyword, time.')
            assert len(sigs_coeffs) > len(has_time)
            assert len(mwave_indexes) > 1, print('Need more than one wave for a mixed-wave')
            mwave_coeffs = sigs_coeffs.pop(mwave_idx)
            mwave_indexes = [i if i < mwave_idx else i - 1 for i in mwave_indexes]
            self.mixed_wave = MixedWave(classes=mwave_indexes, mwave_coeffs=mwave_coeffs)
        else:
            assert len(sigs_coeffs) == len(has_time)
            self.mixed_wave = None

        self.waves = [Wave(*self.features, label=i, **coeffs) for i, coeffs in enumerate(sigs_coeffs)]
        self.n_classes = len(self.waves)

        self.classification_type = 'categorical' if self.n_classes > 2 else classification_type

        run_label = run_label or get_datetime_now(fmt='%Y_%m%d_%H%M')

        self.config_dict = {
            'run_label': run_label,
            'window_size': self._window_size,
            'window_type': self.window_type,
            'network_type': self.network_type,
            'sequence_type': self.sequence_type,
            'sigs_coeffs': sigs_coeffs,
        }

        # TODO: What's the appropriate way to assign the out_dir (regular functionality, unit tests, etc.)
        # TODO: Relative to the root directory of this project?
        # TODO: Relative to the directory of the calling script?
        # TODO: Relative to the directory of this module?
        self.out_dir = os.path.join(os.getcwd(), 'out', run_label)
        self.model_filename = os.path.join(self.out_dir, 'model.h5')
        self.config_filename = os.path.join(self.out_dir, 'mixed_signal_config.json')
        self.model_weights_filename = os.path.join(self.out_dir, 'model_weights.h5')
        self.training_stats_filename = os.path.join(self.out_dir, 'training_stats.csv')

    def generate(self, sequence_code=None):
        """ Generate waves from property values."""

        if self.mixed_wave:
            self.mixed_wave.generate()

        # generate new waves.
        timestamps = []
        labels = []
        inputs = []
        for i, wave in enumerate(self.waves):
            if self.mixed_wave and i in self.mixed_wave.classes:
                indices = np.where(self.mixed_wave.labels == i)[0]
                wave.generate(self.mixed_wave.timestamps, indices=indices, **self.mixed_wave.props)
            else:
                wave.generate()

            timestamps.append(wave.timestamps)
            labels.append(wave.labels)
            inputs.append(wave.inputs)

        timestamps = np.hstack(timestamps)
        labels = np.hstack(labels)
        inputs = np.vstack(inputs)


        window_size = self.window_size or 1

        # clip data from the left so that it divides batch_size evenly.
        if self.window_type == 'boxcar':
            assert len(timestamps) >= window_size * self.batch_size
            chop_index = len(timestamps) % (window_size * self.batch_size)
        else:  # ('sliding' and 'random')
            chop_index = (len(timestamps) - window_size + 1) % self.batch_size

        # Sort the labels and mixed_signal chronologically.
        sorted_indices = np.argsort(timestamps)[chop_index:]

        self.timestamps = timestamps[sorted_indices]
        self.labels = labels[sorted_indices]
        self.inputs = inputs[sorted_indices]

        self.n_timestamps = len(self.timestamps)
        if self._window_size is None:
            self.window_size = self.n_timestamps

        # self.one_hots = create_one_hots_from_labels(self.labels, self.n_classes)
        self.one_hots = one_hot_encode(self.labels, self.n_classes)
        self.mixed_signal = self.inputs[..., 0]

        # Sanity check
        assert len(self.timestamps) == len(self.mixed_signal) == len(self.labels) == len(self.inputs) == len(self.one_hots)

        # window_type, window_size

        # sliding, ws < 0                ->  raise ValueError('window_size must be non negative')
        # boxcar , ws < 0                ->  raise ValueError('window_size must be non negative')
        # random , ws < 0                ->  raise ValueError('window_size must be non negative')

        # sliding, ws = 0                ->  (sliding, n_timestamps)
        # boxcar , ws = 0                ->  (boxcar, n_timestamps)
        # random , ws = 0                ->  (random, n_timestamps)

        # sliding, ws = 1                ->  valid same as (boxcar, 1)
        # boxcar , ws = 1                ->  valid same as (sliding, 1)
        # random , ws = 1                ->  raise NotImplementedError

        # sliding, 1 < ws < n_timestamps ->  valid
        # boxcar , 1 < ws < n_timestamps ->  valid
        # random , 1 < ws < n_timestamps ->  valid

        # sliding, n_timestamps = ws     ->  valid same as boxcar and random
        # boxcar , n_timestamps = ws     ->  valid same as sliding and random
        # random , n_timestamps = ws     ->  valid same as sliding and boxcar

        # sliding, n_timestamps < ws     ->  raise ValueError('window_size must be <= n_timestamps')
        # boxcar , n_timestamps < ws     ->  raise ValueError('window_size must be <= n_timestamps')
        # random , n_timestamps < ws     ->  raise ValueError('window_size must be <= n_timestamps')

        if self.window_type == 'sliding':
            return self.generate_sliding(sequence_code=sequence_code)
        elif self.window_type == 'boxcar':
            return self.generate_boxcar()
        elif self.window_type == 'random':
            return self.generate_sliding(sequence_code=sequence_code)
        else:
            raise ValueError('Invalid window_type: {}. Use "sliding", "boxcar" or "random"')

    @property
    def n_samples(self):
        if self._n_samples is None:
            if self.n_timestamps is None:  # then window_size will become n_timestamps. Delayed.
                self._n_samples = None
            else:
                if self.window_type == 'boxcar':
                    assert self.n_timestamps % self.window_size == 0
                    self._n_samples = self.n_timestamps // self.window_size
                else:
                    self._n_samples = self.n_timestamps - self.window_size + 1

        return self._n_samples

    @n_samples.setter
    def n_samples(self, val):
        if val == 0:
            val = self.n_timestamps

        if self.window_type == 'boxcar' and self.n_timestamps % val != 0:
            raise ValueError('n_samples must divide n_timestamps evenly when using boxcar')

        if val < 0 or val > self.n_timestamps:
            raise ValueError('n_samples must be in the range [0, n_timestamps]')

        if 1 <= val <= self.n_timestamps:
            self._n_samples = val
            self._window_size = None

    @property
    def window_size(self):
        if self._window_size is None:
            if self.n_timestamps is None:
                self._window_size = None
            else:
                if self.window_type == 'boxcar':
                    assert self.n_timestamps % self.window_size == 0
                    self._window_size = self.n_timestamps // self.n_samples
                else:
                    self._window_size = self.n_timestamps - self.n_samples + 1

        return self._window_size

    @window_size.setter
    def window_size(self, val):
        if val == 0:
            val = self.n_timestamps

        if self.window_type == 'boxcar' and self.n_timestamps % val != 0:
            raise ValueError('window_size must divide n_timestamps evenly when using boxcar')

        if val < 0 or val > self.n_timestamps:
            raise ValueError('window_size must be in the range [0, n_timestamps]')

        if 1 <= val <= self.n_timestamps:
            self._window_size = val
            self._n_samples = None

    def in_out_shape_encoder(self, in_shape, out_shape):

        # sequence_types
        # t -> n_[t]imestamps
        # x -> n_samples (or number of sub-samples)
        # w -> [w]indow_size
        # f -> n_[f]eatures
        # c -> n_[c]lasses

        # shape[0] -> number of sequences, number of samples
        # shape[1] -> number of timesteps, sample length, sequence length, window_size
        # shape[2] -> number of features (input), number of classes (output)

        # (01tx, 01tw, 01fc)

        # examples:
        # t = 8
        # x = 6 (sliding), x = 4 (boxcar)
        # w = 3 (sliding), w = 2 (boxcar)
        # f = 9
        # c = 5


        seq_bits = {
            '1': 1,
            't': self.n_timestamps,
            'x': self.n_samples,
            'w': self.window_size,
            'f': self.n_features,
            'c': self.n_classes,
        }

        ic = shape2string(in_shape, seq_bits)
        oc = shape2string(out_shape, seq_bits)

        in_out_code = '_'.join([ic, oc])
        return in_out_code

    def in_out_shape_decoder(self, in_out_code):

        #  one2one    t00_t00  (8,     )  (8,     )
        #  one2one    t01_t00  (8,    1)  (8,     )
        #  one2one    t0f_t00  (8,    9)  (8,     )
        #  one2one    t10_t00  (8  1,  )  (8,     )
        #  one2one    t11_t00  (8, 1, 1)  (8,     )
        #  one2one    t1f_t00  (8, 1, 9)  (8,     )

        #  one2one    t00_t01  (8,     )  (8,    1)
        #  one2one    t01_t01  (8,    1)  (8,    1)
        #  one2one    t0f_t01  (8,    9)  (8,    1)
        #  one2one    t10_t01  (8  1,  )  (8,    1)
        #  one2one    t11_t01  (8, 1, 1)  (8,    1)
        #  one2one    t1f_t01  (8, 1, 9)  (8,    1)

        #  one2one    t00_t0c  (8,     )  (8,    5)
        #  one2one    t01_t0c  (8,    1)  (8,    5)
        #  one2one    t0f_t0c  (8,    9)  (8,    5)
        #  one2one    t10_t0c  (8  1,  )  (8,    5)
        #  one2one    t11_t0c  (8, 1, 1)  (8,    5)
        #  one2one    t1f_t0c  (8, 1, 9)  (8,    5)

        #  one2one    t00_t10  (8,     )  (8, 1,  )
        #  one2one    t01_t10  (8,    1)  (8, 1,  )
        #  one2one    t0f_t10  (8,    9)  (8, 1,  )
        #  one2one    t10_t10  (8  1,  )  (8, 1,  )
        #  one2one    t11_t10  (8, 1, 1)  (8, 1,  )
        #  one2one    t1f_t10  (8, 1, 9)  (8, 1,  )

        #  one2one    t00_t11  (8,     )  (8, 1, 1)
        #  one2one    t01_t11  (8,    1)  (8, 1, 1)
        #  one2one    t0f_t11  (8,    9)  (8, 1, 1)
        #  one2one    t10_t11  (8  1,  )  (8, 1, 1)
        #  one2one    t11_t11  (8, 1, 1)  (8, 1, 1) 6
        #  one2one    t1f_t11  (8, 1, 9)  (8, 1, 1) 6

        #  one2one    t00_t1c  (8,     )  (8, 1, 5)
        #  one2one    t01_t1c  (8,    1)  (8, 1, 5)
        #  one2one    t0f_t1c  (8,    9)  (8, 1, 5)
        #  one2one    t10_t1c  (8  1,  )  (8, 1, 5)
        #  one2one    t11_t1c  (8, 1, 1)  (8, 1, 5) 6
        #  one2one    t1f_t1c  (8, 1, 9)  (8, 1, 5) 6

        # many2one    xw0_x00  (6, 3,  )  (6,     )
        # many2one    xw1_x00  (6, 3, 1)  (6,     )
        # many2one    xwf_x00  (6, 3, 9)  (6,     )
        # many2one    xw0_x01  (6, 3,  )  (6,    1)
        # many2one    xw1_x01  (6, 3, 1)  (6,    1)
        # many2one    xwf_x01  (6, 3, 9)  (6,    1)
        # many2one    xw0_x0c  (6, 3,  )  (6,    5)
        # many2one    xw1_x0c  (6, 3, 1)  (6,    5)
        # many2one    xwf_x0c  (6, 3, 9)  (6,    5)

        # many2one    xw0_x10  (6, 3,  )  (6, 1,  )
        # many2one    xw1_x10  (6, 3, 1)  (6, 1,  )
        # many2one    xwf_x10  (6, 3, 9)  (6, 1,  )
        # many2one    xw0_x11  (6, 3,  )  (6, 1, 1)
        # many2one    xw1_x11  (6, 3, 1)  (6, 1, 1) 6
        # many2one    xwf_x11  (6, 3, 9)  (6, 1, 1) 6
        # many2one    xw0_x1c  (6, 3,  )  (6, 1, 5)
        # many2one    xw1_x1c  (6, 3, 1)  (6, 1, 5) 6
        # many2one    xwf_x1c  (6, 3, 9)  (6, 1, 5) 6

        #  one2many   x00_xw0  (6,     )  (6, 3,  )
        #  one2many   x01_xw0  (6,    1)  (6, 3,  )
        #  one2many   x0f_xw0  (6,    9)  (6, 3,  )
        #  one2many   x00_xw1  (6,     )  (6, 3, 1)
        #  one2many   x01_xw1  (6,    1)  (6, 3, 1)
        #  one2many   x0f_xw1  (6,    9)  (6, 3, 1)
        #  one2many   x00_xwc  (6,     )  (6, 3, 5)
        #  one2many   x01_xwc  (6,    1)  (6, 3, 5)
        #  one2many   x0f_xwc  (6,    9)  (6, 3, 5)

        #  one2many   x10_xw0  (6, 1,  )  (6, 3,  )
        #  one2many   x11_xw0  (6, 1, 1)  (6, 3,  )
        #  one2many   x1f_xw0  (6, 1, 9)  (6, 3,  )
        #  one2many   x10_xw1  (6, 1,  )  (6, 3, 1)
        #  one2many   x11_xw1  (6, 1, 1)  (6, 3, 1) 6
        #  one2many   x1f_xw1  (6, 1, 9)  (6, 3, 1) 6
        #  one2many   x10_xwc  (6, 1,  )  (6, 3, 5)
        #  one2many   x11_xwc  (6, 1, 1)  (6, 3, 5) 6
        #  one2many   x1f_xwc  (6, 1, 9)  (6, 3, 5) 6

        # many2many   xw0_xw0  (6, 3   )  (6, 3,  )
        # many2many   xw1_xw0  (6, 3, 1)  (6, 3,  )
        # many2many   xwf_xw0  (6, 3, 9)  (6, 3,  )
        # many2many   xw0_xw1  (6, 3   )  (6, 3, 1)
        # many2many   xw1_xw1  (6, 3, 1)  (6, 3, 1) 6
        # many2many   xwf_xw1  (6, 3, 9)  (6, 3, 1) 6
        # many2many   xw0_xwc  (6, 3   )  (6, 3, 5)
        # many2many   xw1_xwc  (6, 3, 1)  (6, 3, 5) 6
        # many2many   xwf_xwc  (6, 3, 9)  (6, 3, 5) 6

        # many2many   0t0_0t0  (   8,  )  (   8,  )
        # many2many   0t1_0t0  (   8, 1)  (   8,  )
        # many2many   0tf_0t0  (   8, 9)  (   8,  )
        # many2many   1t0_0t0  (1, 8   )  (   8,  )
        # many2many   1t1_0t0  (1, 8, 1)  (   8,  )
        # many2many   1tf_0t0  (1, 8, 9)  (   8,  )

        # many2many   0t0_0t1  (   8,  )  (   8, 1)
        # many2many   0t1_0t1  (   8, 1)  (   8, 1)
        # many2many   0tf_0t1  (   8, 9)  (   8, 1)
        # many2many   1t0_0t1  (1, 8   )  (   8, 1)
        # many2many   1t1_0t1  (1, 8, 1)  (   8, 1)
        # many2many   1tf_0t1  (1, 8, 9)  (   8, 1)

        # many2many   0t0_0tc  (   8,  )  (   8, 5)
        # many2many   0t1_0tc  (   8, 1)  (   8, 5)
        # many2many   0tf_0tc  (   8, 9)  (   8, 5)
        # many2many   1t0_0tc  (1, 8   )  (   8, 5)
        # many2many   1t1_0tc  (1, 8, 1)  (   8, 5)
        # many2many   1tf_0tc  (1, 8, 9)  (   8, 5)

        # many2many   0t0_1t0  (   8,  )  (1, 8,  )
        # many2many   0t1_1t0  (   8, 1)  (1, 8,  )
        # many2many   0tf_1t0  (   8, 9)  (1, 8,  )
        # many2many   1t0_1t0  (1, 8   )  (1, 8,  )
        # many2many   1t1_1t0  (1, 8, 1)  (1, 8,  )
        # many2many   1tf_1t0  (1, 8, 9)  (1, 8,  )

        # many2many   0t0_1t1  (   8,  )  (1, 8, 1)
        # many2many   0t1_1t1  (   8, 1)  (1, 8, 1)
        # many2many   0tf_1t1  (   8, 9)  (1, 8, 1)
        # many2many   1t0_1t1  (1, 8   )  (1, 8, 1)
        # many2many   1t1_1t1  (1, 8, 1)  (1, 8, 1) 6
        # many2many   1tf_1t1  (1, 8, 9)  (1, 8, 1) 6

        # many2many   0t0_1tc  (   8,  )  (1, 8, 5)
        # many2many   0t1_1tc  (   8, 1)  (1, 8, 5)
        # many2many   0tf_1tc  (   8, 9)  (1, 8, 5)
        # many2many   1t0_1tc  (1, 8   )  (1, 8, 5)
        # many2many   1t1_1tc  (1, 8, 1)  (1, 8, 5) 6
        # many2many   1tf_1tc  (1, 8, 9)  (1, 8, 5) 6

        seq_bits = {
            '1': 1,
            't': self.n_timestamps,
            'x': self.n_samples,
            'w': self.window_size,
            'f': self.n_features,
            'c': self.n_classes,
        }

        ic, oc = in_out_code.split('_')
        in_shape = string2shape(ic, seq_bits)
        out_shape = string2shape(oc, seq_bits)

        return in_shape, out_shape

    @property
    def sequence_code(self):

        # TODO: unit tests to make sure all these pathways are correct.

        if self._sequence_code is None:

            in_seq, out_seq = self.sequence_type.split('2')
            # sequence_type
            st = {'one': {'t', 't0', 't1', 'x', 'x0', 'x1'}, 'many': {'1t', 'xw'}}

            if self.window_size == 0:
                ws = {'1t'}
            elif self.window_size == 1:
                ws = {'t', 't0', 't1'}
            else:
                # ws = {'x', 'x0', 'x1', 'xw'}
                ws = {'x', 'x0', 'xw'}

            in_base = st[in_seq] & ws
            out_base = st[out_seq] & ws

            if (in_seq == 'many' or out_seq == 'many') and self.window_size == 1:
                raise ValueError('Only with one2one can you use a window_size == 1')

            if self.n_features == 1:
                n_feats = ('0', '1')
            elif self.n_features >= 2:
                n_feats = ('f',)
            else:
                raise ValueError('n_features cannot be negative or zero')

            in_codes = [ib + nf for ib in in_base for nf in n_feats]
            if 't10' in in_codes and 't01' in in_codes:
                in_codes.remove('t01')
            if 'x10' in in_codes and 'x01' in in_codes:
                in_codes.remove('x01')
            if 'tf' in in_codes and 't0f' in in_codes:
                in_codes.remove('t0f')
            if 'xf' in in_codes and 'x0f' in in_codes:
                in_codes.remove('x0f')
            in_codes = set([ic.strip('0') for ic in in_codes])
            in_code = max(in_codes, key=lambda x: len(x))

            if self.n_classes == 2:
                if self.classification_type == 'binary':
                    n_class = ('0', '1')
                elif self.classification_type == 'categorical':
                    n_class = ('c',)
                else:
                    raise ValueError(f'incorrect classification_type {self.classification_type}')
            elif self.n_classes >= 3:
                n_class = ('c',)
            else:
                raise ValueError('n_classes must be 2 or greater')

            out_codes = [ob + nc for ob in out_base for nc in n_class]
            if 't10' in out_codes and 't01' in out_codes:
                out_codes.remove('t01')
            if 'x10' in out_codes and 'x01' in out_codes:
                out_codes.remove('x01')
            if 'tc' in out_codes and 't0c' in out_codes:
                out_codes.remove('t0c')
            if 'xc' in out_codes and 'x0c' in out_codes:
                out_codes.remove('x0c')
            out_codes = set([oc.strip('0') for oc in out_codes])
            out_code = max(out_codes, key=lambda x: len(x))

            self._sequence_code = '_'.join([in_code, out_code])

        return self._sequence_code

    def generate_sliding(self, sequence_code=None):

        if self.sequence_type == 'many2one+time':
            # PLSTM: many2one (1088, 100, 2) -> (1088, 3)
            X = np.zeros((self.n_samples, self.window_size, 2))
            for j in range(self.window_size):
                X[:, j, 0] = self.mixed_signal[j:j + self.n_samples]
                X[:, j, 1] = self.timestamps[j + self.n_samples - 1] - self.timestamps[j:j + self.n_samples]
            y = self.one_hots[self.window_size - 1:]
            return X, y

        sequence_code = sequence_code or self.sequence_code
        X_code, y_code = sequence_code.split('_')

        if X_code in ('t', 't0', 't00', '0t0'):
            X = self.mixed_signal
        elif X_code in ('t1', 't01', 't10'):
            X = self.mixed_signal[..., None]
        elif X_code in ('1t', '1t0'):
            X = self.mixed_signal[None, ...]
        elif X_code in ('t11',):
            X = self.mixed_signal[..., None, None]
        elif X_code in ('1t1',):
            X = self.mixed_signal[None, ..., None]
        elif X_code in ('tf', '0tf', 't0f'):
            X = self.inputs
        elif X_code in ('t1f',):
            X = self.inputs[:, None, :]
        elif X_code in ('1tf',):
            X = self.inputs[None, ...]

        elif X_code in ('x', 'x0', 'x00'):
            X = self.mixed_signal[self.window_size - 1:]
        elif X_code in ('x1', 'x10', 'x01'):
            X = self.mixed_signal[self.window_size - 1:, None]
        elif X_code == 'x11':
            X = self.mixed_signal[self.window_size - 1:, None, None]
        elif X_code in ('xf', 'x0f'):
            X = self.inputs[self.window_size - 1:]
        elif X_code == 'x1f':
            X = self.inputs[self.window_size - 1:, None, :]
        elif X_code in ('xw', 'xw0'):
            X = np.zeros((self.n_samples, self.window_size))
            for j in range(self.window_size):
                X[:, j] = self.mixed_signal[j:j + self.n_samples]
        elif X_code == 'xw1':
            X = np.zeros((self.n_samples, self.window_size, 1))
            for j in range(self.window_size):
                X[:, j, 0] = self.mixed_signal[j:j + self.n_samples]
        elif X_code == 'xwf':
            X = np.zeros((self.n_samples, self.window_size, self.n_features))
            for j in range(self.window_size):
                X[:, j] = self.inputs[j:j + self.n_samples]
        else:
            raise NotImplementedError(X_code)

        if y_code in ('t', 't0', 't00', '0t0'):
            y = self.labels
        elif y_code in ('t1', 't01', 't10'):
            y = self.labels[..., None]
        elif y_code in ('1t', '1t0'):
            y = self.labels[None, ...]
        elif y_code in ('t11',):
            y = self.labels[..., None, None]
        elif y_code == '1t1':
            y = self.labels[None, ..., None]
        elif y_code in ('tc', '0tc', 't0c'):
            y = self.one_hots
        elif y_code == 't1c':
            y = self.one_hots[:, None, :]
        elif y_code == '1tc':
            y = self.one_hots[None, ...]

        elif y_code in ('x', 'x0', 'x00'):
            y = self.labels[self.window_size - 1:]
        elif y_code in ('x1', 'x01', 'x10'):
            y = self.labels[self.window_size - 1:, None]
        elif y_code == 'x11':
            y = self.labels[self.window_size - 1:, None, None]
        elif y_code in ('xc', 'x0c'):
            y = self.one_hots[self.window_size - 1:]
        elif y_code == 'x1c':
            y = self.one_hots[self.window_size - 1:, None, :]
        elif y_code in ('xw', 'xw0'):
            y = np.zeros((self.n_samples, self.window_size))
            for j in range(self.window_size):
                y[:, j] = self.labels[j:j + self.n_samples]
        elif y_code == 'xw1':
            y = np.zeros((self.n_samples, self.window_size, 1))
            for j in range(self.window_size):
                y[:, j, 0] = self.labels[j:j + self.n_samples]
        elif y_code == 'xwc':
            y = np.zeros((self.n_samples, self.window_size, self.n_classes))
            for j in range(self.window_size):
                y[:, j] = self.one_hots[j:j + self.n_samples]
        else:
            raise NotImplementedError(y_code)

        return X, y

    def generate_boxcar(self, sequence_code=None):

        # TODO: Need unit tests to make sure all these pathways are correct.
        # sequence_types
        # x -> n_samples (or sub-samples)
        # w -> [w]indow_size
        # c -> n_[c]lasses
        # one2one   t_t     (1088,)        (1088,)  <- binary outputs
        # one2one   t_tc    (1088,)        (1088, 3)
        # one2one   t1_tc   (1088, 1)      (1088, 3)
        # many2one  xw_xc   (1088, 100)    (1088, 3)
        # many2one  xw1_xc  (1088, 100, 1) (1088, 3)
        # many2many xw_xwc  (1088, 100)    (1088, 100, 3)
        # many2many xw1_xwc (1088, 100, 1) (1088, 100, 3)

        sequence_code = sequence_code or self.sequence_code
        X_code, y_code = sequence_code.split('_')

        if X_code == 't':
            X = self.mixed_signal
        elif X_code == 't1':
            X = self.mixed_signal[..., None]
        elif X_code == 'xw':
            X = self.mixed_signal.reshape((self.n_samples, self.window_size))
        elif X_code == 'xw1':
            X = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
        else:
            raise NotImplementedError

        if y_code == 't':
            y = self.labels
        elif y_code == 'tc':
            y = self.one_hots
        elif y_code == 'xc':
            y = self.one_hots.reshape((self.n_samples, self.window_size, self.n_classes))
            y = y[:, -1, :]
            y = y.reshape(self.n_samples, self.n_classes)
        elif y_code == 'xwc':
            y = self.one_hots.reshape((self.n_samples, self.window_size, self.n_classes))
        else:
            raise NotImplementedError

        return X, y

    def generate_groups(self, n_groups, shuffle_inplace=False):
        # This is best suited for generating using the sliding window method.
        # examples:
        # if n_samples == 1
        # (1088, 100, 1) -> (1 * 1088, 100, 1) -> (1088, 100, 1)
        # if n_samples == 32
        # (1088, 100, 1) -> (32 * 1088, 100, 1) -> (34816, 100, 1)
        x, y = self.generate()
        for i in range(n_groups - 1):
            xi, yi = self.generate()
            x = np.vstack((x, xi))
            y = np.vstack((y, yi))
        n_samples = len(x)
        n_batches = n_samples // self.batch_size
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        if shuffle_inplace:
            return x[indices], y[indices]
        else:
            return x, y, indices, n_batches

    def generate_samples(self, n_samples, sequence_code=None):
        # This is best suited for generating data where window_size == n_timestamps.
        # examples:
        # if n_samples == 0
        # (1200, 1) -> (1200, 1)
        # if n_samples == 1
        # (1200, 1) -> (1, 1200, 1)
        # if n_samples == 32
        # (1200, 1) -> (32, 1200, 1)
        if n_samples < 1:
            raise ValueError('n_samples must be >= 1')
        X = []
        y = []
        for i in range(n_samples):
            Xi, yi = self.generate(sequence_code=sequence_code)
            X.append(Xi)
            y.append(yi)
        return np.stack(X), np.stack(y)

    def save_config(self):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.config_filename, 'w') as ofs:
            json.dump(self.config_dict, ofs, indent=4)


class SignalGenerator(Sequence):
    def __init__(self,
                 n_samples,
                 batch_size,
                 msig,
                 inout_shape_code='tf_tc'):

        """Initialization"""
        self.n_samples = n_samples
        self.batch_size = batch_size

        self.inout_shape_code = inout_shape_code
        in_shape, out_shape = msig.in_out_shape_decoder(inout_shape_code)
        self.in_batch_shape = (batch_size,) + in_shape
        self.out_batch_shape = (batch_size,) + out_shape

        self.generate = msig.generate

        self.indexes = np.arange(n_samples)
        # self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        # return int(np.floor(self.n_samples / self.batch_size))
        return self.n_samples // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate and return the data
        X, y = self._batch_generator(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.n_samples)

    def _batch_generator(self, indexes):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty(self.in_batch_shape)
        y = np.empty(self.out_batch_shape, dtype=int)

        # Generate data
        for i, _ in enumerate(indexes):
            X[i], y[i] = self.generate(sequence_code=self.inout_shape_code)
        return X, y
