import os
import json
import threading
import numpy as np
from .utils import factors
from .utils import get_datetime_now
from .utils import timesequence_generator
from .utils import create_label_distribution
from .utils import create_one_hots_from_labels
from .waves import Wave
from .waves import Amplitude
from .waves import Frequency
from .waves import Offset
from .waves import Phase


class MixedSignal:
    def __init__(self,
                 sigs_coeffs,
                 msig_coeffs=None,
                 batch_size=1,
                 window_size=1,
                 window_type='sliding',
                 network_type='RNN',
                 sequence_code='t_tc',
                 run_label=None,
                 name='Mixed',
                 n_groups=5):

        self.lock = threading.Lock()
        self.n_groups = n_groups
        self.group_index = 0
        self.group_indices = None
        self.X = None
        self.y = None

        # Should these be properties?
        self.labels = None
        self.one_hots = None
        self.mixed_signal = None

        self.batch_size = batch_size
        self.window_size = window_size
        self.network_type = network_type
        self.sequence_code = sequence_code
        self._sequence_type = None
        self.name = name

        if window_type is not None:
            window_type = window_type.lower()
            assert window_type in ('sliding', 'boxcar')
        self.window_type = window_type
        assert self.network_type in ('MLP', 'RNN', 'LSTM', 'PLSTM', 'TCN')
        assert self.sequence_code in ('t_t', 't_tc', 't1_tc', 'xw_xc', 'xw1_xc', 'xw_xwc', 'xw1_xwc')

        if 'time' in msig_coeffs:
            self.sequence_generator = timesequence_generator(**msig_coeffs['time'])

        mixed_signal_prop_defaults = {
            'amplitude': {'mean': 1, 'delta': 0},
            'frequency': {'mean': 1, 'delta': 0},
            'offset': {'mean': 0, 'delta': 0},
            'phase': {'mean': 0, 'delta': 0},
        }
        self.mixed_signal_props = {}
        for prop_name, default_coeffs in mixed_signal_prop_defaults.items():
            coeffs = msig_coeffs[prop_name] if prop_name in msig_coeffs else default_coeffs
            if prop_name == 'amplitude':
                self.mixed_signal_props[prop_name] = Amplitude(**coeffs)
            elif prop_name == 'frequency':
                self.mixed_signal_props[prop_name] = Frequency(**coeffs)
            elif prop_name == 'offset':
                self.mixed_signal_props[prop_name] = Offset(**coeffs)
            elif prop_name == 'phase':
                self.mixed_signal_props[prop_name] = Phase(**coeffs)

        self.waves = [Wave(**coeffs) for coeffs in sigs_coeffs]

        self.n_classes = len(self.waves)

        run_label = run_label or get_datetime_now(fmt='%Y_%m%d_%H%M')

        self.config_dict = {
            'run_label': run_label,
            'window_size': self.window_size,
            'window_type': self.window_type,
            'network_type': self.network_type,
            'sequence_code': self.sequence_code,
            'sequence_type': self.sequence_type,
            'sigs_coeffs': sigs_coeffs,
            'msig_coeffs': msig_coeffs,
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

    @property
    def sequence_type(self):
        if self._sequence_type is None:
            if self.sequence_code in ('t_t', 't_tc', 't1_tc'):
                self._sequence_type = 'one2one'
            elif self.sequence_code in ('xw_xc', 'xw1_xc'):
                self._sequence_type = 'many2one'
            elif self.sequence_code in ('xw_xwc', 'xw1_xwc'):
                self._sequence_type = 'many2many'
                if self.network_type in ('PLSTM', 'PhasedLSTM'):
                    self._sequence_type = self._sequence_type + '+time'
            else:
                raise ValueError('Invalid sequence_code')
        return self._sequence_type

    def _generate_signals_old(self):
        """ Generate waves from property values."""
        # generate new timestamps
        self.timestamps = self.sequence_generator()
        # generate new values for each mixed signal property.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}
        # generate new single waves.
        self._signals = np.vstack([wave.generate(**props) for wave in self.waves])

    def _generate_signals(self):
        """ Generate waves from property values."""
        # First process the timestamp dependent waves.  (i.e. make a mixed signal wave.)
        # generate new timestamps
        timestamps = self.sequence_generator()
        classes = np.array([c for c, wave in enumerate(self.waves) if not wave.is_independent])

        # create a uniform distribution for class labels -> np.array([2,1,3, ... ,1])
        labels = create_label_distribution(len(timestamps), len(classes))

        # create one-hots from labels -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
        one_hots = create_one_hots_from_labels(labels, len(classes))

        # generate new mixed signal properties.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}

        # generate new individual waves.
        for wave in self.waves:
            wave.generate(timestamps, **props)

        signals = np.vstack([wave.sample for wave in self.waves if not wave.is_independent])
        mixed_signal = np.sum(one_hots.T * signals, axis=0)

        # make sure the labels align with the classes.
        labels = classes[labels]

        # Now append all the timestamp independent waves to the mixed signal.
        for c, wave in enumerate(self.waves):
            if wave.is_independent:
                timestamps = np.append(timestamps, wave.timestamps)
                mixed_signal = np.append(mixed_signal, wave.sample)
                labels = np.append(labels, np.zeros(len(wave), dtype=int) + c)

        # Sanity check
        assert len(timestamps) == len(mixed_signal) == len(labels)

        # Sort the labels and mixed_signal chronologically.
        sorted_indices = np.argsort(timestamps)

        # clip data from the left so that it divisible by batch_size.
        if self.window_type == 'sliding':
            chop_index = (len(timestamps) - self.window_size + 1) % self.batch_size
        elif self.window_type == 'boxcar':
            # assert len(timestamps) >= self.window_size
            # fact_a = factors(self.batch_size)
            # fact_b = factors(self.window_size)
            # gcm = max(fact_a.intersection(fact_b))
            # chop_index = len(timestamps) % (self.window_size * self.batch_size // gcm)
            assert len(timestamps) >= self.window_size * self.batch_size
            chop_index = len(timestamps) % (self.window_size * self.batch_size)
        else:
            chop_index = len(timestamps) % self.batch_size

        sorted_indices = sorted_indices[chop_index:]

        self.timestamps = timestamps[sorted_indices]
        self.mixed_signal = mixed_signal[sorted_indices]
        self.labels = labels[sorted_indices]

        self.n_timestamps = len(self.timestamps)
        self.t_min = self.timestamps[0]
        self.t_max = self.timestamps[-1]

        if self.window_type == 'sliding':
            self.n_samples = self.n_timestamps - self.window_size + 1
        elif self.window_type == 'boxcar':
            assert self.n_timestamps % self.window_size == 0
            self.n_samples = self.n_timestamps // self.window_size
        else:
            assert self.n_timestamps % self.window_size == 0
            self.n_samples = self.n_timestamps // self.window_size

        assert self.n_samples % self.batch_size == 0

    def generate(self):
        self._generate_signals()
        self.one_hots = create_one_hots_from_labels(self.labels, self.n_classes)
        # window_type, window_size

        # sliding, ws < 1               ->  raise ValueError('Please specify a window_size')
        # sliding, ws = 1               ->  valid same as (boxcar, 1)
        # sliding, 1 < ws < n_timesteps ->  valid
        # sliding, n_timesteps = ws     ->  valid same as (None, 0), (None, 1)
        # sliding, n_timesteps < ws     ->  raise ValueError('window_size must be <= n_timestamps')

        # boxcar, ws < 1               ->  raise ValueError('Please specify a window_size')
        # boxcar, ws = 1               ->  valid same as (sliding, 1)
        # boxcar, 1 < ws < n_timesteps ->  valid
        # boxcar, n_timesteps = ws     ->  valid same as (None, 0), (None, 1)
        # boxcar, n_timesteps < ws     ->  raise ValueError('window_size must be <= n_timestamps')

        # random, ws < 1               ->  raise ValueError('Please specify a window_size')
        # random, ws = 1               ->  raise NotImplementedError
        # random, 1 < ws < n_timesteps ->  valid
        # random, n_timesteps = ws     ->  valid same as (None, 0), (None, 1)
        # random, n_timesteps < ws     ->  raise ValueError('window_size must be <= n_timestamps')

        # None,    < 1  ->  valid
        # None,    = 1  ->  valid
        # None,    > 1  ->  raise ValueError('Please specify a window_size')

        if self.window_type == 'sliding':
            self.X, self.y = self.generate_sliding()
        elif self.window_type == 'boxcar':
            self.X, self.y = self.generate_boxcar()
        elif self.window_type is None:
            self.X, self.y = self.generate_sliding()
        else:
            raise ValueError('Invalid window_type: {}. Use "sliding", "boxcar" or None')
        return self.X, self.y

    def generate_sliding(self, sequence_code=None):

        # TODO: unit tests to make sure all these pathways are correct.
        # sequence_types
        # t -> n_[t]imestamps
        # x -> n_samples (or number of sub-samples)
        # w -> [w]indow_size
        # c -> n_[c]lasses
        # one2one   t_t     (1200,)        (1200,)
        # one2one   t_tc    (1200,)        (1200, 3)
        # one2one   t1_tc   (1200, 1)      (1200, 3)
        # many2one  xw_xc   (1088, 100)    (1088, 3)
        # many2one  xw1_xc  (1088, 100, 1) (1088, 3)
        # many2many xw_xwc  (1088, 100)    (1088, 100, 3)
        # many2many xw1_xwc (1088, 100, 1) (1088, 100, 3)

        if self.sequence_type == 'many2one+time':
            # PLSTM: many2one (1088, 100, 2) -> (1088, 3)
            X = np.zeros((self.n_samples, self.window_size, 2))
            for i in range(self.window_size):
                X[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
                X[:, i, 1] = self.timestamps[i + self.n_samples - 1] - self.timestamps[i:i + self.n_samples]
            y = self.one_hots[(self.window_size - 1):]
            return X, y

        sequence_code = sequence_code or self.sequence_code

        X_code, y_code = sequence_code.split('_')
        if X_code == 't':
            X = self.mixed_signal
        elif X_code == 't1':
            X = self.mixed_signal.reshape((self.n_timestamps, 1))
        elif X_code == 'xw':
            X = np.zeros((self.n_samples, self.window_size))
            for i in range(self.window_size):
                X[:, i] = self.mixed_signal[i:i + self.n_samples]
        elif X_code == 'xw1':
            X = np.zeros((self.n_samples, self.window_size, 1))
            for i in range(self.window_size):
                X[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
        else:
            raise NotImplementedError

        if y_code == 't':
            y = self.labels
        elif y_code == 'tc':
            y = self.one_hots
        elif y_code == 'xc':
            y = self.one_hots[(self.window_size - 1):]
        elif y_code == 'xwc':
            y = np.zeros((self.n_samples, self.window_size, self.n_classes))
            for i in range(self.window_size):
                y[:, i] = self.one_hots[i:i + self.n_samples]
        else:
            raise NotImplementedError

        return X, y

    def generate_boxcar(self, sequence_code=None):

        # TODO: Need unit tests to make sure all these pathways are correct.
        # sequence_types
        # x -> n_samples (or sub-samples)
        # w -> [w]indow_size
        # c -> n_[c]lasses
        # one2one   t_t     (1088,)        (1088,)
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
            X = self.mixed_signal.reshape((self.n_timestamps, 1))
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

    def generate_group(self, n_msigs, shuffle_inplace=False):
        # examples:
        # n_samples = 1
        # (1088, 100, 1) -> (1 * 1088, 100, 1) -> (1088, 100, 1)
        # n_samples = 32
        # (1088, 100, 1) -> (32 * 1088, 100, 1) -> (34816, 100, 1)
        x, y = self.generate()
        for i in range(n_msigs - 1):
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

    def generate_samples(self, n_samples):
        # examples:
        # n_samples = 0
        # (1088, 1) -> (1088, 1)
        # n_samples = 1
        # (1088, 1) -> (1, 1088, 1)
        # n_samples = 32
        # (1088, 1) -> (32, 1088, 1)
        if n_samples < 1:
            raise ValueError('n_samples must be >= 1')
        x_arr = []
        y_arr = []
        for i in range(n_samples):
            xi, yi = self.generate()
            x_arr.append(xi)
            y_arr.append(yi)
        x = np.stack(x_arr)
        y = np.stack(y_arr)
        return x, y

    def generator(self, n_msigs, batch_size, training=False):
        x, y, indices, n_batches = self.generate_group(n_msigs)
        i = 0
        while True:
            # TODO: figure out how to use a threading lock with a data generator.
            # with self.lock:
            if i >= n_batches:
                if training:
                    x, y, indices, n_batches = self.generate_group(n_msigs)
                i = 0
            idx = indices[i * batch_size:(i + 1) * batch_size]
            i += 1
            yield x[idx], y[idx]

    # def __next__(self):
    #     return self.next()
    #
    # def next(self):
    #     # with self.lock:
    #     if self.group_index == 0:
    #         self.X, self.y, self.group_indices, n = self.generate_group(self.n_groups)
    #     self.group_index = (self.group_index + 1) % (len(self.X) // self.batch_size)
    #     idx = self.group_indices[self.group_index * self.batch_size:(self.group_index + 1) * self.batch_size]
    #     return self.X[idx], self.y[idx]

    def save_config(self):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.config_filename, 'w') as ofs:
            json.dump(self.config_dict, ofs, indent=4)
