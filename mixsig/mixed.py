import os
import json
import numpy as np
from .utils import get_datetime_now
from .utils import timesequence_generator
from .utils import create_label_distribution
from .utils import create_one_hots_from_labels
from .waves import Amplitude
from .waves import Frequency
from .waves import Offset
from .waves import Phase
from .waves import Wave


class MixedSignal:
    def __init__(self,
                 sigs_coeffs,
                 msig_coeffs=None,
                 batch_size=1,
                 window_size=0,
                 window_type='sliding',
                 network_type='RNN',
                 sequence_code='t_tc',
                 stateful=False,
                 run_label=None,
                 n_groups=5,
                 name='Mixed'):

        self.group_index = 0
        self.group_indices = None
        self.n_groups = n_groups
        self.X = None
        self.y = None

        if stateful:
            assert batch_size > 0
        self.stateful = stateful
        self.batch_size = batch_size

        # Should these be properties?
        self.name = name
        self.labels = None
        self.one_hots = None
        self.mixed_signal = None
        self.n_timestamps = None
        self._n_samples = None

        self._window_size = None if window_size < 1 else window_size

        self.window_type = window_type.lower()
        assert self.window_type in ('sliding', 'boxcar', 'random')

        self.network_type = network_type
        self.sequence_code = sequence_code
        self._sequence_type = None
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
            'window_size': self._window_size,
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
        # generate new mixed signal properties.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}

        # get the class labels for the waves that will generate the mixed_signal
        classes = np.array([c for c, wave in enumerate(self.waves) if not wave.is_independent])
        # create a uniform distribution of class labels -> np.array([2,1,3, ... ,1])
        labels = create_label_distribution(len(timestamps), len(classes))
        # create one-hots from labels -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
        one_hots = create_one_hots_from_labels(labels, len(classes))

        # generate new individual waves.
        for wave in self.waves:
            wave.generate(timestamps, **props)

        signals = np.vstack([wave.sample for wave in self.waves if not wave.is_independent])
        mixed_signal = np.sum(one_hots.T * signals, axis=0)

        # make sure the labels align with the classes.
        # TODO: refactor this.  It's confusing.
        labels = classes[labels]

        # Now append the remaining independent waves to the end of the mixed_signal.
        for c, wave in enumerate(self.waves):
            if wave.is_independent:
                timestamps = np.append(timestamps, wave.timestamps)
                mixed_signal = np.append(mixed_signal, wave.sample)
                labels = np.append(labels, np.zeros(len(wave), dtype=int) + c)

        # Sanity check
        assert len(timestamps) == len(mixed_signal) == len(labels)

        # Store the indices to the ordered timestamps.
        # sorted_indices = np.argsort(timestamps)

        batch_size = self.batch_size if self.stateful else 1
        window_size = self.window_size if self.window_size is not None else 1

        # clip data from the left so that it divides batch_size evenly.
        if self.window_type == 'boxcar':
            assert len(timestamps) >= window_size * batch_size
            chop_index = len(timestamps) % (window_size * batch_size)
        else:  # ('sliding' and 'random')
            chop_index = (len(timestamps) - window_size + 1) % batch_size

        # Sort the labels and mixed_signal chronologically.
        sorted_indices = np.argsort(timestamps)[chop_index:]

        self.mixed_signal = mixed_signal[sorted_indices]
        self.labels = labels[sorted_indices]
        self.timestamps = timestamps[sorted_indices]
        self.n_timestamps = len(self.timestamps)
        self.t_min = self.timestamps[0]
        self.t_max = self.timestamps[-1]
        if self._window_size is None:
            self.window_size = self.n_timestamps

        # assert self.n_samples % self.batch_size == 0

    def generate(self):
        self._generate_signals()
        self.one_hots = create_one_hots_from_labels(self.labels, self.n_classes)
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
            self.X, self.y = self.generate_sliding()
        elif self.window_type == 'boxcar':
            self.X, self.y = self.generate_boxcar()
        elif self.window_type == 'random':
            self.X, self.y = self.generate_sliding()
        else:
            raise ValueError('Invalid window_type: {}. Use "sliding", "boxcar" or "random"')
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
            for j in range(self.window_size):
                X[:, j, 0] = self.mixed_signal[j:j + self.n_samples]
                X[:, j, 1] = self.timestamps[j + self.n_samples - 1] - self.timestamps[j:j + self.n_samples]
            y = self.one_hots[self.window_size - 1:]
            return X, y

        sequence_code = sequence_code or self.sequence_code
        X_code, y_code = sequence_code.split('_')

        if X_code == 't':
            X = self.mixed_signal
        elif X_code == 't1':
            X = self.mixed_signal[..., None]

        elif X_code == 'xw':
            if self.n_samples == 1:
                X = self.mixed_signal[None, ...]
            else:
                X = np.zeros((self.n_samples, self.window_size))
                for j in range(self.window_size):
                    X[:, j] = self.mixed_signal[j:j + self.n_samples]

        elif X_code == 'xw1':
            if self.n_samples == 1:
                X = self.mixed_signal[None, ..., None]
            else:
                X = np.zeros((self.n_samples, self.window_size, 1))
                for j in range(self.window_size):
                    X[:, j, 0] = self.mixed_signal[j:j + self.n_samples]

        else:
            raise NotImplementedError

        if y_code == 't':
            y = self.labels
        elif y_code == 'tc':
            y = self.one_hots

        elif y_code == 'xc':
            y = self.one_hots[self.window_size - 1:]
        elif y_code == 'xwc':
            y = np.zeros((self.n_samples, self.window_size, self.n_classes))
            for j in range(self.window_size):
                y[:, j] = self.one_hots[j:j + self.n_samples]
        else:
            raise NotImplementedError

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

    def generate_group(self, n_msigs, shuffle_inplace=False):
        # examples:
        # if n_samples == 1
        # (1088, 100, 1) -> (1 * 1088, 100, 1) -> (1088, 100, 1)
        # if n_samples == 32
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
        # if n_samples == 0
        # (1088, 1) -> (1088, 1)
        # if n_samples == 1
        # (1088, 1) -> (1, 1088, 1)
        # if n_samples == 32
        # (1088, 1) -> (32, 1088, 1)
        if n_samples < 1:
            raise ValueError('n_samples must be >= 1')
        x = []
        y = []
        for i in range(n_samples):
            xi, yi = self.generate()
            x.append(xi)
            y.append(yi)
        return np.stack(x), np.stack(y)

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
