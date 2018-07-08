import os
import json
import threading
import numpy as np
# from .utils import TimeSequenceCoeffs
from .utils import factors
from .utils import get_datetime_now
from .utils import timesequence_generator
# from .waves import WaveProps
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
                 sequence_type='many2one',
                 run_label=None,
                 name='Mixed',
                 n_groups=5):

        self.lock = threading.Lock()
        self.X = None
        self.y = None
        self.n_groups = n_groups
        self.group_index = 0
        self.group_indices = None
        self._n_classes = None

        # Should these be properties?
        self.labels = None
        self.one_hots = None
        self.mixed_signal = None

        self.batch_size = batch_size
        self.window_size = window_size
        self.window_type = window_type.lower()
        self.network_type = network_type
        self.sequence_type = sequence_type
        self.name = name

        assert self.network_type in ('MLP', 'RNN')
        assert self.window_type in ('sliding', 'boxcar')
        assert self.sequence_type in ('many2one', 'many2many', 'many2one+time')

        if 'time' in msig_coeffs:
            self.sequence_generator = timesequence_generator(**msig_coeffs['time'])

        self.mixed_signal_props = {}
        for prop_name, coeffs in msig_coeffs.items():
            if prop_name == 'amplitude':
                self.mixed_signal_props[prop_name] = Amplitude(**coeffs)
            elif prop_name == 'frequency':
                self.mixed_signal_props[prop_name] = Frequency(**coeffs)
            elif prop_name == 'offset':
                self.mixed_signal_props[prop_name] = Offset(**coeffs)
            elif prop_name == 'phase':
                self.mixed_signal_props[prop_name] = Phase(**coeffs)
            elif prop_name == 'time':
                pass
            else:
                print(f'got unexpected msig_coeffs {prop_name}')

        self.waves = [Wave(**coeffs) for coeffs in sigs_coeffs]

        self.n_classes = len(self.waves)

        run_label = run_label or get_datetime_now()

        self.config_dict = {
            'run_label': run_label,
            'window_size': window_size,
            'window_type': window_type,
            'network_type': network_type,
            'sequence_type': sequence_type,
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

    def _generate_signals(self):
        """ Generate waves from property values."""
        # First process the timestamp dependent waves.  (i.e. make a mixed signal wave.)
        # generate new timestamps
        timestamps = self.sequence_generator()
        n_timestamps = len(timestamps)
        classes = np.array([c for c, wave in enumerate(self.waves) if not wave.is_independent])
        n_classes = len(classes)

        # create a uniform distribution for class labels -> np.array([2,1,3, ... ,1])
        shuffled_indexes = np.arange(n_timestamps)
        np.random.shuffle(shuffled_indexes)
        labels = np.zeros(n_timestamps, dtype=int)
        for c in range(n_classes):
            labels[np.where(shuffled_indexes < c * n_timestamps // n_classes)] += 1

        # create one-hots from labels -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
        one_hots = np.zeros((n_timestamps, n_classes), dtype=float)
        one_hots[(np.arange(n_timestamps), labels)] = 1

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

        # Now we want to sort the labels and mixed_signal chronologically.
        sorted_indices = np.argsort(timestamps)

        # clip data from the left so that it divisible by batch_size.
        if self.window_type == 'sliding':
            chop_index = (len(timestamps) - self.window_size + 1) % self.batch_size
        else:
            assert len(timestamps) >= self.window_size * self.batch_size
            fact_a = factors(self.batch_size)
            fact_b = factors(self.window_size)
            gcm = max(fact_a.intersection(fact_b))
            chop_index = len(timestamps) % (self.window_size * self.batch_size // gcm)

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

    def _create_label_distribution(self):
        """
        Create a distribution of ints which represent class labels.
        self.labels -> np.array([2,1,3, ... ,1])
        """
        shuffled_indexes = np.arange(self.n_timestamps)
        np.random.shuffle(shuffled_indexes)
        self.labels = np.zeros(self.n_timestamps, dtype=int)
        for s in range(self.n_classes):
            self.labels[np.where(shuffled_indexes < s * self.n_timestamps // self.n_classes)] += 1

    def _create_one_hots_from_labels(self):
        """
        Create one-hot vector from the class label distribution.
        self.one_hots -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
        """
        self.one_hots = np.zeros((self.n_timestamps, self.n_classes), dtype=float)
        self.one_hots[(np.arange(self.n_timestamps), self.labels)] = 1

    def _generate_signals_old(self):
        """ Generate waves from property values."""
        # generate new timestamps
        self.timestamps = self.sequence_generator()
        # generate new values for each mixed signal property.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}
        # generate new single waves.
        self._signals = np.vstack([wave.generate(**props) for wave in self.waves])

    def generate(self):
        self._generate()
        if self.window_type == 'sliding':
            self.generate_sliding()
        elif self.window_type == 'boxcar':
            self.generate_boxcar()
        else:
            raise ValueError('improper window_type: {}. Use "sliding" or "boxcar"')
        return self.X, self.y

    def _generate(self):

        self._generate_signals()
        self._create_one_hots_from_labels()

    def generate_sliding_new(self):

        if self.network_type == 'MLP':
            if self.sequence_type == 'many2one':
                # MLP: many to one
                self.X = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.y = self.one_hots[(self.window_size - 1):]
            else:
                raise NotImplementedError

        else:
            if self.sequence_type == 'many2one':
                # RNN: many to one
                self.X = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.y = self.one_hots[(self.window_size - 1):]

            elif self.sequence_type == 'many2many':
                # RNN: many to many
                self.X = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.y = np.lib.stride_tricks.as_strided(
                    self.one_hots,
                    shape=(self.n_samples, self.window_size, self.n_classes),
                    strides=(self.n_classes * self.one_hots.strides[1], self.one_hots.strides[0], self.one_hots.strides[1]),
                )
            else:
                raise NotImplementedError

    def generate_sliding(self):

        # TODO: unit tests to make sure all these pathways are correct.
        if self.network_type == 'MLP':
            if self.sequence_type == 'many2one':
                # MLP: many to one
                self.X = np.zeros((self.n_samples, self.window_size))
                for i in range(self.window_size):
                    self.X[:, i] = self.mixed_signal[i:i + self.n_samples]
                self.y = self.one_hots[(self.window_size - 1):]
            elif self.sequence_type == 'many2many':
                # MLP: many to many (1088, 100, 1) (1088, 100, 3)
                self.X = np.zeros((self.n_samples, self.window_size, 1))
                self.y = np.zeros((self.n_samples, self.window_size, self.n_classes))
                for i in range(self.window_size):
                    self.X[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
                    self.y[:, i] = self.one_hots[i:i + self.n_samples]
            else:
                raise NotImplementedError
        else:
            if self.sequence_type == 'many2one':
                # RNN: many to one (1088, 100, 1) (1088, 3)
                self.X = np.zeros((self.n_samples, self.window_size, 1))
                for i in range(self.window_size):
                    self.X[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
                self.y = self.one_hots[(self.window_size - 1):]
            elif self.sequence_type == 'many2one+time':
                # RNN: many to one (1088, 100, 2) (1088, 3)
                self.X = np.zeros((self.n_samples, self.window_size, 2))
                for i in range(self.window_size):
                    self.X[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
                    self.X[:, i, 1] = self.timestamps[i + self.n_samples - 1] - self.timestamps[i:i + self.n_samples]
                self.y = self.one_hots[(self.window_size - 1):]
            elif self.sequence_type == 'many2many':
                # RNN: many to many (1088, 100, 1) (1088, 100, 3)
                self.X = np.zeros((self.n_samples, self.window_size, 1))
                self.y = np.zeros((self.n_samples, self.window_size, self.n_classes))
                for i in range(self.window_size):
                    self.X[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
                    self.y[:, i] = self.one_hots[i:i + self.n_samples]
            else:
                raise NotImplementedError

    def generate_boxcar(self):

        if self.network_type == 'MLP':
            if self.sequence_type == 'many2one':
                # MLP: many to one
                self.X = self.mixed_signal.reshape((self.n_samples, self.window_size))
                labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_classes))
                labels = labels[:, -1, :]
                self.y = labels.reshape(self.n_samples, self.n_classes)
            elif self.sequence_type == 'many2many':
                # MLP: many to many
                self.X = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                self.y = self.one_hots.reshape((self.n_samples, self.window_size, self.n_classes))
            else:
                raise NotImplementedError
        else:
            if self.sequence_type == 'many2one':
                # RNN: many to one
                self.X = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_classes))
                labels = labels[:, -1, :]
                self.y = labels.reshape(self.n_samples, self.n_classes)
            elif self.sequence_type == 'many2many':
                # RNN: many to many
                self.X = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                self.y = self.one_hots.reshape((self.n_samples, self.window_size, self.n_classes))
            else:
                raise NotImplementedError

    def generate_group(self, n_msigs, shuffle_inplace=False):
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
