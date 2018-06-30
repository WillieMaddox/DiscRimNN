import os
import json
import threading
import numpy as np
# from .utils import TimeSequenceCoeffs
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
        self._signals = None
        self._wave_names = None
        self._wave_colors = None

        # Should these be properties?
        self.inputs = None
        self.labels = None
        self.classes = None
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

        self.n_signals = len(self.waves)
        run_label = run_label or get_datetime_now()

        self.config_dict = {
            'run_label': run_label,
            'window_size': window_size,
            'window_type': window_type,
            'sequence_type': sequence_type,
            'sigs_coeffs': sigs_coeffs,
            'msig_coeffs': msig_coeffs,
        }

        # TODO: What's the appropriate way to assign the out_dir (regular functionality, unit tests, etc.)
        # TODO: Relative to the root directory of this project?
        # TODO: Relative to the directory of the calling script?
        # TODO: Relative to the directory of this module?
        self.out_dir = os.path.join(os.getcwd(), 'out', run_label)
        self.config_filename = os.path.join(self.out_dir, 'mixedsignal_config.json')
        self.model_weights_filename = os.path.join(self.out_dir, 'model_weights.h5')
        self.training_stats_filename = os.path.join(self.out_dir, 'training_stats.csv')

    @property
    def signals(self):
        if self._signals is None:
            self._generate_signals()
        return self._signals

    def _generate_signals(self):
        """ Generate waves from property values."""
        # generate new timestamps
        timestamps = self.sequence_generator()
        n_timestamps = len(timestamps)
        classes = np.array([c for c, wave in enumerate(self.waves) if not wave.is_independent])
        n_classes = len(classes)

        shuffled_indexes = np.arange(n_timestamps)
        np.random.shuffle(shuffled_indexes)
        labels = np.zeros(n_timestamps, dtype=int)
        for s in range(n_classes):
            labels[np.where(shuffled_indexes < s * n_timestamps // n_classes)] += 1

        one_hots = np.zeros((n_timestamps, n_classes), dtype=float)
        one_hots[(np.arange(n_timestamps), labels)] = 1

        # generate new mixed signal properties.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}

        # generate new individual waves.
        for wave in self.waves:
            wave.generate(timestamps, **props)

        signals = np.vstack([wave.sample for wave in self.waves if not wave.is_independent])
        mixed_signal = np.sum(one_hots.T * signals, axis=0)

        labels = classes[labels]
        for c, wave in enumerate(self.waves):
            if wave.is_independent:
                timestamps = np.append(timestamps, wave.timestamps)
                mixed_signal = np.append(mixed_signal, wave.sample)
                labels = np.append(labels, np.zeros(len(wave), dtype=int) + c)

        assert len(timestamps) == len(mixed_signal) == len(labels)

        sorted_indices = np.argsort(timestamps)

        # trim away the data so we can later chop it up evenly with our batch size.

        if self.window_type == 'sliding':
            chop_index = (len(timestamps) - self.window_size + 1) % self.batch_size
        else:
            chop_index = len(timestamps) % (self.window_size * self.batch_size)

        sorted_indices = sorted_indices[chop_index:]

        self.timestamps = timestamps[sorted_indices]
        self.n_timestamps = len(self.timestamps)
        self.mixed_signal = mixed_signal[sorted_indices]
        self.classes = labels[sorted_indices]

        if self.window_type == 'sliding':
        self.t_min = self.timestamps[0]
        self.t_max = self.timestamps[-1]

            self.n_samples = self.n_timestamps - self.window_size + 1
        else:
            assert self.n_timestamps % self.window_size == 0
            self.n_samples = self.n_timestamps // self.window_size

    def _create_class_distribution(self):
        """
        Create a distribution of ints which represent class labels.
        self.classes -> np.array([2,1,3, ... ,1])
        """
        shuffled_indexes = np.arange(self.n_timestamps)
        np.random.shuffle(shuffled_indexes)
        self.classes = np.zeros(self.n_timestamps, dtype=int)
        for s in range(self.n_signals):
            self.classes[np.where(shuffled_indexes < s * self.n_timestamps // self.n_signals)] += 1

    def _create_one_hots_from_classes(self):
        """
        Create one-hot vector from the class label distribution.
        self.one_hots -> np.array([[0,0,1,0], [0,1,0,0], [0,0,0,1], ... ,[0,1,0,0]])
        """
        self.one_hots = np.zeros((self.n_timestamps, self.n_signals), dtype=float)
        self.one_hots[(np.arange(self.n_timestamps), self.classes)] = 1

    def _generate_signals_old(self):
        """ Generate waves from property values."""
        # generate new timestamps
        self.timestamps = self.sequence_generator()
        # generate new values for each mixed signal property.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}
        # generate new single waves.
        self._signals = np.vstack([sig.generate(**props) for sig in self.waves])

    @property
    def signal_names(self):
        if self._wave_names is None:
            self._wave_names = np.hstack([sig.name for sig in self.waves])
        return self._wave_names

    @property
    def signal_colors(self):
        if self._wave_colors is None:
            self._wave_colors = np.hstack([sig.color for sig in self.waves])
        return self._wave_colors

    def generate(self):
        self._generate()
        if self.window_type == 'sliding':
            self.generate_sliding()
        elif self.window_type == 'boxcar':
            self.generate_boxcar()
        else:
            raise ValueError('improper window_type: {}. Use "sliding" or "boxcar"')
        return self.inputs, self.labels

    def _generate(self):

        self._generate_signals()
        self._create_one_hots_from_classes()

        # self.inputs = np.vstack((self.timestamps, self.mixed_signal)).T
        # self.inputs = self.inputs.reshape(self.window_size, 2, 1)
        # self.inputs = self.inputs.reshape(self.window_size, 1, 2)
        # self.inputs = self.inputs.reshape(1, self.window_size, 2)
        # self.inputs = self.inputs.reshape(self.window_size, 2)

        # self.inputs = self.mixed_signal.reshape(self.n_timestamps, 1, 1)
        # self.inputs = self.mixed_signal.reshape(1, self.window_size, 1)

    def generate_sliding_new(self):

        if self.network_type == 'MLP':
            if self.sequence_type == 'many2one':
                # MLP: many to one
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = self.one_hots[(self.window_size - 1):]
            else:
                raise NotImplementedError

        else:
            if self.sequence_type == 'many2one':
                # RNN: many to one
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = self.one_hots[(self.window_size - 1):]

            elif self.sequence_type == 'many2many':
                # RNN: many to many
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = np.lib.stride_tricks.as_strided(
                    self.one_hots,
                    shape=(self.n_samples, self.window_size, self.n_signals),
                    strides=(self.n_signals * self.one_hots.strides[1], self.one_hots.strides[0], self.one_hots.strides[1]),
                )
            else:
                raise NotImplementedError

    def generate_sliding(self):

        if self.network_type == 'MLP':
            if self.sequence_type == 'many2one':
                # MLP: many to one
                inputs = np.zeros((self.n_samples, self.window_size))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i+self.n_samples]
                self.inputs = inputs
                self.labels = self.one_hots[(self.window_size - 1):]
            elif self.sequence_type == 'many2many':
                # MLP: many to many
                inputs = np.zeros((self.n_samples, self.window_size))
                labels = np.zeros((self.n_samples, self.window_size, self.n_signals))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i + self.n_samples]
                    labels[:, i] = self.one_hots[i:i + self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.window_size, 1)
                self.labels = labels.reshape(self.n_samples, self.window_size, self.n_signals)

            else:
                raise NotImplementedError
        else:
            if self.sequence_type == 'many2one':
                # RNN: many to one (1088, 100, 1) (1088, 3)
                inputs = np.zeros((self.n_samples, self.window_size))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i+self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.window_size, 1)
                self.labels = self.one_hots[(self.window_size - 1):]
            elif self.sequence_type == 'many2one+time':
                # RNN: many to one (1088, 100, 1) (1088, 3)

                inputs = np.zeros((self.n_samples, self.window_size, 2))
                for i in range(self.window_size):
                    inputs[:, i, 0] = self.mixed_signal[i:i + self.n_samples]
                    inputs[:, i, 1] = self.timestamps[i + self.n_samples - 1] - self.timestamps[i:i + self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.window_size, 2)
                self.labels = self.one_hots[(self.window_size - 1):]
            elif self.sequence_type == 'many2many':
                # RNN: many to many (1088, 100, 1) (1088, 100, 3)

                inputs = np.zeros((self.n_samples, self.window_size))
                labels = np.zeros((self.n_samples, self.window_size, self.n_signals))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i + self.n_samples]
                    labels[:, i] = self.one_hots[i:i + self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.window_size, 1)
                self.labels = labels.reshape(self.n_samples, self.window_size, self.n_signals)
            else:
                raise NotImplementedError

    def generate_boxcar(self):

        if self.network_type == 'MLP':
            if self.sequence_type == 'many2one':
                # MLP: many to one
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size))
                labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))
                labels = labels[:, -1, :]
                self.labels = labels.reshape(self.n_samples, self.n_signals)
            elif self.sequence_type == 'many2many':
                # MLP: many to many
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                self.labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))
            else:
                raise NotImplementedError
        else:
            if self.sequence_type == 'many2one':
                # RNN: many to one
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))
                labels = labels[:, -1, :]
                self.labels = labels.reshape(self.n_samples, self.n_signals)
            elif self.sequence_type == 'many2many':
                # RNN: many to many
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                self.labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))
            else:
                raise NotImplementedError

    def __next__(self):
        return self.next()

    def next(self):
        # with self.lock:
        if self.group_index == 0:
            self.X, self.y, self.group_indices, n = self.generate_groups(self.n_groups)
        self.group_index = (self.group_index + 1) % (len(self.X) // self.batch_size)
        idx = self.group_indices[self.group_index * self.batch_size:(self.group_index + 1) * self.batch_size]
        return self.X[idx], self.y[idx]

    def generate_groups(self, n):
        x, y = self.generate()
        for i in range(n - 1):
            xi, yi = self.generate()
            x = np.vstack((x, xi))
            y = np.vstack((y, yi))

        n_samples = len(x)
        n_batches = n_samples // self.batch_size
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        return x, y, indices, n_batches

    def generator(self, n_groups, batch_size, training=False):

        x, y, indices, n = self.generate_groups(n_groups)
        i = 0
        while True:
            # with self.lock:
            if i >= n:
                if training:
                    x, y, indices, n = self.generate_groups(n_groups)
                i = 0
            idx = indices[i * batch_size:(i + 1) * batch_size]
            i += 1
            yield x[idx], y[idx]

    def save_config(self):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.config_filename, 'w') as ofs:
            json.dump(self.config_dict, ofs, indent=4)
