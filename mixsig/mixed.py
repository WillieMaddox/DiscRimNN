import os
import json
import numpy as np
from .utils import timesequence_generator
from .waves import Wave
from .waves import Amplitude
from .waves import Frequency
from .waves import Offset
from .waves import Phase


class MixedSignal:
    def __init__(self,
                 sigs_coeffs,
                 msig_coeffs=None,
                 window_size=1,
                 window_method='sliding',
                 run_label='default',
                 net_type='RNN',
                 model='many2one',
                 name='Mixed'):

        self._signals = None
        self._signal_names = None
        self._signal_colors = None

        # Should these be properties?
        self.inputs = None
        self.labels = None
        self.classes = None
        self.one_hots = None
        self.mixed_signal = None

        self.window_size = window_size
        self.window_method = window_method.lower()
        self.net_type = net_type
        self.model = model
        self.name = name

        assert self.net_type in ('MLP', 'RNN')
        assert self.model in ('many2one', 'many2many')
        assert self.window_method in ('sliding', 'boxcar')

        if 'time' in msig_coeffs:
            self.sequence_generator = timesequence_generator(**msig_coeffs['time'])
            self.timestamps = self.sequence_generator()
        self.n_timestamps = len(self.timestamps)

        self.t_min = msig_coeffs['time']['t_min']
        self.t_max = msig_coeffs['time']['t_max']

        if self.window_method == 'sliding':
            self.n_samples = self.n_timestamps - self.window_size + 1
        else:
            assert self.n_timestamps % self.window_size == 0
            self.n_samples = self.n_timestamps // self.window_size

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

        self.signal_objects = [Wave(self.timestamps, **coeffs) for coeffs in sigs_coeffs]

        self.n_signals = len(self.signals)

        self.config_dict = {
            'run_label': run_label,
            'window_size': window_size,
            'window_method': window_method,
            'sigs_coeffs': sigs_coeffs,
            'msig_coeffs': msig_coeffs,
        }

        # TODO: What's the appropriate way to assign the out_dir (regular functionality, unit tests, etc.)
        # Relative to the directory of the calling script?
        # Relative to the directory of this module?
        # Relative to the root of the project directory?

        self.out_dir = os.path.join(os.getcwd(), 'out', run_label)
        os.makedirs(self.out_dir, exist_ok=True)
        self.config_filename = os.path.join(self.out_dir, 'mixed_signal_config.json')

    @property
    def signals(self):
        if self._signals is None:
            self._generate_signals()
        return self._signals

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

    def _generate_signals(self):
        """ Generate signals from property values."""
        # generate new timestamps
        self.timestamps = self.sequence_generator()
        # generate new values for each mixed signal property.
        props = {name: prop() for name, prop in self.mixed_signal_props.items()}
        # generate new single signals.
        self._signals = np.vstack([sig.generate(**props) for sig in self.signal_objects])

    @property
    def signal_names(self):
        if self._signal_names is None:
            self._signal_names = np.hstack([sig.name for sig in self.signal_objects])
        return self._signal_names

    @property
    def signal_colors(self):
        if self._signal_colors is None:
            self._signal_colors = np.hstack([sig.color for sig in self.signal_objects])
        return self._signal_colors

    def generate(self):
        self._generate()
        if self.window_method == 'sliding':
            self.generate_sliding()
        elif self.window_method == 'boxcar':
            self.generate_boxcar()
        else:
            raise ValueError('improper window_method: {}. Use "sliding" or "boxcar"')
        return self.inputs, self.labels

    def _generate(self):

        self._create_class_distribution()
        self._create_one_hots_from_classes()
        self._generate_signals()

        self.mixed_signal = np.sum(self.one_hots.T * self.signals, axis=0)

        # self.inputs = np.vstack((self.timestamps, self.mixed_signal)).T
        # self.inputs = self.inputs.reshape(self.window_size, 2, 1)
        # self.inputs = self.inputs.reshape(self.window_size, 1, 2)
        # self.inputs = self.inputs.reshape(1, self.window_size, 2)
        # self.inputs = self.inputs.reshape(self.window_size, 2)

        # self.inputs = self.mixed_signal.reshape(self.n_timestamps, 1, 1)
        # self.inputs = self.mixed_signal.reshape(1, self.window_size, 1)

    def generate_sliding_new(self):

        if self.net_type == 'MLP':
            if self.model == 'many2one':
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
            if self.model == 'many2one':
                # RNN: many to one
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.window_size, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = self.one_hots[(self.window_size - 1):]

            elif self.model == 'many2many':
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

    def generate_sliding(self):

        if self.net_type == 'MLP':
            if self.model == 'many2one':
                # MLP: many to one
                inputs = np.zeros((self.n_samples, self.window_size))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i+self.n_samples]
                self.inputs = inputs
                self.labels = self.one_hots[(self.window_size - 1):]
            else:
                # MLP: many to many
                raise NotImplementedError

        else:
            if self.model == 'many2one':
                # RNN: many to one
                inputs = np.zeros((self.n_samples, self.window_size))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i+self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.window_size, 1)
                self.labels = self.one_hots[(self.window_size - 1):]

            else:
                # RNN: many to many
                inputs = np.zeros((self.n_samples, self.window_size))
                labels = np.zeros((self.n_samples, self.window_size, self.n_signals))
                for i in range(self.window_size):
                    inputs[:, i] = self.mixed_signal[i:i + self.n_samples]
                    labels[:, i] = self.one_hots[i:i + self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.window_size, 1)
                self.labels = labels.reshape(self.n_samples, self.window_size, self.n_signals)

    def generate_boxcar(self):

        if self.net_type == 'MLP':
            if self.model == 'many2one':
                # MLP: many to one
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size))
                labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))
                labels = labels[:, -1, :]
                self.labels = labels.reshape(self.n_samples, self.n_signals)
            else:
                # MLP: many to many
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                self.labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))

        else:
            if self.model == 'many2one':
                # RNN: many to one
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))
                labels = labels[:, -1, :]
                self.labels = labels.reshape(self.n_samples, self.n_signals)

            else:
                # RNN: many to many
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.window_size, 1))
                self.labels = self.one_hots.reshape((self.n_samples, self.window_size, self.n_signals))

    # def generate_batch(self, batch_size):
    #     x_batch = np.empty((batch_size, *self.inputs.shape))
    #     y_batch = np.empty((batch_size, *self.one_hots.shape))
    #     for i in range(batch_size):
    #         x, y = self.generate()
    #         x_batch[i] = x
    #         y_batch[i] = y
    #     return x_batch, y_batch

    def save_config(self):
        self.config_filename = os.path.join(self.out_dir, 'mixed_signal_config.json')
        with open(self.config_filename, 'w') as ofs:
            json.dump(self.config_dict, ofs, indent=4)
