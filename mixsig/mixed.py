import os
import json
import numpy as np
from .utils import timesequence_generator
from .waves import Wave, WaveProperty
from .noise import OUNoise


class MixedSignal:
    def __init__(self,
                 time_coeffs,
                 sigs_coeffs,
                 msig_coeffs=None,
                 n_timesteps=1,
                 run_label='default',
                 method='sliding',
                 net_type='RNN',
                 model='many2one'):

        self._signals = None
        self._signal_names = None
        self._signal_colors = None

        # Should these be properties?
        self.inputs = None
        self.labels = None
        self.classes = None
        self.one_hots = None
        self.mixed_signal = None

        self.sequence_generator = timesequence_generator(**time_coeffs)
        self.timestamps = self.sequence_generator()
        self.n_timestamps = len(self.timestamps)
        self.time_start = time_coeffs['start']
        self.time_stop = time_coeffs['stop']
        self.n_timesteps = n_timesteps

        self.name = 'Mixed'
        self.method = method.lower()
        assert self.method in ('sliding', 'boxcar')
        if method.lower() == 'sliding':
            self.n_samples = self.n_timestamps - self.n_timesteps + 1
        else:
            assert self.n_timestamps % self.n_timesteps == 0
            self.n_samples = self.n_timestamps // self.n_timesteps

        self.net_type = net_type
        assert self.net_type in ('MLP', 'RNN')
        self.model = model
        assert self.model in ('many2one', 'many2many')

        self.mixed_signal_props = {}
        for prop_name, coeffs in msig_coeffs.items():
            self.mixed_signal_props[prop_name] = WaveProperty(**coeffs)

        self.signal_objects = []
        for sig, coeffs in sigs_coeffs.items():
            if sig == 'waves':
                for c in coeffs:
                    self.signal_objects.append(Wave(self.timestamps, **c))
            elif sig == 'noise':
                noise_coeffs = {}
                for c in coeffs:
                    for k, v in c.items():
                        if k not in noise_coeffs:
                            noise_coeffs[k] = []
                        noise_coeffs[k].append(v)

                self.signal_objects.append(OUNoise(self.n_timestamps, **noise_coeffs))

        self.n_signals = len(self.signals)

        self.config_dict = {
            'run_label': run_label,
            'method': self.method,
            'time_coeffs': time_coeffs,
            'sigs_coeffs': sigs_coeffs,
            'msig_coeffs': msig_coeffs
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
        """ Create a distribution of ints which represent class labels."""
        shuffled_indexes = np.arange(self.n_timestamps)
        np.random.shuffle(shuffled_indexes)
        self.classes = np.zeros(self.n_timestamps, dtype=int)
        for s in range(self.n_signals):
            self.classes[np.where(shuffled_indexes < s * self.n_timestamps // self.n_signals)] += 1

    def _create_one_hots_from_classes(self):
        """ Create one-hot vector from the class label distribution."""
        self.one_hots = np.zeros((self.n_timestamps, self.n_signals), dtype=float)
        self.one_hots[(np.arange(self.n_timestamps), self.classes)] = 1

    def _generate_signals(self):
        """ Generate signals from property values."""
        # generate new timestamps
        self.timestamps = self.sequence_generator()
        # generate new values for each mixed signal property.
        props = {name: prop.generate() for name, prop in self.mixed_signal_props.items()}
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
        if self.method == 'sliding':
            self.generate_sliding()
        elif self.method == 'boxcar':
            self.generate_boxcar()
        else:
            raise ValueError('improper method: {}. Use "sliding" or "boxcar"')
        return self.inputs, self.labels

    def _generate(self):

        self._create_class_distribution()
        self._create_one_hots_from_classes()
        self._generate_signals()

        self.mixed_signal = np.sum(self.one_hots.T * self.signals, axis=0)

        # self.inputs = np.vstack((self.timestamps, self.mixed_signal)).T
        # self.inputs = self.inputs.reshape(self.n_timesteps, 2, 1)
        # self.inputs = self.inputs.reshape(self.n_timesteps, 1, 2)
        # self.inputs = self.inputs.reshape(1, self.n_timesteps, 2)
        # self.inputs = self.inputs.reshape(self.n_timesteps, 2)

        # self.inputs = self.mixed_signal.reshape(self.n_timestamps, 1, 1)
        # self.inputs = self.mixed_signal.reshape(1, self.n_timesteps, 1)

    def generate_sliding_new(self):

        if self.net_type == 'MLP':
            if self.model == 'many2one':
                # MLP: many to one
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.n_timesteps),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = self.one_hots[(self.n_timesteps - 1):]
            else:
                raise NotImplementedError

        else:
            if self.model == 'many2one':
                # RNN: many to one
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.n_timesteps, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = self.one_hots[(self.n_timesteps - 1):]

            elif self.model == 'many2many':
                # RNN: many to many
                self.inputs = np.lib.stride_tricks.as_strided(
                    self.mixed_signal,
                    shape=(self.n_samples, self.n_timesteps, 1),
                    strides=(self.mixed_signal.itemsize, self.mixed_signal.itemsize, self.mixed_signal.itemsize)
                )
                self.labels = np.lib.stride_tricks.as_strided(
                    self.one_hots,
                    shape=(self.n_samples, self.n_timesteps, self.n_signals),
                    strides=(self.n_signals * self.one_hots.strides[1], self.one_hots.strides[0], self.one_hots.strides[1]),
                )

    def generate_sliding(self):

        if self.net_type == 'MLP':
            if self.model == 'many2one':
                # MLP: many to one
                inputs = np.zeros((self.n_samples, self.n_timesteps))
                for i in range(self.n_timesteps):
                    inputs[:, i] = self.mixed_signal[i:i+self.n_samples]
                self.inputs = inputs
                self.labels = self.one_hots[(self.n_timesteps - 1):]
            else:
                # MLP: many to many
                raise NotImplementedError

        else:
            if self.model == 'many2one':
                # RNN: many to one
                inputs = np.zeros((self.n_samples, self.n_timesteps))
                for i in range(self.n_timesteps):
                    inputs[:, i] = self.mixed_signal[i:i+self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.n_timesteps, 1)
                self.labels = self.one_hots[(self.n_timesteps - 1):]

            else:
                # RNN: many to many
                inputs = np.zeros((self.n_samples, self.n_timesteps))
                labels = np.zeros((self.n_samples, self.n_timesteps, self.n_signals))
                for i in range(self.n_timesteps):
                    inputs[:, i] = self.mixed_signal[i:i + self.n_samples]
                    labels[:, i] = self.one_hots[i:i + self.n_samples]
                self.inputs = inputs.reshape(self.n_samples, self.n_timesteps, 1)
                self.labels = labels.reshape(self.n_samples, self.n_timesteps, self.n_signals)

    def generate_boxcar(self):

        if self.net_type == 'MLP':
            if self.model == 'many2one':
                # MLP: many to one
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.n_timesteps))
                labels = self.one_hots.reshape((self.n_samples, self.n_timesteps, self.n_signals))
                labels = labels[:, -1, :]
                self.labels = labels.reshape(self.n_samples, self.n_signals)
            else:
                # MLP: many to many
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.n_timesteps, 1))
                self.labels = self.one_hots.reshape((self.n_samples, self.n_timesteps, self.n_signals))

        else:
            if self.model == 'many2one':
                # RNN: many to one
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.n_timesteps, 1))
                labels = self.one_hots.reshape((self.n_samples, self.n_timesteps, self.n_signals))
                labels = labels[:, -1, :]
                self.labels = labels.reshape(self.n_samples, self.n_signals)

            else:
                # RNN: many to many
                self.inputs = self.mixed_signal.reshape((self.n_samples, self.n_timesteps, 1))
                self.labels = self.one_hots.reshape((self.n_samples, self.n_timesteps, self.n_signals))

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
