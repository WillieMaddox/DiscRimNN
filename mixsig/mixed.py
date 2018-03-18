import os
import sys
import json
import string
import random
import numpy as np
from .waves import Wave, WaveProperty


class MixedSignal:
    def __init__(self, time_coeffs, sig_coeffs, msig_coeffs=None, run_label='__default__', method='sliding'):
        self.signals = None
        self.inputs = None
        self.labels = None
        self.classes = None
        self.one_hots = None
        self.mixed_signal = None
        self.time_coeffs = time_coeffs
        self.sig_coeffs = sig_coeffs
        self.msig_coeffs = msig_coeffs
        self.run_label = run_label
        self.name = 'Mixed'
        self.method = method
        self.n_timestamps = time_coeffs['n_timestamps']
        self.n_timesteps = time_coeffs['n_timesteps']
        if method == 'boxcar':
            assert self.n_timestamps % self.n_timesteps == 0
        self.n_signals = len(sig_coeffs)
        self.timestamps = np.linspace(time_coeffs['start'], time_coeffs['stop'], self.n_timestamps)

        self.mixed_signal_props = {}
        for prop_name, coeffs in msig_coeffs.items():
            self.mixed_signal_props[prop_name] = WaveProperty(coeffs)

        self.signal_objects = []
        for coeffs in sig_coeffs:
            self.signal_objects.append(Wave(self.timestamps, **coeffs))

        self.out_dir = os.path.join(os.getcwd(), 'out', self.run_label)
        os.makedirs(self.out_dir, exist_ok=True)

    def __len__(self):
        return self.n_signals

    def generate_property_values(self):
        prop_vals = {}
        for name, prop in self.mixed_signal_props.items():
            prop_vals[name] = prop.generate()
        return prop_vals

    def generate(self):
        self._generate()
        if self.method == 'sliding':
            return self.generate_sliding()
        elif self.method == 'boxcar':
            return self.generate_boxcar()
        else:
            raise ValueError('improper method: {}. Use "sliding" or "boxcar"')

    def _generate(self):
        shuffled_indexes = np.arange(self.n_timestamps)
        np.random.shuffle(shuffled_indexes)
        self.classes = np.zeros(self.n_timestamps, dtype=int)
        for s in range(self.n_signals):
            self.classes[np.where(shuffled_indexes < s * self.n_timestamps // self.n_signals)] += 1
        self.one_hots = np.zeros((self.n_timestamps, self.n_signals), dtype=float)
        self.one_hots[(np.arange(self.n_timestamps), self.classes)] = 1

        mixed_prop_vals = self.generate_property_values()
        signals = np.empty((self.n_signals, self.n_timestamps))
        for i, signal in enumerate(self.signal_objects):
            signal.generate(**mixed_prop_vals)
            signals[i, :] = signal()

        self.mixed_signal = np.sum(self.one_hots.T * signals, axis=0)

        # self.inputs = np.vstack((self.timestamps, self.mixed_signal)).T
        # self.inputs = self.inputs.reshape(self.n_timesteps, 2, 1)
        # self.inputs = self.inputs.reshape(self.n_timesteps, 1, 2)
        # self.inputs = self.inputs.reshape(1, self.n_timesteps, 2)
        # self.inputs = self.inputs.reshape(self.n_timesteps, 2)

        # self.inputs = self.mixed_signal.reshape(self.n_timestamps, 1, 1)
        # self.inputs = self.mixed_signal.reshape(1, self.n_timesteps, 1)

    def generate_sliding(self):
        n_samples = self.n_timestamps - (self.n_timesteps - 1)
        mixed_signal = np.zeros((n_samples, self.n_timesteps))
        for i in range(self.n_timesteps):
            mixed_signal[:, i] = self.mixed_signal[i:i+n_samples]
        self.inputs = mixed_signal.reshape(n_samples, self.n_timesteps, 1)
        self.labels = self.one_hots[(self.n_timesteps - 1):]
        return self.inputs, self.labels

    def generate_boxcar(self):
        n_samples = self.n_timestamps // self.n_timesteps
        self.inputs = self.mixed_signal.reshape((n_samples, self.n_timesteps, 1))
        labels = self.one_hots.reshape((n_samples, self.n_timesteps, self.n_signals))
        labels = labels[:, -1, :]
        self.labels = labels.reshape(n_samples, self.n_signals)
        return self.inputs, self.labels

    def generate_batch(self, batch_size):
        x_batch = np.empty((batch_size, *self.inputs.shape))
        y_batch = np.empty((batch_size, *self.one_hots.shape))
        for i in range(batch_size):
            x, y = self.generate()
            x_batch[i] = x
            y_batch[i] = y
        return x_batch, y_batch

    def save_config(self):
        config_dict = {
            'run_label': self.run_label,
            'method': self.method,
            'time_coeffs': self.time_coeffs,
            'sig_coeffs': self.sig_coeffs,
            'msig_coeffs': self.msig_coeffs
        }
        filename = os.path.join(self.out_dir, 'signal_config.json')
        with open(filename, 'w') as ofs:
            json.dump(config_dict, ofs, indent=4)
