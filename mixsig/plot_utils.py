import itertools
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion matrix', cmap_str='Blues', figsize=(8, 6), filename=None):
    if labels is not None:
        assert len(labels) >= np.max(y_pred)
        ymap = {idx: label for idx, label in enumerate(labels)}
        y_true = [ymap[yi] for yi in y_true]
        y_pred = [ymap[yi] for yi in y_pred]
        labels = [ymap[yi] for yi in labels]

    ticks = labels or list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:np.max(y_pred) + 1])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = np.trace(cm) / np.sum(cm)
    size, _ = cm.shape
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_norm = cm / cm_sum

    fig = plt.figure(figsize=figsize)
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.get_cmap(cmap_str))
    plt.colorbar()

    for i, j in itertools.product(range(size), range(size)):
        text = f'{cm[i, j]}\n{cm_norm[i, j]:.1%}' if i == j else f'{cm[i, j]}'
        color = 'white' if cm_norm[i, j] > 0.5 else 'black'
        plt.text(j, i, text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 color=color)

    plt.title(title)
    plt.xticks(range(size), ticks, rotation=20, horizontalalignment='right')
    plt.yticks(range(size), ticks, rotation=70, horizontalalignment='right')
    plt.ylabel('Actual')
    plt.xlabel(f'Prediction ({accuracy:.1%})')
    plt.tight_layout()
    plt.draw()

    if filename is not None:
        plt.savefig(filename)

    plt.show()
    plt.close(fig)


