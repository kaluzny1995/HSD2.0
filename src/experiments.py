import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from .constants import LABELS, LABELS_SMALL

label_map = dict(zip(LABELS + ['overall'], LABELS_SMALL + ['all']))
measure_map = dict(zip(['a', 'p', 'p0', 'p1', 'r', 'r0', 'r1', 'f', 'f0', 'f1'],
                       ['Accuracy', 'Mean precision', 'Negatives precision', 'Positives precision', 'Mean recall',
                        'Negatives recall', 'Positives recall', 'Mean F measure', 'Negatives F measure',
                        'Positives F measure']))


def quality_measures(y_trues, y_preds):
    y_trues = list(y_trues.T) + list([y_trues.T.reshape(-1)])
    y_preds = list(y_preds.T) + list([y_preds.T.reshape(-1)])

    values = list([])

    for y_true, y_pred in zip(y_trues, y_preds):

        a = accuracy_score(y_true=y_true, y_pred=y_pred)
        p0, p1 = precision_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
        r0, r1 = recall_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
        f0, f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
        values.extend([a, p0, p1, r0, r1, f0, f1])

    return np.array(values)


def experimental_result_lines(data, series_names, measure='a', title=None, shrink_to_interval=None, save_file=None):
    assert measure in ['a', 'p', 'p0', 'p1', 'r', 'r0', 'r1', 'f', 'f0', 'f1'], f'Unknown measure: {measure}!'
    if not title:
        title = f'Experimental parameter reseaching {measure_map[measure]} ' \
                f'results for 7 hate speech classes and overall.'
    if not shrink_to_interval:
        shrink_to_interval = [0., 1.]

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    positions = list([tuple((i, j)) for i in range(2) for j in range(4)])
    lsmalls = LABELS_SMALL + ['all']
    labels = LABELS + ['overall']

    for i, (p, label, lsmall) in enumerate(zip(positions, labels, lsmalls)):
        df = data.iloc[:, :2]
        series_values = df.iloc[:, 1].unique()

        if measure == 'p':
            df[measure] = (data[f'{label_map[label]}_p0'].astype(float) + data[f'{label_map[label]}_p1'].astype(float)) / 2
        elif measure == 'r':
            df[measure] = (data[f'{label_map[label]}_r0'].astype(float) + data[f'{label_map[label]}_r1'].astype(float)) / 2
        elif measure == 'f':
            df[measure] = (data[f'{label_map[label]}_f0'].astype(float) + data[f'{label_map[label]}_f1'].astype(float)) / 2
        else:
            df[measure] = data[f'{label_map[label]}_{measure}'].astype(float)

        for sn, sv in zip(series_names, series_values):
            dff = df[df.iloc[:, 1] == sv]
            ax[p[0]][p[1]].plot(dff.iloc[:, 0], dff[measure], label=sn)

        ax[p[0]][p[1]].set_title(f'{label}')
        ax[p[0]][p[1]].set_ylim(shrink_to_interval)

    h, ln = ax[0][0].get_legend_handles_labels()
    fig.legend(h, ln, loc='upper right')

    fig.text(0, 0.5, f'{measure_map[measure].capitalize()} score', fontsize=16, va='center', rotation='vertical')
    fig.text(0.5, 0, f'{data.columns[0]}', fontsize=16, ha='center')

    fig.suptitle(title)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def experimental_measure_lines(data, series_names, measures=None, label='wyzywanie', title=None,
                               shrink_to_interval=None, save_file=None):
    assert len(data) // len(series_names) == int(
        len(data) / len(series_names)), f'Data size must be divisible by count of series names i.e. {len(series_names)}'
    if not title:
        title = f'Experimental parameter reseaching quality measure results for label "{label}"'
    if not shrink_to_interval:
        shrink_to_interval = [0., 1.]

    if not measures:
        measures = ['a', 'p', 'r', 'f']
    for m in measures:
        assert m in ['a', 'p', 'p0', 'p1', 'r', 'r0', 'r1', 'f', 'f0', 'f1'], f'Unknown measure: {m}!'
    fig, ax = plt.subplots(1, len(measures), figsize=(4 * len(measures), 4))

    df = data.iloc[:, :2]
    series_values = df.iloc[:, 1].unique()
    for m in measures:
        if m == 'p':
            df[m] = (data[f'{label_map[label]}_p0'].astype(float) + data[f'{label_map[label]}_p1'].astype(float)) / 2
        elif m == 'r':
            df[m] = (data[f'{label_map[label]}_r0'].astype(float) + data[f'{label_map[label]}_r1'].astype(float)) / 2
        elif m == 'f':
            df[m] = (data[f'{label_map[label]}_f0'].astype(float) + data[f'{label_map[label]}_f1'].astype(float)) / 2
        else:
            df[m] = data[f'{label_map[label]}_{m}'].astype(float)

    for i, m in enumerate(measures):
        for sn, sv in zip(series_names, series_values):
            dff = df[df.iloc[:, 1] == sv]
            ax[i].plot(dff.iloc[:, 0], dff[m], label=sn)

        ax[i].set_title(f'{measure_map[m]}')
        ax[i].set_ylim(shrink_to_interval)

    h, ln = ax[0].get_legend_handles_labels()
    fig.legend(h, ln, loc='upper right')

    fig.text(-0.01, 0.5, f'Score', fontsize=16, va='center', rotation='vertical')
    fig.text(0.5, -0.02, f'{data.columns[0]}', fontsize=16, ha='center')

    fig.suptitle(title)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()
