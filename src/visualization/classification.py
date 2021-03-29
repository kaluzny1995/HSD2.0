import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

from ..constants import LABELS


def confusion_matrices(y_trues, y_preds,
                       title='Confusion matrices with F measures for all hate-speech classes with overall score.',
                       percentage=False,
                       show_acc=False,
                       save_file=None):
    assert y_trues.shape[1] == 7 and y_preds.shape[1] == 7, 'Length of true values and predictions must be exactly 7!'
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    positions = list([tuple((i, j)) for i in range(2) for j in range(4)])
    labels = LABELS + ['overall']
    y_trues = list(y_trues.T) + list([y_trues.T.reshape(-1)])
    y_preds = list(y_preds.T) + list([y_preds.T.reshape(-1)])

    for i, (p, label, y_true, y_pred) in enumerate(zip(positions, labels, y_trues, y_preds)):

        a = accuracy_score(y_true=y_true, y_pred=y_pred)
        f0, f1 = f1_score(y_true=y_true, y_pred=y_pred, labels=[0, 1], average=None, zero_division=1.)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
        if percentage:
            cm = np.array([[cmmm/np.sum(cmm) for cmmm in cmm] for cmm in cm])
        cm = cm.T

        msr = f' | acc.: {a:.2f}' if show_acc else ''
        ax[p[0]][p[1]].set_title(f'"{label}" | f0: {f0:.2f} | f1: {f1:.2f}')
        ax[p[0]][p[1]].set_ylabel(f'Predicted{msr}')
        ax[p[0]][p[1]].set_xlabel('Actual')

        ax[p[0]][p[1]].imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
        tick_marks = np.arange(2)
        ax[p[0]][p[1]].set_xticks(tick_marks)
        ax[p[0]][p[1]].set_yticks(tick_marks)

        for i in range(2):
            for j in range(2):
                ax[p[0]][p[1]].text(j - 0.2, i, f'{cm[i][j]*100:.2f}%' if percentage else f'{cm[i][j]}', fontsize=20)

    fig.suptitle(title)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def models_metric_bars(labels, metrics, title='Models best metric', color='#f9766e', save_file=None):
    plt.figure(figsize=(16, 10))

    x = range(len(labels))
    bars = plt.bar(x, metrics, color=color)
    plt.xticks(ticks=x, labels=labels, rotation=90)

    for idx, rect in enumerate(bars):
        plt.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), f'{metrics[idx]:.4f}',
                 ha='center', va='top', rotation=90, size=20)

    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim((0., 1.))

    if save_file:
        plt.savefig(save_file)
    plt.show()


def models_quality_plot(y_trues, y_preds_s, model_names, title='Models quality analysis.',
                        measure='f', colors=None, save_file=None):
    assert measure in ['f', 'F', 'p', 'P', 'r', 'R'],\
        'Invalid measure! Type in "f" (F measure), "p" (precision) or "r" (recall).'

    def adapt(y):
        y = np.array([[yy.T[i] for yy in y] for i in range(y.shape[2])])
        y_all = np.array([np.concatenate([yy[i] for yy in y]) for i in range(y.shape[1])])
        y = list([*y, y_all])

        return y

    labels = LABELS + ['overall']
    y_trues_s = np.array([y_trues for _ in range(len(model_names))])
    y_trues_s = adapt(y_trues_s)
    y_preds_s = adapt(y_preds_s)

    if not colors:
        colors = ['#f9766e', '#619dff']

    fig, ax = plt.subplots(2, 4, figsize=(16, 10))
    positions = list([tuple((i, j)) for i in range(2) for j in range(4)])

    for i, (p, label, y_true, y_pred) in enumerate(zip(positions, labels, y_trues_s, y_preds_s)):
        x = np.arange(len(model_names))
        measure_fn = f1_score if measure == 'f' else precision_score if measure == 'p' else recall_score
        y_0, y_1 = np.array([measure_fn(y_true=y_t, y_pred=y_p, labels=[0, 1], average=None, zero_division=1.) for y_t, y_p in zip(y_true, y_pred)]).T

        ax[p[0]][p[1]].set_title(f'"{label}" | Max {measure.upper()}1: {np.max(y_1):.4f}')
        ax[p[0]][p[1]].set_xticks(x)
        ax[p[0]][p[1]].set_xticklabels(model_names, rotation='90')

        w = 0.4
        ax[p[0]][p[1]].bar(x - w / 2, y_0, label=f'{measure.upper()}0', color=colors[0], width=w)
        ax[p[0]][p[1]].bar(x + w / 2, y_1, label=f'{measure.upper()}1', color=colors[1], width=w)

    h, ln = ax[0][0].get_legend_handles_labels()
    fig.legend(h, ln, loc='upper right')

    fig.text(0., 0.5, 'Score', fontsize=16, va='center', rotation='vertical')
    fig.text(0.5, 0., 'Model', fontsize=16, ha='center')

    fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def best_model_for_class(y_trues, y_preds_s, model_names,
                         title=f'Best F measures for each hate-speech class and overall.',
                         measure='f', colors=None, save_file=None):
    assert measure in ['f', 'F', 'p', 'P', 'r', 'R'], \
        'Invalid measure! Type in "f" (F measure), "p" (precision) or "r" (recall).'

    def adapt(y):
        y = np.array([[yy.T[i] for yy in y] for i in range(y.shape[2])])
        y_all = np.array([np.concatenate([yy[i] for yy in y]) for i in range(y.shape[1])])
        y = list([*y, y_all])

        return y

    labels = LABELS + ['overall']
    y_trues_s = np.array([y_trues for _ in range(len(model_names))])
    y_trues_s = adapt(y_trues_s)
    y_preds_s = adapt(y_preds_s)

    if not colors:
        colors = ['#f9766e', '#619dff']

    x = np.arange(len(labels))
    y_f0, y_f1 = list([]), list([])
    m_names = list([])
    for y_true, y_pred in zip(y_trues_s, y_preds_s):
        f0, f1 = np.array([f1_score(y_true=y_t, y_pred=y_p, labels=[0, 1], average=None, zero_division=1.) for y_t, y_p in zip(y_true, y_pred)]).T
        max_f1_id = np.argmax(f1)
        y_f0.append(f0[max_f1_id])
        y_f1.append(f1[max_f1_id])
        m_names.append(model_names[max_f1_id])

    plt.figure(figsize=(16, 10))

    w = 0.4
    bars0 = plt.bar(x - w / 2, y_f0, label=f'{measure.upper()}0', color=colors[0], width=w)
    bars1 = plt.bar(x + w / 2, y_f1, label=f'{measure.upper()}1', color=colors[1], width=w)
    plt.xticks(ticks=x, labels=labels)

    for idx, (rect0, rect1) in enumerate(zip(bars0, bars1)):
        plt.text(rect0.get_x() + rect0.get_width() / 2., rect0.get_height(), f'{y_f0[idx]:.4f}',
                 ha='center', va='top', size=20)
        plt.text(rect1.get_x() + rect1.get_width() / 2., rect1.get_height(), f'{y_f1[idx]:.4f}',
                 ha='center', va='bottom', size=20)
        plt.text(rect1.get_x() + rect1.get_width() / 2., 0.25 * rect1.get_height(), m_names[idx],
                 ha='center', va='bottom', rotation=90, size=16)

    plt.title(title)
    plt.xlabel('Hate-speech label')
    plt.ylabel('Score')
    plt.ylim((0., 1.))
    plt.legend(loc='best')

    if save_file:
        plt.savefig(save_file)
    plt.show()

