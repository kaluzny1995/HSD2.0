import numpy as np
from datetime import datetime
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

from ..dataframes.timeline import find_empty_spaces
from ..constants import LABELS


def tweet_yearly_counts_pie(df, show_legend=True, title='Tweet yearly counts.', save_file=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    data = df['count'].values
    labels = df.index.values

    def annotate(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return f'{pct:.2f}%\n({absolute:d})'

    t_props = {'color': 'w', 'fontsize': 20, 'weight': 'bold'}
    l_props = {'size': 20}

    wedges, _, _ = ax.pie(data, autopct=lambda pct: annotate(pct, data), textprops=t_props)

    if show_legend:
        leg = ax.legend(wedges, labels, loc="upper right", fontsize=20)
        leg.set_title("Year", prop=l_props)

    ax.set_title(title)
    if save_file:
        plt.savefig(save_file)
    plt.show()


def tweet_count_bars(df_m, df_wd, df_h, title='All tweets in figures.', save_file=None):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    df_m.plot(kind='bar', ax=ax[0], color='#f8766d')
    df_wd.plot(kind='bar', ax=ax[1], color='#00bfc4')
    df_h.plot(kind='bar', ax=ax[2], color='#c77cff')

    for i, t in zip(range(3), ['mothly', 'weekdaily', 'hourly']):
        ax[i].get_legend().remove()
        ax[i].set_title(t)

    plt.suptitle(title)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def tweets_timeline(df, empty_spaces=None, threshold=100, empty_space_threshold=10,
                    title='Tweet amounts timeline analysis.', save_file=None):
    fig, ax = plt.subplots(figsize=(20, 8))

    # plot tweet amounts
    if empty_spaces:
        df, empty_spaces = find_empty_spaces(df, threshold=empty_space_threshold)
    df.plot(ax=ax, color=['#348abd', '#e24a33'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'))
    fig.autofmt_xdate()

    # plot threshold line
    ax.axhline(threshold, label=f'threshold: {threshold}', color='#c77cff', linestyle='--')

    # annotate peaks above threshold
    peaks_ids, _ = find_peaks(df['count'].values, height=threshold)
    for p_id in peaks_ids:
        x = df.index[p_id]
        y = df['count'].values[p_id]
        ax.text(x, y, f'{datetime.strftime(x, "%d.%m.%Y")} ({int(y)})', rotation='30')

    # highlight empty spaces
    if empty_spaces:
        for es in empty_spaces:
            i = df.index[es[0]]
            j = df.index[es[1]]
            y_pos = (df['count'].max() - df['count'].min()) // 2
            ax.axvspan(i, j, alpha=0.5, color='#e24a33')
            ax.text(i, threshold + 5, f'{datetime.strftime(i, "%d.%m.%Y")}', rotation='90')
            ax.text(j, threshold + 5, f'{datetime.strftime(j, "%d.%m.%Y")}', rotation='90', ha='right')

    plt.legend(loc='best')
    plt.suptitle(title)
    if save_file:
        plt.savefig(save_file)
    plt.show()


def popularity_hists(df, attribute='likes', title=None, color='#f9766e', save_file=None):
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    positions = list([tuple((i, j)) for i in range(2) for j in range(4)])
    labels = LABELS + ['others']
    
    df_pop = df.loc[:, [f'{attribute}_count'] + LABELS]
    df_pop.loc[:, LABELS] = df_pop.loc[:, LABELS].notnull().astype('int')

    data = list([])
    for label in labels:
        if label != 'others':
            d = df_pop.loc[:, [f'{attribute}_count', label]][df_pop[label] != 0][f'{attribute}_count'].values
        else:
            d = df_pop.loc[:, [f'{attribute}_count'] + LABELS][(df_pop[LABELS] == 0).all(axis=1)][f'{attribute}_count'].values
        data.append(d)

    for i, (p, label, d) in enumerate(zip(positions, labels, data)):
        d_norm = d/np.linalg.norm(d)

        ax[p[0]][p[1]].set_title(f'"{label}" | max {attribute}: {np.max(d)}')
        ax[p[0]][p[1]].hist(d_norm, weights=np.ones(len(d_norm))/len(d_norm), color=color, label='count')
        ax[p[0]][p[1]].set_xlim(0, 1)
        ax[p[0]][p[1]].set_yscale('log')
        ax[p[0]][p[1]].yaxis.set_major_formatter(PercentFormatter(1))
        ax[p[0]][p[1]].set_yticks([0.01, 0.1, 1.])

    h, ln = ax[0][0].get_legend_handles_labels()
    fig.legend(h, ln, loc='upper right')

    fig.text(0., 0.5, 'Count percentage', fontsize=16, va='center', rotation='vertical')
    fig.text(0.5, 0., f'{attribute.capitalize()} count', fontsize=16, ha='center')

    if not title:
        m = dict({'likes': 'like', 'replies': 'reply', 'retweets': 'retweet'})
        title = f'Tweet normalized {m[attribute]} counts analysis for each hate-speech type and others (non-hate tweets).'
    fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()

