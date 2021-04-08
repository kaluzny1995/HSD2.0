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
        return f'{pct:.2f}% - ({absolute:d})'

    t_props = {'color': 'w', 'fontsize': 14, 'weight': 'bold'}
    l_props = {'size': 16}

    wedges, _, _ = ax.pie(data, autopct=lambda pct: annotate(pct, data), textprops=t_props)

    if show_legend:
        leg = ax.legend(wedges, labels, loc="upper left", fontsize=20)
        leg.set_title("Year", prop=l_props)

    ax.set_title(title)
    if save_file:
        plt.savefig(save_file)
    plt.show()


def tweet_count_bars(df_m, df_wd, df_h, title='All tweets in figures.', percentages=False, thr=0.14, save_file=None):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    df_m = df_m[df_m.columns].fillna(0.)
    df_wd = df_wd[df_wd.columns].fillna(0.)
    df_h = df_h[df_h.columns].fillna(0.)

    if percentages:
        df_m = df_m.assign(p=(df_m['count']/df_m['all']*100).values)
        df_wd = df_wd.assign(p=(df_wd['count']/df_wd['all']*100).values)
        df_h = df_h.assign(p=(df_h['count']/df_h['all']*100).values)
        df_m.drop(['count', 'all'], axis=1).plot(kind='bar', ax=ax[0], color='#f8766d')
        df_wd.drop(['count', 'all'], axis=1).plot(kind='bar', ax=ax[1], color='#00bfc4')
        df_h.drop(['count', 'all'], axis=1).plot(kind='bar', ax=ax[2], color='#c77cff')
    else:
        df_m.drop('all', axis=1).plot(kind='bar', ax=ax[0], color='#f8766d')
        df_wd.drop('all', axis=1).plot(kind='bar', ax=ax[1], color='#00bfc4')
        df_h.drop('all', axis=1).plot(kind='bar', ax=ax[2], color='#c77cff')

    p = ' [%]' if percentages else ''
    sizes = [11, 16, 8]
    for i, t, df, s in zip(range(3), [f'monthly{p}', f'weekdaily{p}', f'hourly{p}'], [df_m, df_wd, df_h], sizes):
        ax[i].get_legend().remove()
        ax[i].set_title(t)
        if percentages:
            ax[i].set_ylim([0., thr])
            for p, v in zip(ax[i].patches, df['count'].values):
                text = f'{v:.0f}'
                ax[i].text(p.get_x() + p.get_width() / 2, p.get_height(), text,
                           ha='center', va='bottom', fontsize=s)

    plt.suptitle(title)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def tweets_timeline(df, empty_spaces=None, threshold=100, empty_space_threshold=10,
                    title='Tweet amounts timeline analysis.', save_file=None):
    df = df.drop('all', axis=1)
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
            ax.axvspan(i, j, alpha=0.5, color='#e24a33')
            ax.text(i, threshold + 5, f'{datetime.strftime(i, "%d.%m.%Y")}', rotation='90')
            ax.text(j, threshold + 5, f'{datetime.strftime(j, "%d.%m.%Y")}', rotation='90', ha='right')

    plt.legend(loc='upper left')
    plt.suptitle(title)
    if save_file:
        plt.savefig(save_file)
    plt.show()


def popularity_hists(df, attribute='likes', title=None, color='#f9766e', save_file=None):
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    positions = list([tuple((i, j)) for i in range(2) for j in range(4)])
    labels = LABELS + ['others']
    
    df_pop = df.loc[:, [f'{attribute}_count'] + LABELS]
    df_pop.loc[:, LABELS] = df_pop.loc[:, LABELS].fillna(0).astype('int')

    data = list([])
    for label in labels:
        if label != 'others':
            d = df_pop.loc[:, [f'{attribute}_count', label]][df_pop[label] != 0][f'{attribute}_count'].values
        else:
            d = df_pop.loc[:, [f'{attribute}_count'] + LABELS][(df_pop[LABELS] == 0).all(axis=1)][f'{attribute}_count'].values
        data.append(d)

    for i, (p, label, d) in enumerate(zip(positions, labels, data)):
        d_norm = list([0, 1]) if not len(d) else d/np.linalg.norm(d)

        ax[p[0]][p[1]].set_title(f'"{label}" | max {attribute}: {1 if not len(d) else np.max(d)}')
        ax[p[0]][p[1]].hist(d_norm, weights=np.ones(len(d_norm))/len(d_norm), color=color, label='count')
        ax[p[0]][p[1]].set_xlim(0, 1)
        ax[p[0]][p[1]].set_yscale('log')
        ax[p[0]][p[1]].yaxis.set_major_formatter(PercentFormatter(1))
        ax[p[0]][p[1]].set_yticks([0.01, 0.1, 1.])

    h, ln = ax[0][0].get_legend_handles_labels()
    fig.legend(h, ln, loc='upper right')

    fig.text(0., 0.5, 'Count percentage', fontsize=16, va='center', rotation='vertical')
    fig.text(0.5, 0., f'{attribute.capitalize()} normalized count (max count in subtitles)', fontsize=16, ha='center')

    if not title:
        m = dict({'likes': 'like', 'replies': 'reply', 'retweets': 'retweet'})
        title = f'Tweet normalized {m[attribute]} counts analysis for each hate-speech type and others (non-hate tweets).'
    fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    if save_file:
        plt.savefig(save_file)
    plt.show()


def monthly_hateful_amount_lines(df, title='Hateful tweets monthly amount percentages.', threshold=4, save_file=None):
    fig, ax = plt.subplots(1, 1)

    dff = df[df.columns]
    for label in LABELS:
        dff[label] = dff[label] / dff['all'] * 100
    _, empty_spaces = find_empty_spaces(dff, threshold=1, attribute='all')
    dff = dff.fillna(.0)
    if empty_spaces:
        for es in empty_spaces:
            dff.iloc[es[0]:es[1]] = None
    dff.drop('all', axis=1).plot(figsize=(16, 10), ax=ax)

    # highlight empty spaces
    if empty_spaces:
        for es in empty_spaces:
            i = dff.index[es[0]]
            j = dff.index[es[1]]
            ax.axvspan(es[0] - 1, es[1], alpha=0.5, color='#e24a33')
            ax.text(es[0] - 0.5, threshold, i, rotation='90', size=18)
            ax.text(es[1] - 0.5, threshold, j, rotation='90', size=18, ha='right')

    plt.xlabel('Year-month')
    plt.ylabel('% amount')
    plt.title(title)

    if save_file:
        plt.savefig(save_file)
    plt.show()


def monthly_amount_line(df, title='Tweets monthly amounts (with lacks of data).', threshold=2800, save_file=None):
    fig, ax = plt.subplots(1, 1)

    dff = df[df.columns]
    _, empty_spaces = find_empty_spaces(dff, threshold=1, attribute='all')

    if empty_spaces:
        for es in empty_spaces:
            dff.iloc[es[0]:es[1]] = None
    dff.plot(figsize=(16, 10), ax=ax, legend=None, color='#619dff')

    # highlight empty spaces
    if empty_spaces:
        for es in empty_spaces:
            i = dff.index[es[0]]
            j = dff.index[es[1]]
            ax.axvspan(es[0] - 1, es[1], alpha=0.5, color='#e24a33')
            ax.text(es[0]-1, threshold, i, rotation='90', size=14)
            ax.text(es[1]+0.5, threshold, j, rotation='90', size=14, ha='right')

    plt.xlabel('Year-month')
    plt.ylabel('Amount')
    plt.title(title)

    if save_file:
        plt.savefig(save_file)
    plt.show()


def monthly_word_count_line(df, title='Year-monthly median tweet word counts.', threshold=15, save_file=None):
    fig, ax = plt.subplots(1, 1)

    dff = df[df.columns]
    _, empty_spaces = find_empty_spaces(dff, threshold=1, attribute='word count')

    if empty_spaces:
        for es in empty_spaces:
            dff.iloc[es[0]:es[1]] = None
    dff.plot(figsize=(16, 10), ax=ax, legend=None, color='#619dff')

    # highlight empty spaces
    if empty_spaces:
        for es in empty_spaces:
            i = dff.index[es[0]]
            j = dff.index[es[1]]
            ax.axvspan(es[0] - 1, es[1], alpha=0.5, color='#e24a33')
            ax.text(es[0] - 1, threshold, i, rotation='90', size=14)
            ax.text(es[1], threshold, j, rotation='90', size=14, ha='right')

    plt.xlabel('Year-month')
    plt.ylabel('Words count')
    plt.title(title)

    if save_file:
        plt.savefig(save_file)
    plt.show()
