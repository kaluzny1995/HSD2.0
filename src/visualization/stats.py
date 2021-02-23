import numpy as np
from datetime import datetime
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..dataframes.timeline import find_empty_spaces


def tweet_yearly_counts_pie(df, show_legend=True, title='Tweet yearly counts.'):
    fig, ax = plt.subplots(figsize=(16, 10))

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

    plt.show()


def tweet_count_bars(df_m, df_wd, df_h, title='All tweets in figures.'):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))

    df_m.plot(kind='bar', ax=ax[0], color='#f8766d')
    df_wd.plot(kind='bar', ax=ax[1], color='#00bfc4')
    df_h.plot(kind='bar', ax=ax[2], color='#c77cff')

    for i, t in zip(range(3), ['mothly', 'weekdaily', 'hourly']):
        ax[i].get_legend().remove()
        ax[i].set_title(t)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def tweets_timeline(df, empty_spaces=None, threshold=100, empty_space_threshold=10,
                    title='Tweet amounts timeline analysis.'):
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
    plt.show()