import numpy as np

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from upsetplot import UpSet


def single_cardinalities_bar(df_sc, title='Single class cardinalities (%).'):
    ax = df_sc['cardinality'][1:].plot(kind='barh', figsize=(16, 8))
    for p, perc in zip(ax.patches, df_sc['%'][1:]):
        plt.text(p.get_width(), p.get_y()+0.25*p.get_height(), f'{p.get_width()} ({perc:1.2f}%)', fontsize=16)

    plt.title(title)
    plt.show()


def single_cardinalities_comp_bar(df_sc_comp, title='Comparison of single class cardinalities (%) before and after duplication of certain data.'):
    ax = df_sc_comp[1:][['card. before', 'card. after']].plot(kind='barh', stacked=True, figsize=(16, 8))
    for p, (cb, pb, ca, pa) in zip(ax.patches, df_sc_comp[1:].values):
        text0 = f'{int(cb)} ({pb:1.2f}%)'
        plt.text(cb, p.get_y()+0.25*p.get_height(), text0, fontsize=16)
        text1a = f'{int(ca)} ({pa:1.2f}%)'
        text1b = f'[{"+" if pa>pb else ""}{pa-pb:1.2f}% pts]'
        plt.text(ca+cb + (150 if ca-cb<100 else 0), p.get_y()+0.05*p.get_height(), text1a, fontsize=16)
        plt.text(ca+cb + (150 if ca-cb<100 else 0), p.get_y()+0.55*p.get_height(), text1b, fontsize=16)

    plt.title(title)
    plt.show()


def combination_cardinalities_upset(df_cc, title='Combination of classes and single classes cardinalities.', color='red'):
    upset = UpSet(df_cc['cardinality'][1:], sort_by='cardinality', facecolor=color, element_size=22)
    upset.plot()

    plt.suptitle(title)
    plt.show()


def phrase_cardinalities_bar(df_pc, title='Hateful phrases cardinalities (%).'):
    ax = df_pc.plot(kind='bar', figsize=(16, 8))
    ax.get_legend().remove()
    for p in ax.patches:
        perc = p.get_height()/df_pc['cardinality'].sum()
        plt.text(p.get_x() + p.get_width()/2, p.get_height()-50, f'{perc:1.4f}%\n({p.get_height()})',
                 ha='center', va='bottom', fontsize=16)

    plt.title(title)
    plt.show()


def phrase_cardinalities_comp_bar(df_pcc, title='Comparison of hateful phrases cardinalities (%) before and after extension.'):
    ax = df_pcc.plot(kind='bar', stacked=True, figsize=(16, 8))
    for p, (cb, ca) in zip(ax.patches, df_pcc.values):
        pb = cb/df_pcc['cardinality'].sum()
        pa = ca/df_pcc['ext. cardinality'].sum()
        text0 = f'{int(cb)} ({pb:1.4f}%)'
        plt.text(p.get_x() + p.get_width()/2, cb, text0, ha='center', va='bottom', fontsize=16)
        text1a = f'{int(ca)} ({pa:1.4f}%)'
        text1b = f'[{"+" if pa>pb else ""}{pa-pb:1.2f}% pts]'
        plt.text(p.get_x() + p.get_width()/2, ca+cb + (150 if ca-cb<50 else 0), text1a, ha='center', va='bottom', fontsize=16)
        plt.text(p.get_x() + p.get_width()/2, ca+cb + (150 if ca-cb<50 else 0)-80, text1b, ha='center', va='bottom', fontsize=16)

    plt.title(title)
    plt.show()


def vulgars_cardinalities_venn(vulgars, labels, title='Vulgar words cardinalities.'):
    plt.figure(figsize=(12, 10))
    venn3([set(v) for v in vulgars],
          set_colors=('#3E64AF', '#3EAF5D', '#D74E3B'),
          set_labels=tuple(labels),
          alpha=0.75)
    venn3_circles([set(v) for v in vulgars], lw=0.7)

    plt.title(title)
    plt.show()


def sentiment_counts_pie(data, labels, title='Empirically annotated sentiment cardinalities.'):
    fig, ax = plt.subplots(figsize=(16, 10))

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return f'{pct:.2f}%\n({absolute:d})'

    t_props = {'color': 'w', 'fontsize': 20, 'weight': 'bold'}
    l_props = {'size': 20}

    wedges, _, _ = ax.pie(data, autopct=lambda pct: func(pct, data), textprops=t_props)

    leg = ax.legend(wedges, labels, loc="upper right", fontsize=20)
    leg.set_title("Sentiment", prop=l_props)

    ax.set_title(title)

    plt.show()