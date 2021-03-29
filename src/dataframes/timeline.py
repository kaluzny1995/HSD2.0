import numpy as np
import pandas as pd
from datetime import datetime

from ..utils.dates import daterange
from ..utils.ops import closing_empty_spaces
from ..constants import LABELS


def get_stats(df, hate_type=None, month_names=None, weekday_names=None):
    all_years = np.unique(list([d.split('-')[0] for d in df['date']]))
    all_months = range(1, 13)
    all_weekdays = range(7)
    all_hours = range(24)
    all_dates = list(daterange(df['date'].iloc[0], df['date'].iloc[-1]))

    if hate_type:
        df = df[df[hate_type] == 1.0][['date', 'time']]

    # yearly counts dataframe
    df_yc = pd.DataFrame({'year': all_years})
    y = list([d.split('-')[0] for d in df['date']])
    y, cnt = np.unique(y, return_counts=True)
    df_yc = df_yc.merge(pd.DataFrame({
        'year': y,
        'count': cnt
    }), left_on='year', right_on='year', how='left')
    df_yc = df_yc.set_index('year')

    # monthly counts
    df_mc = pd.DataFrame({'month': all_months})
    m = np.array([int(d.split('-')[1]) for d in df['date']])
    m, cnt = np.unique(m, return_counts=True)
    df_mc = df_mc.merge(pd.DataFrame({
        'month': m,
        'count': cnt
    }), left_on='month', right_on='month', how='left')
    if month_names:
        df_mc['month'] = df_mc['month'].map(dict(zip(range(1, 13), month_names)))
    df_mc = df_mc.set_index('month')

    # weekdaily counts
    df_wdc = pd.DataFrame({'weekday': all_weekdays})
    wd = np.array([datetime.strptime(d, '%Y-%m-%d').weekday() for d in df['date']])
    wd, cnt = np.unique(wd, return_counts=True)
    df_wdc = df_wdc.merge(pd.DataFrame({
        'weekday': wd,
        'count': cnt
    }), left_on='weekday', right_on='weekday', how='left')
    if weekday_names:
        df_wdc['weekday'] = df_wdc['weekday'].map(dict(zip(range(7), weekday_names)))
    df_wdc = df_wdc.set_index('weekday')

    # hourly counts
    df_hc = pd.DataFrame({'hour': all_hours})
    h = np.array([int(t.split(':')[0]) for t in df['time']])
    h, cnt = np.unique(h, return_counts=True)
    df_hc = df_hc.merge(pd.DataFrame({
        'hour': h,
        'count': cnt
    }), left_on='hour', right_on='hour', how='left')
    df_hc['hour'] = list([f'             {h}' for h in df_hc['hour'].values])
    df_hc = df_hc.set_index('hour')

    # daily (by date) counts
    df_dc = pd.DataFrame({'date': all_dates})
    d, cnt = np.unique(df['date'].values, return_counts=True)
    df_dc = df_dc.merge(pd.DataFrame({
        'date': list([datetime.strptime(dt, '%Y-%m-%d') for dt in d]),
        'count': cnt
    }), left_on='date', right_on='date', how='left')
    df_dc = df_dc.set_index('date')

    return df_yc, df_mc, df_wdc, df_hc, df_dc


# highlight empty spaces in timeline dataframe
# (p - positive element | n - negative element | threshold - min. empty space length)
def find_empty_spaces(df_timeline, p=.0, n=None, threshold=10, attribute='count'):
    df = pd.DataFrame(df_timeline.values, columns=df_timeline.columns, index=df_timeline.index)

    # find all positions with empty values and highlight them as .0 oppositely None
    empty_space = np.where(pd.isna(df[attribute]), p, n)
    # reduce small empty spaces (with length under threshold)
    empty_space, empty_space_bounds = closing_empty_spaces(empty_space, p=.0, n=None,
                                                           threshold=threshold)
    df['empty space'] = empty_space

    return df, empty_space_bounds


def get_monthly_stats(df, labeled=True):
    all_dates = list(daterange(df['date'].iloc[0], df['date'].iloc[-1]))
    all_year_months = np.unique([f'{d.year}-{"0" + str(d.month) if d.month < 10 else d.month}' for d in all_dates])

    stats = list([])
    labels = ['all'] + LABELS if labeled else ['all']

    for label in labels:
        df_label = df[df.columns]
        if label != 'all':
            df_label = df[df[label] == 1.0][['date']]

        df_ymc = pd.DataFrame({'year-month': all_year_months})
        ym = list(['-'.join(d.split('-')[:-1]) for d in df_label['date']])
        ym, cnt = np.unique(ym, return_counts=True)
        df_ymc = df_ymc.merge(pd.DataFrame({
            'year-month': ym,
            'count': cnt
        }), left_on='year-month', right_on='year-month', how='left')
        stats.append(df_ymc['count'])

    df_stats = pd.DataFrame(np.array(stats).T, columns=labels, index=all_year_months)

    return df_stats


def empty_date_intervals(df, threshold=10):
    dff = get_stats(df)[-1]
    dff, empty_ids = find_empty_spaces(dff, threshold=threshold)

    return list([f'{dff.iloc[es[0]].name.strftime("%Y-%m-%d")} - ' +
                 f'{dff.iloc[es[1]].name.strftime("%Y-%m-%d")}' for es in empty_ids])
