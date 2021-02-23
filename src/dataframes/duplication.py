import pandas as pd
import random

from ..constants import LABELS


def duplicate_under_threshold(df, df_cc, threshold=5):
    df[LABELS] = df[LABELS].fillna(.0).astype('int')
    combinations = df_cc[df_cc['cardinality'] < threshold].index

    df_dupl = pd.DataFrame(df)
    for combination in combinations:

        # reduce dataframe to only relevant examples for a combination of classes (_labels)
        df_relev = pd.DataFrame(df)
        for label, c in zip(LABELS, combination):
            df_relev = df_relev[df_relev[label] == c]

        # random order of relevant examples (for duplication)
        rand_pos = [0 if len(df_relev) <= 1 else random.randint(0, len(df_relev) - 1)
                    for i in range(threshold - len(df_relev))]

        for rp in rand_pos:
            row = df_relev.iloc[rp]
            df_dupl = df_dupl.append(row)

    for label in LABELS:
        df_dupl[label] = df_dupl[label].astype('int')

    return df_dupl
