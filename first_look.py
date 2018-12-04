import pandas as pd

import conf

IGNORE = ['Timestamp', 'Normal/Attack']

for filename in ['dat/SWaT_Normal.csv', 'dat/SWaT_Attack.csv']:
    print(f'Target file: {filename}')
    df = pd.read_csv(filename)
    df = df.rename(columns=lambda x: x.strip())
    for col in conf.ALL_SRCS:
        desc = df[col].describe()
        print(f'Source {col:>7}: [{desc["min"]:10.6} - {desc["max"]:10.6}], ({desc["mean"]:10.6}, {desc["std"]:10.6})')
