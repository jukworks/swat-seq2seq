import argparse
import csv
import pickle
import sys
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

import conf

NORMAL = 'dat/SWaT_Normal.csv'


class Normalizer:
    def __init__(self):
        # Mean and standard dev are calculated from the normal dataset
        self.LR = {}
        df = pd.read_csv(NORMAL)
        for col in conf.ALL_SRCS:
            desc = df[col].describe()
            self.LR[col] = (desc['min'], desc['max'])
            print(f'{col:>10}: {self.LR[col][0]:10.6} - {self.LR[col][1]:10.6}')

    def normalize(self, dic: Dict, col: str) -> float:
        if col not in conf.ALL_SRCS:
            sys.exit('[FAILED] Invalid column name: {}'.format(col))
        cv = float(dic[col])
        if col in conf.DIGITAL_SRCS:
            return cv / 2  # 0, 0.5, 1
        L, R = self.LR[col]
        return (cv - L) / (R - L)


def attack_window_p(w):
    for att_p in w:
        if att_p:
            return True
    return False


def data_generator(filename: str):
    normalizer = Normalizer()
    with open(filename) as fh:
        reader = csv.reader(fh, delimiter=',', quoting=csv.QUOTE_NONE)
        header = list(map(lambda x: x.strip(), next(reader)))
        q_ts: deque = deque(maxlen=conf.WINDOW_SIZE)
        q_signals: List[deque] = [deque(maxlen=conf.WINDOW_SIZE) for _ in range(conf.N_PROCESS)]
        q_attack: deque = deque(maxlen=conf.WINDOW_SIZE)
        for line in reader:
            line = list(map(lambda x: x.strip(), line))
            dic = dict(zip(header, line))  # make a map from header's field names
            q_ts.append(datetime.strptime(dic[conf.HEADER_STRING_TIMESTAMP], conf.SWaT_TIME_FORMAT))
            for pidx in range(conf.N_PROCESS):
                q_signals[pidx].append(
                    np.array([normalizer.normalize(dic,
                                                   x) for x in conf.P_SRCS[pidx]],
                             dtype=np.float32)
                )
            q_attack.append(True if dic[conf.HEADER_STRING_NORMAL_OR_ATTACK].strip().upper() == 'ATTACK' else False)
            if len(q_ts) == conf.WINDOW_SIZE:
                if q_ts[0] + timedelta(seconds=conf.WINDOW_SIZE - 1) != q_ts[-1]:
                    continue
                signals_window = [np.array(q_signals[pidx]) for pidx in range(conf.N_PROCESS)]
                split_window = [
                    (w[:conf.WINDOW_GIVEN],
                     w[conf.WINDOW_GIVEN:conf.WINDOW_GIVEN + conf.WINDOW_PREDICT],
                     w[-1]) for w in signals_window
                ]
                yield q_ts[conf.WINDOW_GIVEN], split_window, attack_window_p(q_attack)


def main():
    parser = argparse.ArgumentParser(description='Preprocessor for the SWaT dataset')
    parser.add_argument('csv_file', help='data filename')
    parser.add_argument('out_file', help='preprocessed filename')
    args = parser.parse_args()

    print(f'* loading {args.csv_file}')
    print(f'* collecting a window size: {conf.WINDOW_SIZE}')
    print(f'\t- {conf.WINDOW_GIVEN} for training')
    print(f'\t- {conf.WINDOW_PREDICT} for prediction')
    print(f'\t- 1 for the answer')

    g = data_generator(args.csv_file)
    lines = 0
    lines_attack = 0
    picks = [[] for _ in range(conf.N_PROCESS)]
    while True:
        try:
            ts, window, is_attack = next(g)
            for pidx in range(conf.N_PROCESS):
                given, prediction, answer = window[pidx]
                picks[pidx].append([ts, given, prediction, answer, is_attack])
            lines += 1
            if is_attack:
                lines_attack += 1
        except StopIteration:
            break
    for pidx in range(conf.N_PROCESS):
        out_filename = '{}-P{}.dat'.format(args.out_file, pidx + 1)
        with open(out_filename, "wb") as fh:
            pickle.dump(picks[pidx], fh)
        print(f'* writing to {out_filename}')
    print(f'* {lines:,} data-points have been written ({lines_attack:,} attack windows)')


if __name__ == '__main__':
    main()
