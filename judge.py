import argparse
import configparser
from collections import deque
from datetime import datetime, timedelta

import numpy as np

import conf
from db import InfluxDB, datetime_to_nanosec

DB = InfluxDB('swat')
FETCH_SIZE = 1_000


class Smooth:
    def __init__(self, start: datetime, pidx: int, n_aggregate: int, n_cutoff: int):
        self.db_cursor = start
        self.pidx = pidx
        assert n_aggregate > n_cutoff * 2
        self.n_aggregate = n_aggregate
        self.n_cutoff = n_cutoff
        self.bucket: deque = deque(maxlen=n_aggregate)
        self.moving_avg: deque = deque(maxlen=n_aggregate)
        self.buffer: deque = deque(maxlen=FETCH_SIZE)
        while len(self.bucket) < n_aggregate:
            self.push()

    def fetch_to_buffer(self) -> int:
        r = DB.read(conf.EVAL_MEASUREMENT.format(self.pidx), self.db_cursor, seconds=FETCH_SIZE).json()
        count = 0
        last_time_str = ''
        for statement in r['results']:
            for s in statement['series']:
                cols = s['columns']
                for v in s['values']:
                    v_dict = dict(zip(cols, v))
                    self.buffer.append(v_dict)
                    last_time_str = v_dict['time']
                    count += 1
        if last_time_str != '':
            self.db_cursor = datetime.strptime(last_time_str,
                                               conf.INFLUX_RETURN_TIME_FORMAT) + timedelta(hours=9
                                                                                           ) + timedelta(seconds=1)
        print(f'\t- fetched {count:,} datapoints')
        return count

    def push(self):
        while True:
            try:
                self.bucket.append(self.buffer.popleft())
                break
            except IndexError:  # when buffer is empty
                self.fetch_to_buffer()

    def check(self, high: int, low: int, ratio: int):
        dist_window = np.array([v['distance'] for v in self.bucket], dtype=np.float32)
        dist_window = np.delete(dist_window, dist_window.argsort()[-self.n_cutoff:])
        self.moving_avg.append(np.sum(dist_window))
        sus = 0
        # write = True
        if len(self.moving_avg) == self.n_aggregate:
            avg_win = np.array(self.moving_avg, dtype=np.float32)
            sus = min(np.percentile(avg_win, high) / np.percentile(avg_win, low) / ratio, 1.0)
            # i_high = abs(avg_win - np.percentile(avg_win, high, interpolation='lower')).argmin()
            # i_low = abs(avg_win - np.percentile(avg_win, low, interpolation='lower')).argmin()
            # write = i_high > i_low
        return datetime.strptime(self.bucket[-1]['time'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=9), self.bucket[-1]['attack'], sus


def main():
    parser = argparse.ArgumentParser(description='Judge')
    parser.add_argument('--process', type=int, help='Process index (1-6)')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.ini')

    judge_conf = config['judge']
    n_aggregate = int(judge_conf['Aggregate'])
    n_cutoff = int(judge_conf['Cutoff'])

    high_percentile = int(judge_conf['HighPercentile'])
    low_percentile = int(judge_conf['LowPercentile'])
    alert_ratio = int(judge_conf['AlertRatio'])

    start = datetime(2015, 12, 28, 10, 1, 30)
    pidx = args.process
    assert 1 <= pidx <= 6

    DB.clear(conf.JUDGE_MEASUREMENT.format(pidx))

    judge = Smooth(start, pidx, n_aggregate, n_cutoff)
    ts_buffer = []
    att_buffer = []
    sus_buffer = []
    count = 0
    while True:
        current, att, suspicious = judge.check(high_percentile, low_percentile, alert_ratio)
        if current.second == 0 and current.minute == 0:
            print(f'* {current}')
        ts_buffer.append(current)
        att_buffer.append(att)
        sus_buffer.append(suspicious)
        count += 1
        if count >= FETCH_SIZE:
            DB.write(
                conf.JUDGE_MEASUREMENT.format(pidx),
                {'attack': att_buffer},
                {
                    'suspicious': sus_buffer,
                    'mark': [1.0] * count
                },
                [datetime_to_nanosec(ts) for ts in ts_buffer]
            )
            count = 0
            ts_buffer = []
            att_buffer = []
            sus_buffer = []
        judge.push()


if __name__ == '__main__':
    main()
