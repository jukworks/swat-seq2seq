import pickle
from datetime import datetime, timedelta
from typing import Dict, List

import conf
from db import InfluxDB

DB = InfluxDB('swat')
FETCH_SIZE = 1_000


def dig() -> Dict:
    db_cursor = datetime(2015, 12, 28, 10, 1, 30)
    aggregate = {}
    while True:
        try:
            fetched = [
                DB.read(conf.JUDGE_MEASUREMENT.format(pidx + 1),
                        db_cursor,
                        seconds=FETCH_SIZE).json() for pidx in range(5)
            ]
            count = 0
            last_time_str = ''
            buffer: List[Dict] = [{} for _ in range(5)]
            for i in range(5):
                for statement in fetched[i]['results']:
                    for s in statement['series']:
                        for v in s['values']:
                            last_time_str = str(
                                datetime.strptime(v[0],
                                                  conf.INFLUX_RETURN_TIME_FORMAT) + timedelta(hours=9)
                            )
                            buffer[i][last_time_str] = v[-1]
                            count += 1
            if last_time_str != '':
                db_cursor = datetime.strptime(last_time_str, conf.DATETIME_BASIC_TIME_FORMAT) + timedelta(seconds=1)
        except Exception:
            break
        print(db_cursor)
        assert len(set([len(b) for b in buffer])) == 1  # the lengths of fetched row from all processes are the same
        dates = buffer[0].keys()
        for d in dates:
            aggregate[d] = [buffer[i][d] for i in range(5)]
    return aggregate


def main():
    anomaly_probs = dig()
    print(f'# of extracted timestamps: {len(anomaly_probs)}')
    with open(conf.ANOMALY_PROBS_PICKLE, 'wb') as fh:
        pickle.dump(anomaly_probs, fh, pickle.HIGHEST_PROTOCOL)
        print('SUCCESS: pickled anomaly probs')


if __name__ == '__main__':
    main()
