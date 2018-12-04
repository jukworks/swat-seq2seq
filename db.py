from datetime import datetime, timedelta
from typing import Dict, List

import requests

import conf


class InfluxDB:
    BUFFER_SIZE = 1500

    def __init__(self, db_name: str):
        self.url = 'http://localhost:8086'
        self.db_name = db_name

    def write(self, measurement: str, tags: Dict, fields: Dict, timestamp: List) -> None:
        assert measurement != ''
        assert fields and len(fields) > 0

        tag_and_fields = {**tags, **fields}
        try:
            assert len(set([len(x) for x in tag_and_fields.values()] + [len(timestamp)])) == 1
        except TypeError:
            print(f'[FAILED] FIELDS: {fields}')
            print(f'[FAILED] TIMESTAMPS: {timestamp}')
            raise

        msg = ''
        lines = 0
        for x in zip(*(tags.values()), *(fields.values()), timestamp):
            msg += measurement
            if tags:
                msg += ','
                msg += self.equal_pair(x, tags, 0, len(tags))
            msg += ' '
            msg += self.equal_pair(x, fields, len(tags), len(tags) + len(fields))
            if x[-1] is not None:
                msg += ' {}'.format(x[-1])
            msg += '\n'
            lines += 1
            if lines >= InfluxDB.BUFFER_SIZE:
                self.write_request(msg)
                lines = 0
                msg = ''
        if lines > 0:
            self.write_request(msg)

    def write_request(self, msg):
        r = requests.post('{}/write?db={}'.format(self.url, self.db_name), msg)
        try:
            r.raise_for_status()
        except (TypeError, AssertionError):
            print(f'[FAILED] message: {msg}')
            raise

    @staticmethod
    def equal_pair(x, keys, L, R):
        return ','.join(['{}={}'.format(k, v) for k, v in zip(keys.keys(), x[L:R])])

    def clear(self, measurement: str):
        msg = {'q': 'DROP SERIES FROM {}'.format(measurement)}
        r = requests.post('{}/query?db={}'.format(self.url, self.db_name), msg)
        r.raise_for_status()

    def read(self, measurement: str, start: datetime, seconds: int):
        query = {
            'q':
                'SELECT * FROM {} WHERE time >= {} and time < {}'.format(
                    measurement,
                    datetime_to_nanosec(start),
                    datetime_to_nanosec(start + timedelta(seconds=seconds))
                )
        }
        r = requests.post('{}/query?db={}'.format(self.url, self.db_name), query)
        r.raise_for_status()
        return r


def swat_time_to_nanosec(t: str) -> int:
    return int(datetime.strptime(t, conf.SWaT_TIME_FORMAT).timestamp() * 1_000_000_000)


def datetime_to_nanosec(t: datetime) -> int:
    return int(t.timestamp() * 1_000_000_000)
