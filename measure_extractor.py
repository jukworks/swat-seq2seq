import csv
import pickle
from datetime import datetime

import conf

THRESHOLD = 0.3


def main():
    with open(conf.ANOMALY_PROBS_PICKLE, 'rb') as fh:
        anomaly_probs = pickle.load(fh)
        print(f'{len(anomaly_probs):,} rows loaded')

    detection = open('dat/detection', 'w')

    with open('dat/SWaT_Attack.csv') as fh:
        reader = csv.reader(fh, delimiter=',', quoting=csv.QUOTE_NONE)
        next(reader)  # remove header

        lines = 0
        for line in reader:
            d = str(datetime.strptime(line[0].strip(), conf.SWaT_TIME_FORMAT))
            measure = ['1' if p > THRESHOLD else '0' for p in anomaly_probs[d]] if d in anomaly_probs else ['0'] * 5
            measure = ['1' if '1' in measure else '0'] + measure
            print(','.join(measure), file=detection)
            lines += 1
    print(f'{lines:,} lines out')

    detection.close()


if __name__ == '__main__':
    main()
