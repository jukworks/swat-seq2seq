import argparse
import configparser
import sys
from datetime import datetime

import torch

import conf
from db import InfluxDB, datetime_to_nanosec
from network import Network
from swat_dataset import SWaTDataset

assert torch.cuda.device_count() >= 6

BATCH_SIZE = 4096
DB = InfluxDB('swat')

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument(
    '--process',
    type=int,
    help='A list of process indexes (1-6): for example --processes 1 3',
    default=[],
    nargs='+'
)
parser.add_argument('--gpu', type=int, help='GPU index for learning (1-6)')
parser.add_argument('--save', type=int, help='The index of trained network')
args = parser.parse_args()

print(f'* target processes: {args.process}')
print(f'* use CUDA with GPU #{args.gpu}')
print(f'* N-programming index: #{args.save}')

config = configparser.ConfigParser()
config.read('config.ini')
training_conf = config['train']

for pidx in args.process:
    print(f'* starts to train process {pidx} with GPU {args.gpu} (parallel #{args.save})')

    dataset = SWaTDataset('dat/normal-P{}.dat'.format(pidx))

    p_features = len(conf.P_SRCS[pidx - 1])
    print(f'* # of features: {p_features}')
    pnet = Network(pidx=pidx, gidx=args.gpu, n_features=p_features, n_hiddens=conf.N_HIDDEN_CELLS)

    influx_measurement = conf.TRAIN_LOSS_MEASUREMENT.format(pidx)
    DB.clear(influx_measurement)

    training_start = datetime.now()
    epochs = int(training_conf['epochs'])
    print(f'* training is going to repeat {epochs:,} times (epochs)')
    pnet.train_mode()

    min_loss = sys.float_info.max
    for e in range(epochs):
        start = datetime.now()
        loss = pnet.train(dataset, BATCH_SIZE)
        DB.write(
            influx_measurement,
            {},
            {'train_loss_{}'.format(args.save): [loss]},
            [datetime_to_nanosec(datetime.now())]
        )
        saved = False
        if loss < min_loss:
            min_loss = loss
            pnet.save(args.save, min_loss)
            saved = True
        print(f'[{e+1:>4}] {loss:10.6} ({datetime.now() - start})' + (' -> saved' if saved else ''))
    print(f'* the total training time: {datetime.now() - training_start}')
