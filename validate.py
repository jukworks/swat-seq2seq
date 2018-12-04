import argparse
import sys
from datetime import datetime

import torch
from torch.utils.data import ConcatDataset

import conf
from db import InfluxDB, datetime_to_nanosec
from network import Network
from swat_dataset import SWaTDataset

N_DUPLICATE_RUNS = 2

BATCH_SIZE = 1024  # larger validation batch is possible, but DB is the bottleneck
DB = InfluxDB('swat')

parser = argparse.ArgumentParser(description='validator')
parser.add_argument('--process', type=int, help='Process index (1-6)')
args = parser.parse_args()

pidx = args.process
assert 1 <= pidx <= 6
DB.clear(conf.EVAL_MEASUREMENT.format(pidx))

p_features = len(conf.P_SRCS[pidx - 1])
pnet = Network(pidx=pidx, gidx=pidx, n_features=p_features, n_hiddens=conf.N_HIDDEN_CELLS)

current_min_loss = sys.float_info.max
min_idx = 1
for i in range(N_DUPLICATE_RUNS):
    min_loss = pnet.load(i + 1)
    print(f'* loaded saved network #{i+1}, min_loss: {min_loss}')
    if min_loss < current_min_loss:
        min_idx = i + 1
        current_min_loss = min_loss
print(f'* the minimum loss index: {min_idx}')
pnet.load(min_idx)

dataset = ConcatDataset(
    [SWaTDataset('dat/normal-P{}.dat'.format(pidx)),
     SWaTDataset('dat/attack-P{}.dat'.format(pidx))]
)

pnet.eval_mode()
start = datetime.now()
with torch.no_grad():
    loss = pnet.eval(dataset, BATCH_SIZE, db_write=True)
DB.write(conf.TRAIN_LOSS_MEASUREMENT.format(pidx), {}, {'val_loss': [loss]}, [datetime_to_nanosec(datetime.now())])
print(f'* val loss: {loss} ({datetime.now() - start})')
