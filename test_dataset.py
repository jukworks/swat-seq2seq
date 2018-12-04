import torch
from torch.utils.data import DataLoader

import conf
from swat_dataset import SWaTDataset


def test_parsed_dataset():
    BATCH_SIZE = 5
    N_SAMPLES = 20
    for datatype in ['normal', 'attack']:
        for pidx in range(conf.N_PROCESS):
            dataset = SWaTDataset('dat/{}-P{}.dat'.format(datatype, pidx + 1))
            for idx, batch in enumerate(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)):
                assert type(batch['ts'][0]) == str

                given = batch['given']
                assert given.size() == torch.Size([BATCH_SIZE, conf.WINDOW_GIVEN, len(conf.P_SRCS[pidx])])
                if datatype == 'normal':
                    assert not (given.numpy() > 1.0).any()
                    assert not (given.numpy() < 0.0).any()

                assert batch['predict'].size() == torch.Size([BATCH_SIZE, conf.WINDOW_PREDICT, len(conf.P_SRCS[pidx])])
                assert batch['answer'].size() == torch.Size([BATCH_SIZE, len(conf.P_SRCS[pidx])])
                assert batch['attack'][0].item() in [0, 1]
                if idx >= N_SAMPLES:
                    break
