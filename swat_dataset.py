import pickle

import torch
from torch.utils.data import Dataset

import conf


class SWaTDataset(Dataset):
    def __init__(self, pickle_jar: str):
        with open(pickle_jar, 'rb') as f:
            self.picks = pickle.load(f)

    def __len__(self):
        return len(self.picks)

    def __getitem__(self, idx):
        return {
            'ts': self.picks[idx][0].strftime(conf.SWaT_TIME_FORMAT),
            'given': torch.from_numpy(self.picks[idx][1]),
            'predict': torch.from_numpy(self.picks[idx][2]),
            'answer': torch.from_numpy(self.picks[idx][3]),
            'attack': torch.tensor(1 if self.picks[idx][4] else 0,
                                   dtype=torch.uint8)
        }
