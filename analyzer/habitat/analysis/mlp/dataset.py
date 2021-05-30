import numpy as np
import torch

from torch.utils.data import Dataset

from habitat.analysis.mlp.dataset_process import get_dataset


class HabitatDataset(Dataset):
    def __init__(self, dataset_path, features):
        self.x, self.y = get_dataset(dataset_path, features)

        # input normalization
        self.x = np.array(self.x)

        self.mu = np.mean(self.x, axis=0)
        self.sigma = np.std(self.x, axis=0)

        self.x = np.divide(np.subtract(self.x, self.mu), self.sigma)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.from_numpy(np.array(self.x[idx]).astype(np.float32), ), float(self.y[idx])
