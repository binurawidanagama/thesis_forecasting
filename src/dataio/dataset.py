from torch.utils.data import Dataset
import torch

class WindowDataset(Dataset):
    def __init__(self, X_dcenn, Y):
        self.X = X_dcenn
        self.Y = Y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

