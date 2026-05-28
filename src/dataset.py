import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, manifest_path, split, remove):
        self._manifest_path = manifest_path
        self._manifest = pd.read_csv(manifest_path)
        self._manifest = self._manifest[self._manifest["split"] == split].reset_index(
            drop=True
        )
        self._remove = remove

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        item = self._manifest.iloc[idx]
        array = torch.from_numpy(np.load(item["file"])).float()
        if "onset" in self._remove:
            array[:, 0] = 0.0
        if "duration" in self._remove:
            array[:, 1] = 0.0
        if "velocity" in self._remove:
            array[:, 2] = 0.0
        if "ioi" in self._remove:
            array[:, 3] = 0.0
        label = torch.tensor(item["label"]).long()

        return array, label


def get_dataloaders(manifest_path, batch_size=32, num_workers=0, remove=[]):
    train_ds = Dataset(manifest_path, split="train", remove=remove)
    val_ds = Dataset(manifest_path, split="val", remove=remove)
    test_ds = Dataset(manifest_path, split="test", remove=remove)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_dataloaders("Data/processed/manifest.csv")
batch = next(iter(train_loader))
print(batch[0].shape, batch[1].shape)  # expect (32, 512, 4) and (32,)
