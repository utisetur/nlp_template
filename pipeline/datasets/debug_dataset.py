import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class DebugDataset(Dataset):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        sequence_batch = torch.rand((4, 1000))
        targets = torch.randint(0, 2, (631, 1)).float()
        target_mask = torch.ones((631, 1), dtype=torch.bool)

        return sequence_batch, targets, target_mask

    def __len__(self) -> int:
        return int(500 * self.cfg.datamodule.batch_size)
