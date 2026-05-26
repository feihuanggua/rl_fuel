"""PyTorch Dataset + 4倍旋转数据增强."""
import torch
from torch.utils.data import Dataset, random_split


class ExpertDataset(Dataset):
    """专家数据集，支持 4 倍旋转增强."""

    def __init__(self, data_path: str, augment: bool = True):
        data = torch.load(data_path, weights_only=False)
        self.inputs = data["inputs"].float()   # [N, 3, V, V, V]
        self.targets = data["targets"]          # [N, 4]
        self.augment = augment

    def __len__(self):
        return len(self.inputs) * (4 if self.augment else 1)

    def __getitem__(self, idx):
        if self.augment:
            base_idx = idx // 4
            rot = idx % 4
        else:
            base_idx = idx
            rot = 0

        grid = self.inputs[base_idx].clone()
        target = self.targets[base_idx].clone()

        if rot > 0:
            grid = torch.rot90(grid, k=rot, dims=[1, 2])
            target = self._rotate_target(target, rot)

        return grid, target

    @staticmethod
    def _rotate_target(target, rot):
        """旋转标签: (dx,dy,dz,dyaw) 对应 XY 平面旋转."""
        dx, dy, dz, dyaw = target
        for _ in range(rot):
            dx, dy = -dy, dx
            dyaw = (dyaw + 0.5) % 2.0
            if dyaw > 1.0:
                dyaw -= 2.0
        return torch.tensor([dx, dy, dz, dyaw])


def make_dataloaders(data_path, batch_size=128, val_split=0.1, augment=True):
    """创建训练和验证 DataLoader."""
    dataset = ExpertDataset(data_path, augment=augment)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len

    train_set, val_set = random_split(
        dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    print(f"Dataset: {train_len} train, {val_len} val (augment={augment})")
    return train_loader, val_loader
