import json
import math
import os
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

import datasets
from datasets.utils import read_pickle
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class NeroSyntheticDatasetBase:
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True


        num_imgs = len(glob(os.path.join(self.config.root_dir, "*.pkl")))

        W, H = 800, 800
        self.w, self.h = W, H
        self.img_wh = (self.w, self.h)

        cam_0 = read_pickle(os.path.join(self.config.root_dir, f"0-camera.pkl"))
        K = cam_0[1]
        self.focal = (K[0, 0] + K[1, 1]) / 2

        self.near, self.far = self.config.near_plane, self.config.far_plane

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.w, self.h, self.focal, self.focal, self.w // 2, self.h // 2).to(
            self.rank
        )  # (h, w, 3)

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, frame in enumerate(range(num_imgs)):
            cam = read_pickle(os.path.join(self.config.root_dir, f"{i}-camera.pkl"))
            w2c = cam[0]
            R = np.transpose(w2c[:3, :3])
            C = -R @ w2c[:3, 3]
            c2w = np.concatenate([R, C[..., None]], -1)
            c2w[:3, 1:3] *= -1
            self.all_c2w.append(torch.from_numpy(c2w))

            img_path = os.path.join(self.config.root_dir, f"{i}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)

            depth_path = os.path.join(self.config.root_dir, f"{i}-depth.png")
            depth = Image.open(depth_path).convert("I;16")
            depth = depth.resize(self.img_wh, Image.BICUBIC)
            depth = np.array(depth).astype(np.float32) / 65535.0 * 15.0

            mask = (depth < 14.5).astype(np.float32)
            mask = torch.from_numpy(mask)

            self.all_fg_masks.append(mask)  # (h, w)
            self.all_images.append(img[..., :3])

        self.all_c2w, self.all_images, self.all_fg_masks = (
            torch.stack(self.all_c2w, dim=0).float().to(self.rank),
            torch.stack(self.all_images, dim=0).float().to(self.rank),
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank),
        )


class NeroSyntheticDataset(Dataset, NeroSyntheticDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class NeroSyntheticIterableDataset(IterableDataset, NeroSyntheticDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register("nerosync")
class NeroSyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = NeroSyntheticIterableDataset(self.config, self.config.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = NeroSyntheticDataset(self.config, self.config.val_split)
        if stage in [None, "test"]:
            self.test_dataset = NeroSyntheticDataset(self.config, self.config.test_split)
        if stage in [None, "predict"]:
            self.predict_dataset = NeroSyntheticDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(dataset, num_workers=os.cpu_count(), batch_size=batch_size, pin_memory=True, sampler=sampler)

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
