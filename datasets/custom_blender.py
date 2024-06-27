import os
import json
import math
import numpy as np
from PIL import Image

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class RotBlenderDatasetBase:
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), "r") as f:
            meta = json.load(f)

        if "w" in meta and "h" in meta:
            W, H = int(meta["w"]), int(meta["h"])
        else:
            W, H = 800, 800

        if "img_wh" in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif "img_downscale" in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, H
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane
        
        if "camera_angle_x" in meta:
            self.focal = 0.5 * w / math.tan(0.5 * meta["camera_angle_x"])  # scaled focal length
            self.focal_x = self.focal_y = self.focal
        else:
            self.focal_x = meta["focal_x"]
            self.focal_y = meta["focal_y"]
            if "img_downscale" in self.config:
                self.focal_x /= self.config.img_downscale
                self.focal_y /= self.config.img_downscale

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.w, self.h, self.focal_x, self.focal_y, self.w // 2, self.h // 2).to(
            self.rank
        )  # (h, w, 3)

        self.all_c2w, self.all_cc2w, self.all_images, self.all_fg_masks = [], [], [], []

        for i, frame in enumerate(meta["frames"]):
            obj_pose = np.array(frame["object"])
            cam_pose = np.array(frame["transform_matrix"])
            vcam_pose = np.linalg.inv(obj_pose) @ cam_pose

            self.all_c2w.append(torch.from_numpy(vcam_pose[:3, :4]))
            self.all_cc2w.append(torch.from_numpy(cam_pose[:3, :4]))

            img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            
            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[..., :3])
        
        self.all_c2w, self.all_cc2w, self.all_images, self.all_fg_masks = (
            torch.stack(self.all_c2w, dim=0).float().to(self.rank),
            torch.stack(self.all_cc2w, dim=0).float().to(self.rank),
            torch.stack(self.all_images, dim=0).float().to(self.rank),
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank),
        )


class RotBlenderDataset(Dataset, RotBlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class RotBlenderIterableDataset(IterableDataset, RotBlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}
            
@datasets.register("rot-blender")
class RotBlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = RotBlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RotBlenderDataset(self.config, self.config.val_split)
        if stage in [None, "test"]:
            self.test_dataset = RotBlenderDataset(self.config, self.config.test_split)
        if stage in [None, "predict"]:
            self.predict_dataset = RotBlenderDataset(self.config, self.config.train_split)
    
    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler,
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)
    
    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
