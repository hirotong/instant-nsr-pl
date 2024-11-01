import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info
from torch.optim.optimizer import Optimizer
from torch_efficient_distloss import flatten_eff_distloss
import pandas as pd
import models
import systems
from models.ray_utils import get_rays
from models.utils import cleanup
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
from utils.eval import MeshEvaluator


@systems.register("rot-neus-system")
class RotNeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """

    def prepare(self):
        self.criterions = {"psnr": PSNR()}
        self.train_num_samples = self.config.model.train_num_rays * (
            self.config.model.num_samples_per_ray + self.config.model.get("num_samples_per_ray_bg", 0)
        )
        self.train_num_rays = self.config.model.train_num_rays

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, batch):
        return self.model(batch["rays"])

    def preprocess_data(self, batch, stage):
        if "index" in batch:  # validation / testing
            index = batch["index"]
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(self.train_num_rays,),
                    device=self.dataset.all_images.device,
                )
            else:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(1,),
                    device=self.dataset.all_images.device,
                )
        if stage in ["train"]:
            c2w = self.dataset.all_c2w[index]
            cc2w = self.dataset.all_cc2w[index]
            x = torch.randint(
                0,
                self.dataset.w,
                size=(self.train_num_rays,),
                device=self.dataset.all_images.device,
            )
            y = torch.randint(
                0,
                self.dataset.h,
                size=(self.train_num_rays,),
                device=self.dataset.all_images.device,
            )
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            crays_o, crays_d = get_rays(directions, cc2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            cc2w = self.dataset.all_cc2w[index][0]
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            crays_o, crays_d = get_rays(directions, cc2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
        crays = torch.cat([crays_o, F.normalize(crays_d, p=2, dim=-1)], dim=-1)

        # TODO: Disable this part.
        if stage in ["train"]:
            if self.config.model.background_color == "white":
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == "random":
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)

        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[..., None] + self.model.background_color * (1 - fg_mask[..., None])

        batch.update({"rays": torch.cat([rays, crays], dim=-1), "rgb": rgb, "fg_mask": fg_mask})

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out["num_samples_full"].sum().item()))
            self.train_num_rays = min(
                int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                self.config.model.max_train_num_rays,
            )

        loss_rgb_mse = F.mse_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )
        self.log("train/loss_rgb_mse", loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )
        self.log("train/loss_rgb", loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        loss_eikonal = ((torch.linalg.norm(out["sdf_grad_samples"], ord=2, dim=-1) - 1.0) ** 2).mean()
        self.log("train/loss_eikonal", loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        opacity = torch.clamp(out["opacity"].squeeze(-1), 1.0e-3, 1.0 - 1.0e-3)
        loss_mask = binary_cross_entropy(opacity, batch["fg_mask"].float())
        self.log("train/loss_mask", loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out["sdf_samples"].abs()).mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert (
                "sdf_laplace_samples" in out
            ), "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out["sdf_laplace_samples"].abs().mean()
            self.log("train/loss_curvature", loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out["weights"], out["points"], out["intervals"], out["ray_indices"])
            self.log("train/loss_distortion", loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(
                out["weights_bg"], out["points_bg"], out["intervals_bg"], out["ray_indices_bg"]
            )
            self.log("train/loss_distortion_bg", loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f"train/loss_{name}", value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        self.log("train/inv_s", out["inv_s"], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith("lambda"):
                self.log(f"train_params/{name}", self.C(value))

        self.log("train/num_rays", float(self.train_num_rays), prog_bar=True)

        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(name)

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """

    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """

    """
    https://github.com/Lightning-AI/pytorch-lightning/pull/16520
    """

    def training_step_end(self, training_step_output):
        training_step_output = self.trainer.strategy.reduce(training_step_output)
        self.training_step_outputs.append(training_step_output)
        return training_step_output

    def on_train_epoch_end(self):
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions["psnr"](out["comp_rgb_full"].to(batch["rgb"]), batch["rgb"])
        W, H = self.dataset.img_wh
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0].item()}.png",
            [
                {
                    "type": "rgb",
                    "img": batch["rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_rgb_full"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "grayscale",
                    "img": out["opacity"].view(H, W),
                    "kwargs": {"data_range": (0, 1), "cmap": None},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_bg"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "grayscale",
                        "img": out["opacity_bg"].view(H, W),
                        "kwargs": {"data_range": (0, 1), "cmap": None},
                    },
                ]
                if self.config.model.learned_background
                else []
            )
            + [
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "rgb",
                    "img": out["comp_normal"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                },
            ],
        )
        self.validation_step_outputs.append({"psnr": psnr, "index": batch["index"]})
        return {"psnr": psnr, "index": batch["index"]}

    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """

    def on_validation_epoch_end(self):
        out = self.validation_step_outputs
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out["index"].ndim == 1:
                    out_set[step_out["index"].item()] = {"psnr": step_out["psnr"]}
                # DDP
                else:
                    for oi, index in enumerate(step_out["index"]):
                        out_set[index[0].item()] = {"psnr": step_out["psnr"][oi]}
            psnr = torch.mean(torch.stack([o["psnr"] for o in out_set.values()]))
            self.log("val/psnr", psnr, prog_bar=True, rank_zero_only=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions["psnr"](out["comp_rgb_full"].to(batch["rgb"]), batch["rgb"])
        W, H = self.dataset.img_wh
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0].item()}.png",
            [
                {
                    "type": "rgb",
                    "img": batch["rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
                {
                    "type": "rgb",
                    "img": out["comp_rgb_full"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb_bg"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"].view(H, W, 3),
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if self.config.model.learned_background
                else []
            )
            + [
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "rgb",
                    "img": out["comp_normal"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                },
            ],
        )
        self.test_step_outputs.append({"psnr": psnr, "index": batch["index"]})
        return {"psnr": psnr, "index": batch["index"]}

    def on_test_epoch_end(self):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(self.test_step_outputs)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out["index"].ndim == 1:
                    out_set[step_out["index"].item()] = {"psnr": step_out["psnr"]}
                # DDP
                else:
                    for oi, index in enumerate(step_out["index"]):
                        out_set[index[0].item()] = {"psnr": step_out["psnr"][oi]}
            psnr = torch.mean(torch.stack([o["psnr"] for o in out_set.values()]))
            self.log("test/psnr", psnr, prog_bar=True, rank_zero_only=True)

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                "(\d+)\.png",
                save_format="mp4",
                fps=3,
            )

            mesh_pred = self.export()
            self.evaluate_mesh(mesh_pred)

    def evaluate_mesh(self, mesh_pred):
        mesh_gt_path = os.path.join(self.config.dataset.root_dir, "mesh_gt.obj")
        mesh_gt = trimesh.load(mesh_gt_path, force="mesh", process=False)

        num_vertices = mesh_gt.vertices.shape[0]
        n_points = self.config.evaluation.n_points
        sample_ratio = self.config.evaluation.sample_ratio
        # n_points = max(n_points, int(num_vertices * sample_ratio))

        pointcloud_gt, faces = mesh_gt.sample(n_points, return_index=True)
        pointcloud_gt = pointcloud_gt.astype(np.float32)
        normals_gt = mesh_gt.face_normals[faces]
        
        mesh_evaluator = MeshEvaluator(n_points=n_points)
        eval_mesh_dict = mesh_evaluator.eval_mesh(mesh_pred, pointcloud_gt, normals_gt, None, None)

        out_file = os.path.join(self.save_dir, "eval.pkl")
        out_file_csv = os.path.join(self.save_dir, "eval.csv")

        # Create pandas dataframe and save
        eval_df = pd.DataFrame(eval_mesh_dict, index=[0])
        eval_df.to_pickle(out_file)
        eval_df.to_csv(out_file_csv)
        
        print(eval_mesh_dict)

    def export(self):
        mesh = self.model.export(self.config.export)
        mesh_pred = self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh,
        )
        return mesh_pred
