#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math
import pickle
import warnings
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=r"The feature ([^\s]+) is currently marked under review")

import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics as tm
import torchvision.transforms.functional as TF
from termcolor import colored
from torchvision import models
from transformers import get_cosine_schedule_with_warmup

import wandb
from src.losses import compute_sharingan_loss
from src.metrics import AUC, Distance, GFTestAUC, GFTestDistance
from src.utils.common import build_2d_sincos_posemb, pair, spatial_argmax2d

TERM_COLOR = "cyan"


# ==================================================================================================================
#                                           SHARINGAN LIGHTNING MODULE                                             #
# ==================================================================================================================
class SharinganModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.model = Sharingan(
            patch_size=cfg.model.sharingan.patch_size,
            token_dim=cfg.model.sharingan.token_dim,
            image_size=cfg.model.sharingan.image_size,
            gaze_feature_dim=cfg.model.sharingan.gaze_feature_dim,
            encoder_depth=cfg.model.sharingan.encoder_depth,
            encoder_num_heads=cfg.model.sharingan.encoder_num_heads,
            encoder_num_global_tokens=cfg.model.sharingan.encoder_num_global_tokens,
            encoder_mlp_ratio=cfg.model.sharingan.encoder_mlp_ratio,
            encoder_use_qkv_bias=cfg.model.sharingan.encoder_use_qkv_bias,
            encoder_drop_rate=cfg.model.sharingan.encoder_drop_rate,
            encoder_attn_drop_rate=cfg.model.sharingan.encoder_attn_drop_rate,
            encoder_drop_path_rate=cfg.model.sharingan.encoder_drop_path_rate,
            decoder_feature_dim=cfg.model.sharingan.decoder_feature_dim,
            decoder_hooks=cfg.model.sharingan.decoder_hooks,
            decoder_hidden_dims=cfg.model.sharingan.decoder_hidden_dims,
            decoder_use_bn=cfg.model.sharingan.decoder_use_bn,
        )

        self.cfg = cfg
        self.dataset = cfg.experiment.dataset
        if self.dataset == "gazefollow":
            self.num_train_samples = cfg.data.gf.num_train_samples
        elif self.dataset == "videoattentiontarget":
            self.num_train_samples = math.ceil(cfg.data.vat.num_train_samples / cfg.data.stride)
        elif self.dataset == "childplay":
            self.num_train_samples = math.ceil(cfg.data.cp.num_train_samples / cfg.data.stride)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")
        self.num_steps_in_epoch = math.ceil(self.num_train_samples / cfg.train.batch_size)
        self.test_step_outputs = []

        # Model weights Paths
        self.model_weights = cfg.model.weights
        self.gaze360_weights = cfg.model.pretraining.gaze360
        self.multivit_weights = cfg.model.pretraining.multivit

        # Define Metrics
        self.metrics = nn.ModuleDict(
            {
                "val_dist": Distance(),
                "val_auc": AUC(),
                "val_ap": tm.AveragePrecision(task="binary"),
                "test_ap": tm.AveragePrecision(task="binary"),
                "test_auc": GFTestAUC() if self.dataset == "gazefollow" else AUC(),
                "test_dist": GFTestDistance() if self.dataset == "gazefollow" else Distance()
            }
        )

        # Define Loss Function
        self.compute_loss = partial(compute_sharingan_loss, 
                                    w_hm=cfg.loss.weight_heatmap, 
                                    w_ang=cfg.loss.weight_angular, 
                                    w_bce=cfg.loss.weight_bce)

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        if self.model_weights is not None:
            model_ckpt = torch.load(self.model_weights, map_location="cpu")
            model_weights = OrderedDict(
                [
                    (name.replace("model.", ""), value)
                    for name, value in model_ckpt["state_dict"].items()
                ]
            )
            self.model.load_state_dict(model_weights, strict=True)
            print(
                colored(
                    f"Successfully loaded pre-trained weights for the entire model from {self.model_weights}.",
                    TERM_COLOR,
                )
            )
            del model_ckpt
        else:
            # Load weights for Multi ViT
            multivit_ckpt = torch.load(self.multivit_weights, map_location="cpu")
            image_tokenizer_weights = OrderedDict(
                [
                    (name.replace("input_adapters.rgb.", ""), value)
                    for name, value in multivit_ckpt["model"].items()
                    if "input_adapters.rgb" in name
                ]
            )
            self.model.image_tokenizer.load_state_dict(
                image_tokenizer_weights, strict=True
            )
            print(
                colored(
                    f"Successfully loaded weights for the image tokenizer from {self.multivit_weights}.",
                    TERM_COLOR,
                )
            )

            encoder_weights = OrderedDict(
                [
                    (name.replace("encoder.", ""), value)
                    for name, value in multivit_ckpt["model"].items()
                    if "encoder" in name
                ]
            )
            self.model.encoder.encoder.load_state_dict(encoder_weights, strict=True)
            print(
                colored(
                    f"Successfully loaded weights for the ViT encoder from {self.multivit_weights}.",
                    TERM_COLOR,
                )
            )

            # Load Gaze Encoder Gaze360 Pre-trained Weights
            gaze360_ckpt = torch.load(self.gaze360_weights, map_location="cpu")
            gaze360_weights = OrderedDict(
                [
                    (name.replace("base_head.", ""), value)
                    for name, value in gaze360_ckpt["model_state_dict"].items()
                    if "base_head" in name
                ]
            )
            self.model.gaze_encoder.backbone.load_state_dict(
                gaze360_weights, strict=True
            )
            print(
                colored(
                    f"Successfully loaded weights for the gaze backbone from {self.gaze360_weights}.",
                    TERM_COLOR,
                )
            )

            # Delete checkpoints
            del (
                multivit_ckpt,
                image_tokenizer_weights,
                encoder_weights,
                gaze360_ckpt,
                gaze360_weights,
            )

        # Freeze weights
        self.freeze()

    def _set_batchnorm_eval(self, module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    def _set_dropout_eval(self, module):
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def freeze(self):
        # Freeze parameters based on sub-modules
        if self.cfg.train.freeze.gaze_encoder:
            print(colored(f"Freezing the Gaze Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.gaze_encoder)
        if self.cfg.train.freeze.image_tokenizer:
            print(colored(f"Freezing the Image Tokenizer layers.", TERM_COLOR))
            self.freeze_module(self.model.image_tokenizer)
        if self.cfg.train.freeze.vit_encoder:
            print(colored(f"Freezing the ViT Encoder layers.", TERM_COLOR))
            self.freeze_module(self.model.encoder)
        if self.cfg.train.freeze.gaze_decoder:
            print(colored(f"Freezing the Gaze Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.heatmap_decoder)
        if self.cfg.train.freeze.inout_decoder:
            print(colored(f"Freezing the InOut Decoder layers.", TERM_COLOR))
            self.freeze_module(self.model.inout_decoder)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        if self.dataset == "gazefollow":  # train all params
            # Optimizer
            optimizer = optim.AdamW(self.parameters(), 
                                    lr=self.cfg.optimizer.lr.base, 
                                    weight_decay=self.cfg.optimizer.weight_decay)
        else:  # train non frozen params with module-specific learning rates
            param_groups = [
                {
                    "params": self.model.gaze_encoder.parameters(),
                    "name": "gaze-encoder",
                    "lr": self.cfg.optimizer.lr.gaze_encoder,
                },
                {
                    "params": self.model.image_tokenizer.parameters(),
                    "name": "image-tokenizer",
                    "lr": self.cfg.optimizer.lr.image_tokenizer,
                },
                {
                    "params": self.model.encoder.parameters(),
                    "name": "vit-encoder",
                    "lr": self.cfg.optimizer.lr.vit_encoder,
                },
                {
                    "params": self.model.heatmap_decoder.parameters(),
                    "name": "gaze-decoder",
                    "lr": self.cfg.optimizer.lr.gaze_decoder,
                },
                {
                    "params": self.model.inout_decoder.parameters(),
                    "name": "inout-decoder",
                    "lr": self.cfg.optimizer.lr.inout_decoder,
                },
            ]
            optimizer = optim.AdamW(param_groups, 
                                    lr=self.cfg.optimizer.lr.base, 
                                    weight_decay=self.cfg.optimizer.weight_decay)

        # Scheduler: Cosine Annealing with Warmup or None
        if self.cfg.scheduler.type == "cosine_warmup":
            warmup_steps = self.cfg.scheduler.warmup_epochs * self.num_steps_in_epoch
            max_steps = self.cfg.train.epochs * self.num_steps_in_epoch
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)
            scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer

    def on_fit_start(self):
        # Define metrics
        if self.cfg.wandb.log:
            if self.dataset == "gazefollow":
                wandb.define_metric("metric/test/dist_to_avg", summary="min")
                wandb.define_metric("metric/test/avg_dist", summary="min")
                wandb.define_metric("metric/test/min_dist", summary="min")
            else:
                wandb.define_metric("metric/test/dist", summary="min")
                wandb.define_metric("metric/test/ap", summary="max")
            wandb.define_metric("metric/test/auc", summary="max")

            wandb.define_metric("loss/val", summary="min")
            wandb.define_metric("metric/val/dist", summary="min")
            wandb.define_metric("metric/val/auc", summary="max")
            wandb.define_metric("metric/val/ap", summary="max")

            wandb.define_metric("loss/train_epoch", summary="min")

    def on_train_epoch_start(self):
        # Set BN layers to eval mode for frozen modules
        if self.cfg.train.freeze.gaze_encoder:
            self.model.gaze_encoder.apply(self._set_batchnorm_eval)
            self.model.gaze_encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.image_tokenizer:
            self.model.image_tokenizer.apply(self._set_batchnorm_eval)
            self.model.image_tokenizer.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.vit_encoder:
            self.model.encoder.apply(self._set_batchnorm_eval)
            self.model.encoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.gaze_decoder:
            self.model.heatmap_decoder.apply(self._set_batchnorm_eval)
            self.model.heatmap_decoder.apply(self._set_dropout_eval)
        if self.cfg.train.freeze.inout_decoder:
            self.model.inout_decoder.apply(self._set_batchnorm_eval)
            self.model.inout_decoder.apply(self._set_dropout_eval)

    def training_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass
        gaze_vec_pred, gaze_heatmap_pred, inout_pred = self(batch)
        # Retrieve the target annotated person (last person)
        gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        inout_pred = inout_pred[:, -1, :].squeeze(1)  # (b, n, 1) >> (b, 1) >> (b,)
        gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...]  # (b, n, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Compute loss
        loss, logs = self.compute_loss(batch["gaze_vec"], batch["gaze_heatmap"], batch["inout"], 
                                       gaze_vec_pred, gaze_heatmap_pred, inout_pred)

        # Logging losses
        self.log("loss/train/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=True, on_epoch=True)
        self.log("loss/train", logs["total_loss"], batch_size=n, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass
        gaze_vec_pred, gaze_heatmap_pred, inout_pred = self(batch)
        # Retrieve the target annotated person (last person)
        gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        inout_pred = inout_pred[:, -1, :].squeeze(1)  # (b, n, 1) >> (b, 1) >> (b,)
        gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...]  # (b, n, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        # Compute loss
        loss, logs = self.compute_loss(batch["gaze_vec"], batch["gaze_heatmap"], batch["inout"], 
                                       gaze_vec_pred, gaze_heatmap_pred, inout_pred)

        # Update metrics
        self.metrics["val_auc"].update(gaze_heatmap_pred, batch["gaze_heatmap"], batch["inout"])
        self.metrics["val_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])
        self.metrics["val_ap"].update(inout_pred.sigmoid(), batch["inout"].long())

        # Logging losses
        self.log("loss/val/heatmap", logs["heatmap_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/angular", logs["angular_loss"], batch_size=ni, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val/bce", logs["bce_loss"], batch_size=n, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss/val", logs["total_loss"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Logging metrics
        self.log("metric/val/auc", self.metrics["val_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/dist", self.metrics["val_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
        self.log("metric/val/ap", self.metrics["val_ap"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        n = len(batch["image"])
        ni = int(batch["inout"].sum().item())

        # Forward pass (heatmap)
        gaze_vec_pred, gaze_heatmap_pred, inout_pred = self(batch)  # (b, n, 2), (b, n, h, w), (b, n, 1)
        # Retrieve the target annotated person (last person)
        gaze_vec_pred = gaze_vec_pred[:, -1, :]  # (b, n, 2) >> (b, 2)
        inout_pred = inout_pred[:, -1, :].squeeze(1)  # (b, n, 1) >> (b, 1) >> (b,)
        gaze_heatmap_pred = gaze_heatmap_pred[:, -1, ...]  # (b, n, 64, 64) >> (b, 64, 64)
        gaze_pt_pred = spatial_argmax2d(gaze_heatmap_pred, normalize=True)  # (b, 2)

        if self.dataset == "gazefollow":
            # Update metrics
            self.metrics["test_auc"].update(gaze_heatmap_pred, batch["gaze_pt"])
            test_dist_to_avg, test_avg_dist, test_min_dist = self.metrics["test_dist"](gaze_pt_pred, batch["gaze_pt"])

            # Log metrics
            self.log("metric/test/auc", self.metrics["test_auc"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist_to_avg", test_dist_to_avg, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/avg_dist", test_avg_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/min_dist", test_min_dist, batch_size=n, prog_bar=True, on_step=False, on_epoch=True)
        else:
            # Update metrics
            self.metrics["test_auc"].update(gaze_heatmap_pred, batch["gaze_heatmap"], batch["inout"])
            self.metrics["test_dist"].update(gaze_pt_pred, batch["gaze_pt"], batch["inout"])
            self.metrics["test_ap"].update(inout_pred.sigmoid(), batch["inout"].long())

            # Log metrics
            self.log("metric/test/auc", self.metrics["test_auc"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/dist", self.metrics["test_dist"], batch_size=ni, prog_bar=True, on_step=False, on_epoch=True)
            self.log("metric/test/ap", self.metrics["test_ap"], batch_size=n, prog_bar=True, on_step=False, on_epoch=True)

        # Build output dict
        step_output = {
            "gp_pred": gaze_pt_pred,
            "inout_pred": inout_pred,
            "gp_gt": batch["gaze_pt"],
            "inout_gt": batch["inout"],
            "path": batch["path"],
            "pid": batch["id"],
        }
        self.test_step_outputs.append(step_output)

    def on_test_epoch_end(self):
        # Save test predictions
        self._save_predictions(self.test_step_outputs, self.dataset)

    def _save_predictions(self, outputs, dataset):
        columns = ["gp_pred_x", "gp_pred_y", "inout_pred", "gp_gt_x", "gp_gt_y", "inout_gt", "pid", "path"]
        df_pred = pd.DataFrame(columns=columns)

        for output in outputs:
            batch_size = len(output["gp_pred"])
            for k in range(batch_size):
                if dataset == "gazefollow":
                    gp_gt = output["gp_gt"][k].cpu().numpy()
                    mask = gp_gt[:, 0] != -1.0
                    gp_gt_x, gp_gt_y = gp_gt[mask, 0], gp_gt[mask, 1]
                else:
                    gp_gt = output["gp_gt"][k].cpu().numpy()
                    gp_gt_x, gp_gt_y = gp_gt[0], gp_gt[1]

                gp_pred = output["gp_pred"][k].cpu().numpy()
                gp_pred_x, gp_pred_y = gp_pred[0], gp_pred[1]

                inout_pred = output["inout_pred"][k].cpu().numpy()
                inout_gt = output["inout_gt"][k].cpu().numpy()

                path = output["path"][k]
                pid = output["pid"][k].cpu().numpy().item()
                row = {
                    "gp_pred_x": gp_pred_x,
                    "gp_pred_y": gp_pred_y,
                    "inout_pred": inout_pred,
                    "gp_gt_x": gp_gt_x,
                    "gp_gt_y": gp_gt_y,
                    "inout_gt": inout_gt,
                    "pid": pid,
                    "path": path,
                }
                df_pred.loc[len(df_pred)] = row

        df_pred.to_csv(f"./test-predictions-{dataset}.csv", index=False)


# ==================================================================================================================== #
#                                                SHARINGAN ARCHITECTURE                                                #
# ==================================================================================================================== #
class Sharingan(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        token_dim: int = 768,
        image_size: int = 224,
        heatmap_size: int = 64,
        gaze_feature_dim: int = 512,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_num_global_tokens: int = 1,
        encoder_mlp_ratio: float = 4.0,
        encoder_use_qkv_bias: bool = True,
        encoder_drop_rate: float = 0.0,
        encoder_attn_drop_rate: float = 0.0,
        encoder_drop_path_rate: float = 0.0,
        decoder_feature_dim: int = 256,
        decoder_hooks: list = [2, 5, 8, 11],
        decoder_hidden_dims: list = [96, 192, 384, 768],
        decoder_use_bn: bool = False,
    ):

        super().__init__()

        self.patch_size = patch_size
        self.token_dim = token_dim
        self.image_size = pair(image_size)
        self.heatmap_size = heatmap_size
        self.gaze_feature_dim = gaze_feature_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_num_global_tokens = encoder_num_global_tokens
        self.encoder_mlp_ratio = encoder_mlp_ratio
        self.encoder_use_qkv_bias = encoder_use_qkv_bias
        self.encoder_drop_rate = encoder_drop_rate
        self.encoder_attn_drop_rate = encoder_attn_drop_rate
        self.encoder_drop_path_rate = encoder_drop_path_rate
        self.decoder_feature_dim = decoder_feature_dim
        self.decoder_hooks = decoder_hooks
        self.decoder_hidden_dims = decoder_hidden_dims
        self.decoder_use_bn = decoder_use_bn

        self.gaze_encoder = GazeEncoder(
            token_dim=token_dim, 
            feature_dim=gaze_feature_dim
        )

        self.image_tokenizer = SpatialInputTokenizer(
            num_channels=3,
            stride_level=1,
            patch_size=patch_size,
            token_dim=token_dim,
            use_sincos_pos_emb=True,
            is_learnable_pos_emb=False,
            image_size=image_size,
        )

        self.encoder = ViTEncoder(
            num_global_tokens=encoder_num_global_tokens,
            token_dim=token_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            use_qkv_bias=encoder_use_qkv_bias,
            drop_rate=encoder_drop_rate,
            attn_drop_rate=encoder_attn_drop_rate,
            drop_path_rate=encoder_drop_path_rate,
        )

        embed_size = image_size // patch_size
        self.heatmap_decoder = ConditionalDPTDecoder(
            token_dim=token_dim,
            feature_dim=decoder_feature_dim,
            patch_size=patch_size,
            hooks=decoder_hooks,
            hidden_dims=decoder_hidden_dims,
            use_bn=decoder_use_bn,
        )

        self.inout_decoder = LinearDecoder(
            input_dim=2 * token_dim,
            hidden_dims=[2 * 384, 2 * 192, 2 * 96, 2 * 48],
            output_dim=1,
            use_sigmoid=False,
        )

    def forward(self, x):
        # Expected x = {"image": image, "heads": heads, "head_bboxes": head_bboxes}
        b, c, h, w = x["image"].shape
        _, n, _, _, _ = x["heads"].shape  # n = total nb of people

        # Encode Gaze Tokens ===================================================
        gaze_tokens, gaze_vec = self.gaze_encoder(x["heads"], x["head_bboxes"])  # (b, n, d), (b, n, 2)

        # Tokenize Inputs ===================================================
        image_tokens = self.image_tokenizer(x["image"])  # (b, t, d) / t = num_tokens, d = token_dim
        input_tokens = torch.cat([image_tokens, gaze_tokens], dim=1)  # (b, t+n, d)
        t = image_tokens.shape[1]

        # Encode Tokens =====================================================
        output_tokens = self.encoder(input_tokens, return_all_layers=True)  # (b, t+n+g, d) / +g for global tokens / for DPT
        # Keep only gaze tokens (discard image tokens)
        output_gaze_tokens = output_tokens[-1][:, t : t + n, :].reshape(b * n, -1)

        # Decode Heatmap =====================================================
        hm_decoder_input = {"input": output_tokens, "image_size": (h, w), "num_people": n}
        gaze_hm = self.heatmap_decoder(hm_decoder_input)  # for Conditional DPT

        # Decode InOut ====================================================
        inout_decoder_input = torch.cat([output_gaze_tokens, gaze_tokens.view(b * n, -1)], dim=1)  # (b*n, 2*d)
        inout = self.inout_decoder(inout_decoder_input)  # (b*n, 1)
        inout = inout.view(b, n, -1)  # (b, n, 1)

        return gaze_vec, gaze_hm, inout


# ==================================================================================================================== #
#                                                   SHARINGAN BLOCKS                                                   #
# ==================================================================================================================== #


# ****************************************************** #
#                CONDITIONAL DPT DECODER                 #
# ****************************************************** #
class ConditionalDPTDecoder(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        hooks: List[int] = [2, 5, 8, 11],
        hidden_dims: List[int] = [96, 192, 384, 768],
        token_dim: int = 768,
        feature_dim: int = 128,
        use_bn: bool = True,
    ):
        super().__init__()

        self.patch_size = pair(patch_size)
        self.hooks = hooks
        self.token_dim = token_dim
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim
        self.use_bn = use_bn

        self.patch_h = self.patch_size[0]
        self.patch_w = self.patch_size[1]

        assert len(hooks) <= 4, f"The argument hooks can't have more than 4 elements."
        self.factors = [4, 8, 16, 32][-len(hooks) :]
        self.reassemble_blocks = nn.ModuleDict(
            {
                f"r{factor}": Reassemble(factor, hidden_dims[idx], feature_dim=feature_dim, token_dim=token_dim)
                for idx, factor in enumerate(self.factors)
            }
        )

        self.fusion_blocks = nn.ModuleDict(
            {
                f"f{factor}": FusionBlock(feature_dim, use_bn=use_bn)
                for idx, factor in enumerate(self.factors)
            }
        )

        self.gaze_projs = nn.ModuleDict(
            {
                f"g{factor}": nn.Linear(token_dim, feature_dim, bias=True)
                for idx, factor in enumerate(self.factors)
            }
        )

        self.head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        n = x["num_people"]
        img_h, img_w = x["image_size"]
        feat_h = img_h // self.patch_h
        feat_w = img_w // self.patch_w
        t = feat_h * feat_w  # num tokens

        # Retrieve intermediate encoder activations
        layers = x["input"]
        layers = [layers[hook] for hook in self.hooks]

        # Filter output tokens
        img_layers = [tokens[:, 0:t, :] for tokens in layers]  # [(b, t, d) ...]
        gaze_layers = [tokens[:, t : t + n, :] for tokens in layers]  # [(b, n, d) ...]
        b, _, _ = gaze_layers[-1].shape

        # Reshape tokens into spatial representation
        img_layers = [einops.rearrange(l, "b (fh fw) d -> b d fh fw", fh=feat_h, fw=feat_w) for l in img_layers]

        # Apply reassemble and fusion blocks
        for idx, (factor, img_layer, gaze_layer) in enumerate(
            zip(self.factors[::-1], img_layers[::-1], gaze_layers[::-1])
        ):
            f = self.reassemble_blocks[f"r{factor}"](img_layer)
            _, d, h, w = f.shape
            g = self.gaze_projs[f"g{factor}"](gaze_layer)  # (b, n, d) > # (b, n, d')
            f = torch.einsum("bdhw,bnd->bndhw", f, g).view(-1, self.feature_dim, h, w)  # (b, n, d', H/32, W/32) > (b*n, d', H/32, W/32)
            if idx == 0:
                z = self.fusion_blocks[f"f{factor}"](f)  # (b*n, d', H/f, W/f)
            else:
                z = self.fusion_blocks[f"f{factor}"](f, z)  # (b*n, d', H/f, W/f)

        # Apply prediction head and downscale (224 > 64)
        z = self.head(z)  # (b*n, d', H/2, W/2) > (b*n, 1, H/2, W/2)
        z = F.interpolate(z, size=(64, 64), mode="bilinear", align_corners=False)  # (b*n, 1, H, W) > (b*n, 1, 64, 64)
        z = z.view(b, n, 64, 64)  # (b*n, 1, 64, 64) > (b, n, 64, 64)
        return z


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

    def __repr__(self):
        return f"Interpolate(scale_factor={self.scale_factor}, mode={self.mode}, align_corners={self.align_corners})"


class Reassemble(nn.Module):
    def __init__(self, factor, hidden_dim, feature_dim=256, token_dim=768):
        super().__init__()

        assert factor in [4, 8, 16, 32], f"Argument `factor` not supported. Choose from [0.5, 4, 8, 16, 32]."
        self.factor = factor
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.token_dim = token_dim

        if factor == 4:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0, bias=True),
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif factor == 8:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2, padding=0, bias=True),
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif factor == 16:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        elif factor == 32:
            self.resample = nn.Sequential(
                nn.Conv2d(token_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=True),
            )
            self.proj = nn.Conv2d(hidden_dim, feature_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.resample(x)
        x = self.proj(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, feature_dim, use_bn=False):
        """Init.
        Args:
            features (int): dimension of feature maps
            use_bn (bool): whether to use batch normalization in the Residual Conv Units.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.use_bn = use_bn

        modules = nn.ModuleList(
            [
                nn.ReLU(inplace=False),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=(not self.use_bn)),
                nn.ReLU(inplace=False),
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=(not self.use_bn)),
            ]
        )
        if self.use_bn:
            modules.insert(2, nn.BatchNorm2d(feature_dim))
            modules.insert(5, nn.BatchNorm2d(feature_dim))
        self.residual_module = nn.Sequential(*modules)

    def forward(self, x):
        z = self.residual_module(x)
        return z + x


class FusionBlock(nn.Module):
    def __init__(self, feature_dim, use_bn=False):
        super().__init__()

        self.feature_dim = feature_dim
        self.use_bn = use_bn

        self.rcu1 = ResidualConvUnit(feature_dim, use_bn=use_bn)
        self.rcu2 = ResidualConvUnit(feature_dim, use_bn=use_bn)
        self.resample = Interpolate(2, "bilinear", align_corners=True)
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, *xs):
        assert (1 <= len(xs) <= 2), f"Can only accept inputs of length <= 2. Received len(xs)={len(xs)}"

        z = self.rcu1(xs[0])
        if len(xs) == 2:
            z = z + xs[1]
        z = self.rcu2(z)
        z = self.resample(z)
        z = self.proj(z)

        return z


# ****************************************************** #
#                     LINEAR DECODER                     #
# ****************************************************** #
class ResidualLinearBlock(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384, output_dim=192):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.res_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = torch.relu(self.bn1(self.fc1(x)))
        o = torch.relu(self.bn2(self.fc2(z)) + self.res_fc(x))
        return o


class LinearDecoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dims=[384, 192, 96, 48],
        output_dim=2,
        use_sigmoid=True,
    ):
        super().__init__()

        self.use_sigmoid = use_sigmoid
        self.block1 = ResidualLinearBlock(input_dim=input_dim, hidden_dim=hidden_dims[0], output_dim=hidden_dims[1])
        self.block2 = ResidualLinearBlock(input_dim=hidden_dims[1], hidden_dim=hidden_dims[2], output_dim=hidden_dims[3])
        self.fc = nn.Linear(hidden_dims[3], output_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.fc(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


# ****************************************************** #
#                      GAZE ENCODER                      #
# ****************************************************** #
class GazeEncoder(nn.Module):
    def __init__(self, token_dim=768, feature_dim=512):
        super().__init__()

        self.feature_dim = feature_dim
        self.token_dim = token_dim

        base = models.resnet18(weights=None)  # type: ignore
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.backbone(dummy_head)
        embed_dim = dummy_head.size(1)

        self.gaze_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.pos_proj = nn.Linear(4, token_dim)

        self.gaze_predictor = nn.Sequential(
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),  # 2 = number of outputs (x, y) unit vector
            nn.Tanh(),
        )

        # Initialize weights
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, head, head_bbox):
        b, n, p = head_bbox.shape
        b, n, c, h, w = head.shape

        head_bbox_emb = self.pos_proj(head_bbox.view(-1, p))  # (b*n, token_dim)
        gaze_emb = self.backbone(head.view(-1, c, h, w)).flatten(1, -1) # (b*n, embed_dim)

        gaze_token = self.gaze_proj(gaze_emb) + head_bbox_emb  # (b*n, token_dim)
        gaze_token = gaze_token.view(b, n, -1) # (b, n, token_dim)

        gaze_vec = self.gaze_predictor(gaze_emb)  # (b*n, 2)
        gaze_vec = F.normalize(gaze_vec, p=2, dim=1)  # normalize to unit vector
        gaze_vec = gaze_vec.view(b, n, -1)  # (b, n, 2)

        return gaze_token, gaze_vec


# ****************************************************** #
#                      VIT ENCODER                       #
# ****************************************************** #
class ViTEncoder(nn.Module):
    def __init__(
        self,
        num_global_tokens: int = 1,
        token_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        # Add global tokens
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, token_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        # Add encoder layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.Sequential(
            *[
                TransformerBlock(
                    dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qkv_bias=use_qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                )
                for i in range(depth)
            ]
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Initialize the weights of Q, K, V separately
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and ("qkv" in name):
                val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                nn.init.uniform_(m.weight, -val, val)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def __len__(self):
        return len(self.encoder)

    def forward(self, input_tokens: torch.Tensor, return_all_layers: bool = True):

        # Add global tokens to input tokens
        global_tokens = einops.repeat(self.global_tokens, "() n d -> b n d", b=len(input_tokens))
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        # Pass tokens through Transformer
        if not return_all_layers:
            encoder_tokens = self.encoder(input_tokens)
        else:
            # Optionally access every intermediate layer
            encoder_tokens = []
            tokens = input_tokens
            for block in self.encoder:
                tokens = block(tokens)
                encoder_tokens.append(tokens)

        return encoder_tokens


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        use_qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, use_qkv_bias=use_qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate)
        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity())
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop_rate=drop_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_rate=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        use_qkv_bias=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=use_qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# ****************************************************** #
#                SPATIAL INPUT TOKENIZER                 #
# ****************************************************** #
class SpatialInputTokenizer(nn.Module):
    """Tokenizer for spatial inputs, like images or heatmaps.
    Creates tokens from patches over the input image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image (e.g. 4 for 1/4th the size of the image).
    :param patch_size: Int or tuple of the patch size over the full image size. Patch size for smaller inputs will be computed accordingly.
    :param token_dim: Dimension of output tokens.
    :param use_sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings.
    :param is_learnable_pos_emb: Set to True to learn positional embeddings instead.
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size: Union[int, Tuple[int, int]],
        token_dim: int = 768,
        use_sincos_pos_emb: bool = True,
        is_learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224,
    ):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.token_dim = token_dim
        self.use_sincos_pos_emb = use_sincos_pos_emb
        self.is_learnable_pos_emb = is_learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self._init_pos_emb()
        self.proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.token_dim,
            kernel_size=(self.P_H, self.P_W),
            stride=(self.P_H, self.P_W),
        )

    def _init_pos_emb(self):
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_pos_emb = self.image_size[0] // (self.stride_level * self.P_H)
        w_pos_emb = self.image_size[1] // (self.stride_level * self.P_W)

        if self.use_sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_pos_emb, w=w_pos_emb, embed_dim=self.token_dim)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.is_learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.token_dim, h_pos_emb, w_pos_emb))
            nn.init.trunc_normal_(self.pos_emb, mean=0.0, std=0.02, a=-2.0, b=2.0)

    def forward(self, x):
        # input.shape = BxCxHxW >> output.shape = BxNxD (where N=n_tokens, D=token_dim)

        B, C, H, W = x.shape

        assert (H % self.P_H == 0) and (W % self.P_W == 0), f"Image size {H}x{W} must be divisible by patch size {self.P_H}x{self.P_W}"
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Create tokens [B, C, PH, PW] >> [B, D, H, W] >> [B, (H*W), D]
        x_tokens = einops.rearrange(self.proj(x), "b d h w -> b (h w) d")

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False)
        x_pos_emb = einops.rearrange(x_pos_emb, "b d h w -> b (h w) d")

        # Add patches and positional embeddings
        x = x_tokens + x_pos_emb

        return x
