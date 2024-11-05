#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import torch
import torchvision.transforms.functional as TF
import numpy as np

import torchmetrics as tm
from torchmetrics.functional.classification.auroc import binary_auroc
from torchmetrics.functional.classification import binary_precision


from src.utils.common import generate_binary_gaze_heatmap, spatial_argmax2d, is_point_in_box


class PLAH(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes Precision of Looking At Heads (PLAH) for general datasets.
        """
        super().__init__()
        self.add_state("sum_plah", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_pos_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_pt_pred: torch.Tensor,
        gaze_pt_gt: torch.Tensor,
        head_bboxes_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        for gp_pred, gp_gt, hb_gt, io_gt in zip(gaze_pt_pred, gaze_pt_gt, head_bboxes_gt, inout_gt):
            if io_gt == 1:
                lah_pred = is_point_in_box(gp_pred, hb_gt).any().float().unsqueeze(0)
                lah_gt = is_point_in_box(gp_gt, hb_gt).any().int().unsqueeze(0)
                self.sum_plah += binary_precision(lah_pred, lah_gt, threshold=0.5)
                self.num_pos_obs += lah_pred.sum()

    def compute(self):
        if self.num_pos_obs != 0:
            plah = self.sum_plah / self.num_pos_obs
        else:
            plah = torch.tensor(-1000.0, device=self.device)
        return plah
    
class GFTestPLAH(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes Precision of Looking At Heads (PLAH) for general datasets.
        """
        super().__init__()
        self.add_state("sum_plah", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_pos_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_pt_pred: torch.Tensor,
        gaze_pt_gt: torch.Tensor,
        head_bboxes_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        for gp_pred, gp_gt, hb_gt, io_gt in zip(gaze_pt_pred, gaze_pt_gt, head_bboxes_gt, inout_gt):
            if io_gt == 1:
                lah_pred = is_point_in_box(gp_pred, hb_gt).any().float().unsqueeze(0)
                lah_gt = (is_point_in_box(gp_gt, hb_gt).sum(0) >= 2).any().int().unsqueeze(0)
                self.sum_plah += binary_precision(lah_pred, lah_gt, threshold=0.5)
                self.num_pos_obs += lah_pred.sum()
                self.num_obs += 1

    def compute(self):
        if self.num_pos_obs != 0:
            plah = self.sum_plah / self.num_pos_obs
        else:
            plah = torch.tensor(-1000.0, device=self.device)
        return plah

class Distance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_point_pred: torch.Tensor,
        gaze_point_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        mask = inout_gt == 1
        if mask.any():
            self.sum_dist += (gaze_point_gt[mask] - gaze_point_pred[mask]).pow(2).sum(1).sqrt().sum()
            self.num_obs += mask.sum()

    def compute(self):
        if self.num_obs != 0:
            dist = self.sum_dist / self.num_obs  # type: ignore
        else:
            dist = torch.tensor(-1000.0, device=self.device)
        return dist


class GFTestDistance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist_to_avg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_avg_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_min_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, gaze_point_pred: torch.Tensor, gaze_point_gt: torch.Tensor):
        for k, (gp_pred, gp_gt) in enumerate(zip(gaze_point_pred, gaze_point_gt)):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points

            # Compute average gaze point
            gp_gt_avg = gp_gt.mean(0)
            # Compute distance from pred to avg gt point
            self.sum_dist_to_avg += (gp_gt_avg - gp_pred).pow(2).sum().sqrt()
            # Compute avg distance between pred and gt points
            self.sum_avg_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().mean()
            # Compute min distance between pred and gt points
            self.sum_min_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().min()
        self.num_obs += len(gaze_point_pred)

    def compute(self):
        dist_to_avg = self.sum_dist_to_avg / self.num_obs
        avg_dist = self.sum_avg_dist / self.num_obs
        min_dist = self.sum_min_dist / self.num_obs
        return dist_to_avg, avg_dist, min_dist


class AUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes AUC for general datasets.
        """
        super().__init__()
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_heatmap_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        for hm_pred, hm_gt, io_gt in zip(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt):
            if io_gt == 1:
                hm_gt_binary = (hm_gt > 0).int()
                self.sum_auc += binary_auroc(hm_pred, hm_gt_binary)
        self.num_obs += inout_gt.sum()

    def compute(self):
        if self.num_obs != 0:
            auc = self.sum_auc / self.num_obs
        else:
            auc = torch.tensor(-1000.0, device=self.device)
        return auc


class GFTestAUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self):
        """
        Computes AUC for GazeFollow Test set. The AUC is computed for each image in the batch, after resizing the predicted
        heatmap to the original size of the image. The ground-truth binary heatmap is generated from the ground-truth gaze
        point(s) in the original image size. At the end, the mean is returned.
        """

        super().__init__()
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_pt: torch.Tensor,
    ):
        size = gaze_heatmap_pred.shape[1:]  # (b, h, w) >> (h, w)
        for hm_pred, gp_gt in zip(gaze_heatmap_pred, gaze_pt):
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points
            hm_gt_binary = generate_binary_gaze_heatmap(gp_gt, size=size)
            self.sum_auc += binary_auroc(hm_pred, hm_gt_binary)
        self.num_obs += len(gaze_heatmap_pred)

    def compute(self):
        auc = self.sum_auc / self.num_obs
        return auc
