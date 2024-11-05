#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_sharingan_loss(gaze_vec_gt, gaze_heatmap_gt, inout_gt, gaze_vec_pred, gaze_heatmap_pred, inout_pred, w_hm=1000, w_ang=3, w_bce=0):
    heatmap_loss = torch.tensor(0.0)
    angular_loss = torch.tensor(0.0)
        
    if torch.sum(inout_gt) > 0:  # to avoid case where all samples of the batch are outside (i.e. division by 0)
        heatmap_loss = compute_heatmap_loss(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt)
        angular_loss = compute_angular_loss(gaze_vec_pred, gaze_vec_gt, inout_gt)
        
    bce_loss = compute_bce_loss(inout_pred, inout_gt, use_focal_loss=False)
    total_loss = w_hm * heatmap_loss + w_ang * angular_loss + w_bce * bce_loss

    logs = {
        "heatmap_loss": heatmap_loss.item(),
        "angular_loss": angular_loss.item(),
        "bce_loss": bce_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, logs


def compute_dist_loss(gp_pred, gp_gt, io_gt):
    dist_loss = (gp_pred - gp_gt).pow(2).sum(dim=1)
    dist_loss = torch.mul(dist_loss, io_gt)
    dist_loss = torch.sum(dist_loss) / torch.sum(io_gt)
    return dist_loss

def compute_heatmap_loss(hm_pred, hm_gt, io_gt):
    heatmap_loss = F.mse_loss(hm_pred, hm_gt, reduce=False).mean([1, 2])
    heatmap_loss = torch.mul(heatmap_loss, io_gt)
    heatmap_loss = torch.sum(heatmap_loss) / torch.sum(io_gt)
    return heatmap_loss

def compute_angular_loss(gv_pred, gv_gt, io_gt):
    angular_loss = (1 - torch.einsum("ij,ij->i", gv_pred, gv_gt)) / 2
    angular_loss = torch.mul(angular_loss, io_gt)
    angular_loss = torch.sum(angular_loss) / torch.sum(io_gt)
    return angular_loss

def compute_bce_loss(io_pred, io_gt, use_focal_loss=False, gamma=2):
    if use_focal_loss:
        bce_loss = F.binary_cross_entropy_with_logits(io_pred, io_gt, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** gamma * bce_loss
        bce_loss = torch.mean(focal_loss)
    else:
        bce_loss = F.binary_cross_entropy_with_logits(io_pred, io_gt)
    return bce_loss
