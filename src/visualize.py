#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import math
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.patches import Rectangle

from src.utils.common import get_img_size

# Color palette
COLORS = [
    (100, 149, 237),
    (220, 20, 60),
    (60, 179, 113),
    (210, 105, 30),
    (255, 105, 180),
]


def draw_gaze(
    image: np.ndarray,
    gaze_point: Union[Tuple, List, np.ndarray],
    inout: Union[float, List[float]],
    head_bbox: Union[Tuple, List, np.ndarray],
    person_id: Union[int, List[int]],
    thr: float = 0.5,
    circle_thickness: int = -1,
):
    """
    Function to draw gaze results for a single person.

    Args:
        image: input image to draw gaze for.
        gaze_point: 2d coordinates of the gaze target point.
        inout: gaze class.
        head_bbox: head bounding box for person_id.
        person_id: id of the person for which to draw gaze and head bbox.
        heatmaps: predicted gaze heatmaps corresponding to the head bounding boxes.

    Returns:
        canvas or canvas_ext: the output image with gaze predictions drawn, and optionally, the other modalities.
    """
    # Convert to numpy
    if not isinstance(gaze_point, np.ndarray):
        gaze_point = np.array(gaze_point)
    if not isinstance(head_bbox, np.ndarray):
        head_bbox = np.array(head_bbox)

    # Create canvas on which to draw predictions
    img_h, img_w = image.shape[:2]
    canvas = image.copy()

    #
    gaze_point = gaze_point[np.newaxis, :] if gaze_point.ndim == 1 else gaze_point
    head_bbox = head_bbox[np.newaxis, :] if head_bbox.ndim == 1 else head_bbox
    inout = [inout] if isinstance(inout, float) else inout
    person_id = [person_id] if isinstance(person_id, int) else person_id

    n = len(gaze_point)
    thickness = 4
    fs = 0.6

    for i in range(n):
        x_min, y_min, x_max, y_max = head_bbox[i]
        pid = person_id[i]
        io = inout[i]
        io_text = f"inout: {io}"
        gp = gaze_point[i]
        if (0 <= gp[0] <= 1) and (0 <= gp[1] <= 1):
            gp = (gp * np.array([img_w, img_h])).astype(int)
        else:
            gp = gp.astype(int)

        color = COLORS[pid]

        # Compute Head Center
        head_center = np.array([int((x_min + x_max) / 2), int((y_min + y_max) / 2)], dtype=int)

        rec_pt1 = (int(x_min), int(y_min))
        rec_pt2 = (int(x_max), int(y_max))

        # Draw Head Bounding Box and Center Point
        canvas = cv2.rectangle(canvas, rec_pt1, rec_pt2, color, thickness)
        (w_text, h_text), _ = cv2.getTextSize(io_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        canvas = cv2.rectangle(canvas, head_center - 10, head_center + 10, color, -1)

        # Draw Gaze Class Header
        canvas = cv2.rectangle(canvas, rec_pt1, (int(x_min + w_text), int(y_min + h_text + 5)), color, -1)
        # Write Gaze Class
        canvas = cv2.putText(
            canvas,
            io_text,
            (int(x_min), int(y_min + h_text)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw Gaze Line if Score > Threshold
        if io > thr:
            canvas = cv2.line(canvas, head_center, gp, color, thickness)
            canvas = cv2.circle(canvas, gp, 10, color, circle_thickness)

    return canvas


def show_gazefollow_sample(sample, hm_alpha=0.3, colors=["crimson", "mediumpurple", "goldenrod"]):
    fig = plt.figure(figsize=(16, 8), layout="constrained")

    num_people = len(sample["heads"])
    n = math.ceil(num_people / 2) * 2
    spec = fig.add_gridspec(4, n)

    # Draw image + gaze heatmap + head mask
    ax11 = fig.add_subplot(spec[0:3, : n // 2])
    img_w, img_h = get_img_size(sample["image"])
    image = sample["image"].permute(1, 2, 0) if isinstance(sample["image"], torch.Tensor) else sample["image"]
    gaze_heatmap = TF.resize(sample["gaze_heatmap"].unsqueeze(0), (img_h, img_w), antialias=True)[0]
    ax11.imshow(image)
    if "head_masks" in sample:
        head_masks = sample["head_masks"]
        ax11.imshow(head_masks[-1, 0], cmap="gray", alpha=0.6)
    ax11.imshow(gaze_heatmap, cmap="viridis", alpha=hm_alpha)
    ax11.set_title(f"{sample['path']} | inout: {sample['inout'].item()} | (w, h) = ({img_w}, {img_h})")
    ax11.set_xlim(0, img_w)
    ax11.set_ylim(img_h, 0)
    ax11.axis("off")

    # Draw depth map
    ax12 = fig.add_subplot(spec[0:3, n // 2 :])
    if "depth" in sample:
        depth = sample["depth"][0] if isinstance(sample["depth"], torch.Tensor) else TF.to_tensor(sample["depth"]).squeeze(0)
        ax12.imshow(depth, cmap="gray")
    else:
        ax12.imshow(image)
    ax12.axis("off")

    # Draw head_bboxes, gaze points and gaze vectors
    heads = sample["heads"].permute(0, 2, 3, 1) if isinstance(sample["heads"], torch.Tensor) else sample["heads"]
    head_bboxes = sample["head_bboxes"] * torch.tensor([[img_w, img_h, img_w, img_h]])
    head_centers = sample["head_centers"] * torch.tensor([[img_w, img_h]])
    gaze_pt = sample["gaze_pt"]
    gaze_vec = sample["gaze_vec"]
    if gaze_pt.ndim == 1:
        gaze_pt = gaze_pt.unsqueeze(0)
        gaze_vec = gaze_vec.unsqueeze(0)
    mask = gaze_pt[:, 0] != -1
    gaze_pt = gaze_pt[mask] * torch.tensor([img_w, img_h])
    gaze_vec = gaze_vec[mask] * 30

    for j in range(len(head_bboxes)):
        color = "white"
        linestyle = "-."
        if j == len(head_bboxes) - 1:
            color = colors[0]
            linestyle = "-"
            hcx, hcy = head_centers[j].tolist()
            for gp, gv in zip(gaze_pt, gaze_vec):
                gpx, gpy = gp.tolist()
                gvx, gvy = gv.tolist()
                ax11.scatter(gpx, gpy, s=50, color=color)
                ax11.arrow(hcx, hcy, gpx - hcx, gpy - hcy, color=color)
                ax11.arrow(hcx, hcy, gvx, gvy, color="white", head_width=3)

        xmin, ymin, xmax, ymax = head_bboxes[j]

        head = heads[j]
        head_w, head_h = get_img_size(head)
        ax = fig.add_subplot(spec[3, j])
        ax.imshow(head)
        ax.axis("off")
        ax.set_title(f"(w, h) = ({head_w}, {head_h})")

        rec = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor="none", edgecolor=color, linestyle=linestyle, linewidth=2)
        ax11.add_patch(rec)

        
def show_videoatt_sample(sample, hm_alpha=0.3, colors=["crimson", "mediumpurple", "goldenrod", "cornflowerblue", "mediumaquamarine", "white"]):
    
    fig = plt.figure(figsize=(16, 8), layout="constrained")

    num_people = len(sample["heads"])
    n = math.ceil(num_people / 2) * 2
    spec = fig.add_gridspec(4, n)

    # Draw image + gaze heatmap + head mask
    ax11 = fig.add_subplot(spec[0:3, : n // 2])
    img_w, img_h = get_img_size(sample["image"])
    image = sample["image"].permute(1, 2, 0) if isinstance(sample["image"], torch.Tensor) else sample["image"]
    gaze_heatmap = TF.resize(sample["gaze_heatmap"].unsqueeze(0), (img_h, img_w), antialias=True)[0]
    ax11.imshow(image)
    if "head_masks" in sample:
        head_masks = sample["head_masks"]
        ax11.imshow(head_masks[-1, 0], cmap="gray", alpha=0.6)
        
    inout = sample["inout"]
    if inout == 1.:
        ax11.imshow(gaze_heatmap, cmap="viridis", alpha=hm_alpha)
    ax11.set_title(f"{sample['path']} | inout: {sample['inout'].item()} | (w, h) = ({img_w}, {img_h})")
    ax11.set_xlim(0, img_w)
    ax11.set_ylim(img_h, 0)
    ax11.axis("off")

    # Draw depth map
    ax12 = fig.add_subplot(spec[0:3, n // 2 :])
    if "depth" in sample:
        depth = sample["depth"][0] if isinstance(sample["depth"], torch.Tensor) else TF.to_tensor(sample["depth"]).squeeze(0)
        ax12.imshow(depth, cmap="gray")
    else:
        ax12.imshow(image)
    ax12.axis("off")

    # Draw head_bboxes, gaze points and gaze vectors
    heads = sample["heads"].permute(0, 2, 3, 1) if isinstance(sample["heads"], torch.Tensor) else sample["heads"]
    head_bboxes = sample["head_bboxes"] * torch.tensor([[img_w, img_h, img_w, img_h]])
    head_centers = sample["head_centers"] * torch.tensor([[img_w, img_h]])
    gaze_pt = sample["gaze_pt"] * torch.tensor([img_w, img_h])
    gaze_vec = sample["gaze_vec"] * 30

    for j in range(len(head_bboxes)):
        color = "white"
        linestyle = "-."
        
        if (j == len(head_bboxes) - 1):
            color = colors[0]
            linestyle = "-"
            if inout == 1.:
                hcx, hcy = head_centers[j].tolist()
                gpx, gpy = gaze_pt.tolist()
                gvx, gvy = gaze_vec.tolist()
                ax11.scatter(gpx, gpy, s=50, color=color)
                ax11.arrow(hcx, hcy, gpx - hcx, gpy - hcy, color=color)
                ax11.arrow(hcx, hcy, gvx, gvy, color="white", head_width=3)

        xmin, ymin, xmax, ymax = head_bboxes[j]

        head = heads[j]
        head_w, head_h = get_img_size(head)
        ax = fig.add_subplot(spec[3, j])
        ax.imshow(head)
        ax.axis("off")
        ax.set_title(f"(w, h) = ({head_w}, {head_h})")

        rec = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor="none", edgecolor=color, linestyle=linestyle, linewidth=2)
        ax11.add_patch(rec)