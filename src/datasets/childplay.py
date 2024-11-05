#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
from glob import glob
from typing import Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from src.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from src.utils.common import (
    Stage,
    expand_bbox,
    generate_gaze_heatmap,
    generate_mask,
    get_img_size,
    pair,
    square_bbox,
)

IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]


# ============================================================================= #
#                                CHILDPLAY DATASET                              #
# ============================================================================= #
class ChildPlayDataset(Dataset):
    def __init__(
        self,
        root,
        root_heads,
        split: str = "train",
        stride: int = 1,
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 2,
        head_thr: float = 0.5,
        return_depth: bool = False,
        return_head_mask: bool = False,
    ):

        super().__init__()

        assert split in (
            "train",
            "val",
            "test",
        ), f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        assert (num_people == -1) or (
            num_people > 0
        ), f"Expected `num_people` to be strictly positive or `-1`, but received {num_people} instead."

        self.root = root
        self.split = split
        self.stride = stride
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.root_heads = root_heads
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.annotations = self.load_annotations()

    def load_annotations(
        self,
        exclude_cls=[
            "gaze_shift",
            "inside_occluded",
            "inside_uncertain",
            "eyes_closed",
        ],
    ):
        annotation_files = glob(
            os.path.join(self.root, "annotations", self.split, "*.csv")
        )
        li = []
        for file in annotation_files:
            df = pd.read_csv(file)
            li.append(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)

        # Filter Annotations based on stride
        if (self.split == "train") and (self.stride > 1):
            mask = (annotations.frame % self.stride == 1)  # keep 1 frame in every `self.stride` frames
            annotations = annotations[mask].reset_index(drop=True)

        # Filter Annotations based on gaze class
        if len(exclude_cls) > 0:
            cond = annotations.gaze_class.isin(exclude_cls)
            annotations = annotations[~cond].reset_index(drop=True)

        # Clean annotations (remove cases where inside_visible & gaze_pt = (-1, -1))
        cond = (annotations.gaze_class == "inside_visible") & (annotations.gaze_x == -1.0)
        annotations = annotations[~cond].reset_index(drop=True)

        return annotations

    def __getitem__(self, index: int) -> Dict:

        item = self.annotations.iloc[index]  # get the annotation row from the dataframe
        clip_name = item["clip"]
        video_id, interval = clip_name.replace("-downsample", "").rsplit("_", 1)
        offset = int(interval.split("-")[0])
        frame_nb = item["frame"]

        # Load image
        basename = f"{video_id}_{offset + frame_nb - 1}"
        img_name = f"{basename}.jpg"
        path = os.path.join(clip_name, img_name)
        img_path = os.path.join(self.root, "images", path)
        image = Image.open(img_path)
        img_w, img_h = image.size

        # Load pid, inout and gaze point
        pid = item["person_id"]
        inout = 1.0 if item["gaze_class"] == "inside_visible" else 0.0
        inout = torch.tensor(inout, dtype=torch.float)
        gaze_pt = torch.tensor([item["gaze_x"] / img_w, item["gaze_y"] / img_h], dtype=torch.float)

        # Load head bboxes
        ## For target person
        bbox_xmin, bbox_ymin, bbox_w, bbox_h = (item["bbox_x"], item["bbox_y"], item["bbox_width"], item["bbox_height"])
        target_head_bbox = [bbox_xmin, bbox_ymin, bbox_xmin + bbox_w, bbox_ymin + bbox_h]
        target_head_bbox = torch.tensor(target_head_bbox, dtype=float).unsqueeze(0)
        # target_head_bbox = expand_bbox(target_head_bbox, img_w, img_h, k=0.05) # ChildPlay head bboxes are not tight
        target_head_center = torch.stack([target_head_bbox[0, [0, 2]].mean(), target_head_bbox[0, [1, 3]].mean()])
        target_head_center /= torch.tensor([img_w, img_h], dtype=torch.float)

        ## For context people (ie. detected w/ Yolo)
        context_head_bboxes = torch.zeros((0, 4))
        if (self.num_people == -1) or (self.num_people > 1):
            det_file = f"{clip_name}/{basename}-head-detections.npy"
            detections = np.load(os.path.join(self.root_heads, det_file))

            # Process context head bboxes
            if len(detections) > 0:
                scores = torch.tensor(detections[:, -1])
                context_head_bboxes = torch.tensor(detections[(scores >= self.head_thr).tolist(), :-1])
                ious = box_iou(context_head_bboxes, target_head_bbox).flatten()
                # TODO: parametrize iou threshold
                context_head_bboxes = context_head_bboxes[ious <= 0.5]

            # Shuffle context people and keep the first `num_people - 1` indices
            if self.split == "train":
                perm_indices = torch.randperm(context_head_bboxes.size(0))
                context_head_bboxes = context_head_bboxes[perm_indices]
            num_context_heads = len(context_head_bboxes)
            num_keep = num_context_heads if self.num_people == -1 else self.num_people - 1
            context_head_bboxes = context_head_bboxes[:num_keep]

        # Concatenate main head bbox with others and apply jitter
        head_bboxes = torch.concat([context_head_bboxes, target_head_bbox], dim=0).to(torch.float)
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract Heads (negative values add padding)
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))  # type:ignore

        # Normalize Head Bboxes and clip to [0, 1]
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        head_bboxes = torch.clamp(head_bboxes, min=0.0, max=1.0)

        # Build Sample
        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pt": gaze_pt,
            "inout": inout,
            "id": pid,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": path,
        }

        # Transform
        if self.transform:
            sample = self.transform(sample)

        # Pad missing people (ie. heads + bboxes)
        num_heads = len(head_bboxes)
        num_missing_heads = self.num_people - num_heads if self.num_people != -1 else 0
        if num_missing_heads > 0:
            pad = (0, 0, num_missing_heads, 0)
            sample["head_bboxes"] = F.pad(sample["head_bboxes"], pad, mode="constant", value=0.0)
            if isinstance(sample["heads"], torch.Tensor):
                pad = (0, 0, 0, 0, 0, 0, num_missing_heads, 0)
                sample["heads"] = F.pad(sample["heads"], pad, mode="constant", value=0.0)
            else:
                sample["heads"] = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * num_missing_heads + heads

        # Compute head centers
        sample["head_centers"] = torch.hstack(
            [
                (sample["head_bboxes"][:, [0]] + sample["head_bboxes"][:, [2]]) / 2,
                (sample["head_bboxes"][:, [1]] + sample["head_bboxes"][:, [3]]) / 2,
            ]
        )

        # Generate gaze heatmap
        if sample["inout"] == 1.0:
            sample["gaze_heatmap"] = generate_gaze_heatmap(sample["gaze_pt"], sigma=self.heatmap_sigma, size=self.heatmap_size)
        else:
            sample["gaze_heatmap"] = torch.zeros((self.heatmap_size, self.heatmap_size), dtype=torch.float)

        # Compute gaze vec
        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][-1]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)

        # Generate head mask
        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample

    def __len__(self):
        return len(self.annotations)


# ============================================================================= #
#                              CHILDPLAY DATAMODULE                             #
# ============================================================================= #
class ChildPlayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_heads: str,
        stride: int = 1,
        image_size: Union[int, tuple[int, int]] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 2,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
        batch_size: Union[int, dict] = 32,
    ):

        super().__init__()
        self.root = root
        self.root_heads = root_heads
        self.image_size = pair(image_size)
        self.stride = stride
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.batch_size = (
            {stage: batch_size for stage in ["train", "val", "test"]}
            if isinstance(batch_size, int)
            else batch_size
        )

    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=1.0, p_safe=0.0),
                    RandomHorizontalFlip(p=0.5),
                    ColorJitter(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.0, 1.5),
                        hue=None,
                        p=0.8,
                    ),
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.train_dataset = ChildPlayDataset(
                root=self.root,
                root_heads=self.root_heads,
                split="train",
                stride=self.stride,
                transform=train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                head_thr=self.head_thr,
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = ChildPlayDataset(
                root=self.root,
                root_heads=self.root_heads,
                split="val",
                stride=1,
                transform=val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=5,  # 5 is an arbitrary high number to avoid validating with all people during training.
                head_thr=self.head_thr,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "validate":
            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = ChildPlayDataset(
                root=self.root,
                root_heads=self.root_heads,
                split="val",
                stride=1,
                transform=val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=-1,  # always validate with all people in the image
                head_thr=self.head_thr,
                return_head_mask=self.return_head_mask,
            )

        elif stage == "test":
            test_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.test_dataset = ChildPlayDataset(
                root=self.root,
                root_heads=self.root_heads,
                split="test",
                stride=1,
                transform=test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=-1,  # always test with all people in the image
                head_thr=self.head_thr,
                return_head_mask=self.return_head_mask,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size["train"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size["val"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size["test"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return dataloader
