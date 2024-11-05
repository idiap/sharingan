#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
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
from src.utils.common import pair, expand_bbox, generate_gaze_heatmap, generate_mask, get_img_size, square_bbox



IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

# ============================================================================= #
#                               GAZEFOLLOW DATASET                              #
# ============================================================================= #
class GazeFollowDataset(Dataset):
    def __init__(
        self,
        root,
        root_annotations,
        root_heads,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 2,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        assert (num_people == -1) or (num_people > 0), f"Expected `num_people` to be strictly positive or `-1`, but received {num_people} instead."
        assert 0 <= head_thr <= 1, f"Expected `head_thr` to be in [0, 1]. Received {head_thr} instead."

        self.root = root
        self.root_annotations = root_annotations
        self.root_heads = root_heads
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.annotations = self.load_annotations()

    def load_annotations(self) -> pd.DataFrame:
        annotations = pd.DataFrame()
        if self.split == "test":
            column_names = ["path", "id", "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h", "eye_x", "eye_y", 
                            "gaze_x", "gaze_y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root, "test_annotations_release.txt"),
                sep=",",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Add inout col for consistency (ie. missing from test set)
            annotations["inout"] = 1
            # Each test image is annotated by multiple people (around 10 on avg.)
            self.image_paths = annotations.path.unique().tolist()
            self.length = len(self.image_paths)

        elif self.split in ["train", "val"]:
            column_names = ["path", "id", "body_bbox_x", "body_bbox_y", "body_bbox_w", "body_bbox_h", "eye_x", "eye_y", 
                            "gaze_x", "gaze_y", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max", "inout", "origin", "meta"]
            annotations = pd.read_csv(
                os.path.join(self.root_annotations, f"{self.split}_annotations_new.txt"), # reprocessed train/val head bboxes
                sep=",",
                names=column_names,
                index_col=False,
                encoding="utf-8-sig",
            )
            # Clean annotations (e.g. remove invalid ones)
            annotations = self._clean_annotations(annotations)
            self.length = len(annotations)

        return annotations

    def _clean_annotations(self, annotations):
        # Only keep "in" and "out". (-1 is invalid)
        annotations = annotations[annotations.inout != -1]
        # Discard instances where max in bbox coordinates is smaller than min
        annotations = annotations[annotations.bbox_x_min < annotations.bbox_x_max]
        annotations = annotations[annotations.bbox_y_min < annotations.bbox_y_max]
        return annotations.reset_index(drop=True)

    def __getitem__(self, index: int) -> Dict:
        if self.split in ["train", "val"]:
            item = self.annotations.iloc[index]
            gaze_pt = torch.tensor([item["gaze_x"], item["gaze_y"]], dtype=torch.float)
            idx = item["id"]
        elif self.split == "test":
            image_path = self.image_paths[index]
            p_annotations = self.annotations[self.annotations.path == image_path]
            gaze_pt = torch.from_numpy(p_annotations[["gaze_x", "gaze_y"]].values).float()
            p = 20 - len(gaze_pt)
            # Pad to have same length across samples for dataloader
            gaze_pt = F.pad(gaze_pt, (0, 0, 0, p), value=-1.0)
            idx = p_annotations["id"].values.tolist() + [-1] * p  # pad to 20 for consistency
            item = p_annotations.iloc[0]

        # eyes_pt = torch.tensor([item["eye_x"], item["eye_y"]], dtype=torch.float) # not used
        inout = torch.tensor(item["inout"], dtype=torch.float)
        path = item["path"]
        split, partition, img_name = item["path"].split('/')
        basename, ext = os.path.splitext(img_name)

        # Load image
        image = Image.open(os.path.join(self.root, item["path"])).convert("RGB")
        img_w, img_h = image.size
        
        # Load head bboxes
        ## For target person
        target_head_bbox = item[["bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]]
        target_head_bbox = torch.from_numpy(target_head_bbox.values.astype(np.float32)).unsqueeze(0)
        target_head_bbox = expand_bbox(target_head_bbox, img_w, img_h, k=0.1) # annotated boxes are a bit tight
        target_head_center =  torch.stack([target_head_bbox[0, [0, 2]].mean(), target_head_bbox[0, [1, 3]].mean()])
        target_head_center /= torch.tensor([img_w, img_h], dtype=torch.float)

        ## For context people (ie. detected w/ Yolo)
        context_head_bboxes = torch.zeros((0, 4))
        if (self.num_people == -1) or (self.num_people > 1):
            det_file = f"{split}/{partition}/{basename}-head-detections.npy"
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
            "id": idx,
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
            sample["head_bboxes"] = F.pad(sample["head_bboxes"], pad, mode="constant", value=0.)
            if isinstance(sample["heads"], torch.Tensor):
                pad = (0, 0, 0, 0, 0, 0, num_missing_heads, 0)
                sample["heads"] = F.pad(sample["heads"], pad, mode="constant", value=0.)
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
        
        # Compute gaze vector (only for target person)
        new_img_w, new_img_h = get_img_size(sample["image"])
        gaze_vec = sample["gaze_pt"] - sample["head_centers"][-1]
        gaze_vec = gaze_vec * torch.tensor([new_img_w, new_img_h])
        sample["gaze_vec"] = F.normalize(gaze_vec, p=2, dim=-1)
        
        # Generate head mask
        if self.return_head_mask:
            sample["head_masks"] = generate_mask(sample["head_bboxes"], new_img_w, new_img_h)

        return sample

    def __len__(self):
        return self.length


# ============================================================================= #
#                             GAZEFOLLOW DATAMODULE                             #
# ============================================================================= #
class GazeFollowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        root_annotations: str,
        root_heads: str,
        batch_size: Union[int, dict] = 32,
        image_size: Union[int, tuple[int, int]] = (224, 224),
        heatmap_sigma: int = 3,
        heatmap_size: Union[int, tuple[int, int]] = 64,
        num_people: dict = {"train": 1, "val": 1, "test": 1},
        return_head_mask: bool = False,
    ):
        super().__init__()
        self.root = root
        self.root_annotations = root_annotations
        self.root_heads = root_heads
        self.image_size = pair(image_size)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.batch_size = {stage: batch_size for stage in ["train", "val", "test"]} if isinstance(batch_size, int) else batch_size
        self.return_head_mask = return_head_mask
        
    def setup(self, stage: str):
        if stage == "fit":
            train_transform = Compose(
                [
                    RandomCropSafeGaze(aspect=1.0, p=0.8, p_safe=1.0),
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
            self.train_dataset = GazeFollowDataset(
                self.root,
                self.root_annotations,
                self.root_heads,
                "train",
                train_transform,
                tr=(-0.1, 0.1),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=self.num_people,
                return_head_mask=self.return_head_mask,
            )

            val_transform = Compose(
                [
                    Resize(img_size=self.image_size, head_size=(224, 224)),
                    ToTensor(),
                    Normalize(img_mean=IMG_MEAN, img_std=IMG_STD),
                ]
            )
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_annotations,
                self.root_heads,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=-1, # always valide with all people in the image
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
            self.val_dataset = GazeFollowDataset(
                self.root,
                self.root_annotations,
                self.root_heads,
                "val",
                val_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=-1, # always valide with all people in the image
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
            self.test_dataset = GazeFollowDataset(
                self.root,
                self.root_annotations,
                self.root_heads,
                "test",
                test_transform,
                tr=(0.0, 0.0),
                heatmap_size=self.heatmap_size,
                heatmap_sigma=self.heatmap_sigma,
                num_people=-1, # always test with all people in the image
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
