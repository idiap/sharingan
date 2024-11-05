#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

from ctypes import Union

import os
import shutil
import omegaconf
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger

from termcolor import colored

TERM_COLOR = "cyan"


def init_logger(cfg):
    if cfg.wandb.log:
        id = wandb.util.generate_id()  # type: ignore
        logger = WandbLogger(
            project=cfg.project.name,
            entity="samyer",
            group=cfg.experiment.group,
            log_model=False,
            id=id,
            name=cfg.experiment.name,
            save_dir="./",
            allow_val_change=True,
        )

        cfg_dict = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        logger.experiment.config.update(cfg_dict)
    else:
        logger = False

    return logger


def save_code_snapshot(code_folder, output_folder="."):
    print(colored("Saving a snapshot of the source code folder ...", TERM_COLOR), end=" ")
    output_basename = os.path.join(output_folder, "src")
    shutil.make_archive(output_basename, "zip", code_folder)
    print(colored("Done.", TERM_COLOR))
    