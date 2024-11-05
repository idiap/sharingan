#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import datetime as dt
import warnings
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from termcolor import colored

from src.datasets.childplay import ChildPlayDataModule
from src.datasets.gazefollow import GazeFollowDataModule
from src.datasets.videoattentiontarget import VideoAttentionTargetDataModule
from src.modeling.sharingan import SharinganModule
from src.tracking import init_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)

TERM_COLOR = "cyan"


# ============================================================================================================ #
#                                             BASE EXPERIMENT CLASS                                            #
# ============================================================================================================ #
class BaseExperiment(ABC):
    """Base class for experiments."""

    @abstractmethod
    def parse_experiment(self, experiment):
        """Parse experiment string."""
        pass

    @abstractmethod
    def setup(self):
        """Setup experiment (e.g. create callbacks, init losses and metrics etc.)."""
        pass

    @abstractmethod
    def run(self):
        """Run experiment."""
        pass


# ============================================================================================================ #
#                                                 EXPERIMENT CLASS                                             #
# ============================================================================================================ #
class Experiment(BaseExperiment):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = self.parse_experiment(self.cfg.experiment.task)

    def parse_experiment(self, experiment):
        return experiment.split("+")

    def set_seed(self):
        if self.cfg.train.seed is not None:
            pl.seed_everything(self.cfg.train.seed)

    def init_model(self):
        model = SharinganModule(cfg=self.cfg)
        return model

    def init_data(self):
        if self.cfg.experiment.dataset == "gazefollow":
            data = GazeFollowDataModule(
                root=self.cfg.data.gf.root,
                root_annotations=self.cfg.data.gf.root_annotations,
                root_heads=self.cfg.data.gf.root_heads,
                batch_size={
                    "train": self.cfg.train.batch_size,
                    "val": self.cfg.val.batch_size,
                    "test": self.cfg.test.batch_size,
                },
                image_size=self.cfg.data.image_size,
                heatmap_size=self.cfg.data.heatmap_size,
                heatmap_sigma=self.cfg.data.heatmap_sigma,
                num_people=self.cfg.data.num_people,
                return_head_mask=self.cfg.data.return_head_mask,
            )
        elif self.cfg.experiment.dataset == "videoattentiontarget":
            data = VideoAttentionTargetDataModule(
                root=self.cfg.data.vat.root,
                root_heads=self.cfg.data.vat.root_heads,
                batch_size={
                    "train": self.cfg.train.batch_size,
                    "val": self.cfg.val.batch_size,
                    "test": self.cfg.test.batch_size,
                },
                stride=self.cfg.data.stride,
                image_size=self.cfg.data.image_size,
                heatmap_size=self.cfg.data.heatmap_size,
                heatmap_sigma=self.cfg.data.heatmap_sigma,
                num_people=self.cfg.data.num_people,
                return_head_mask=self.cfg.data.return_head_mask,
            )
        elif self.cfg.experiment.dataset == "childplay":
            data = ChildPlayDataModule(
                root=self.cfg.data.cp.root,
                root_heads=self.cfg.data.cp.root_heads,
                batch_size={
                    "train": self.cfg.train.batch_size,
                    "val": self.cfg.val.batch_size,
                    "test": self.cfg.test.batch_size,
                },
                stride=self.cfg.data.stride,
                image_size=self.cfg.data.image_size,
                heatmap_size=self.cfg.data.heatmap_size,
                heatmap_sigma=self.cfg.data.heatmap_sigma,
                num_people=self.cfg.data.num_people,
                return_head_mask=self.cfg.data.return_head_mask,
            )
        else:
            raise ValueError(
                f"Expected dataset to be one of [gazefollow, videoattentiontarget, childplay]. Got {self.cfg.experiment.dataset}."
            )
        print(
            colored(
                f"Using the {self.cfg.experiment.dataset.upper()} dataset.", TERM_COLOR
            )
        )
        return data

    def init_callbacks(self):
        callbacks = []

        checkpoint_cb = ModelCheckpoint(
            dirpath="./checkpoints",
            filename="best",  # custom: "{epoch:02d}-{step:02d}-{val_acc:.3f}"
            monitor=self.cfg.train.checkpointing.monitor,
            mode=self.cfg.train.checkpointing.mode,
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end=False,
            verbose=True,
        )
        callbacks.append(checkpoint_cb)

        if self.cfg.wandb.log:
            lr_monitor_callback = LearningRateMonitor(
                logging_interval="step", log_momentum=False
            )
            callbacks.append(lr_monitor_callback)

        if self.cfg.train.swa.use:
            print(colored(f"Using Stochastic Weight Averaging (SWA).", TERM_COLOR))
            swa_callback = StochasticWeightAveraging(
                swa_lrs=self.cfg.train.swa.lr,
                swa_epoch_start=self.cfg.train.swa.epoch_start,
                annealing_epochs=self.cfg.train.swa.annealing_epochs,
            )
            callbacks.append(swa_callback)

        return callbacks

    def init_trainer(self, logger, callbacks):
        profiler = None  # AdvancedProfiler(dirpath='.', filename='advanced-profile', line_count_restriction=1.0) #None
        trainer = pl.Trainer(
            accelerator="gpu" if self.cfg.train.device == "cuda" else "auto",
            precision=self.cfg.train.precision,  # 64 (double), 32 (full), 16 (16bit mixed), bf16-mixed (bfloat16 mixed). Defaults to 32.
            logger=logger,
            callbacks=callbacks,
            fast_dev_run=False,  # uncover bugs without any lengthy training by running all the code. Doesn't generate logs or checkpoints.
            max_epochs=self.cfg.train.epochs,
            overfit_batches=0.0,  # overfit one or a few batches to find bugs. Set it to 0 to disable.
            val_check_interval=1.0,  # int for nb of batches or float in [0., 1.] for fraction of the training epoch.
            check_val_every_n_epoch=1,  # Use None to validate every n batches through `val_check_interval`. default is 1.
            num_sanity_val_steps=2,  # Sanity check runs n val batches before the training routine. Set to -1 to run all batches.
            enable_checkpointing=True,  # If True, enable checkpointing. Configures a default one if there is no ModelCheckpoint callback.
            enable_progress_bar=True,  # Whether to enable to progress bar
            enable_model_summary=True,  # Whether to enable model summarization
            accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,  # accumulate gradients every k batches
            gradient_clip_val=None,  # clip gradients to this value
            gradient_clip_algorithm=None,  # "value" or "norm"
            deterministic=False,  # guarantee reproducible results by removing most of the randomness from training, but may slow it down.
            benchmark=True,  # set to True to speed up training if the input sizes for your model are fixed (e.g. during inference).
            inference_mode=False,  # Whether to use torch.inference_mode() or torch.no_grad() during evaluation (ie. validate/test/predict)
            profiler=profiler,  # None, "simple" or "advanced" to identify bottlenecks
            detect_anomaly=False,  # Enable anomaly detection for the autograd engine
        )
        return trainer

    def setup(self):
        print(colored("Setting up the experiment ...", TERM_COLOR))
        # SET SEED
        self.set_seed()

        # SET MATMUL PRECISION
        torch.set_float32_matmul_precision(self.cfg.train.matmul_precision)

        # INIT DATA MODULE
        self.data = self.init_data()

        # INIT LIGHTNING MODULE (ie. MODEL)
        self.model = self.init_model()

        # INIT LOGGER
        self.logger = init_logger(cfg=self.cfg) if ("train" in self.tasks) else None

        # INIT CALLBACKS
        self.callbacks = self.init_callbacks() if ("train" in self.tasks) else None

        # INIT TRAINER
        self.trainer = self.init_trainer(self.logger, self.callbacks)

    def train(self):
        # Log model parameters and/or gradients
        if (self.cfg.wandb.log) and (self.cfg.wandb.watch is not None):
            print(
                colored(f"Tracking model enabled: {self.cfg.wandb.watch}.", TERM_COLOR)
            )
            self.logger.watch(
                self.model,
                log=self.cfg.wandb.watch,
                log_freq=self.cfg.wandb.watch_freq,
                log_graph=False,
            )

        ckpt_path = self.cfg.train.resume_from if self.cfg.train.resume else None
        print(colored(f"Resuming model training from: `{ckpt_path}`.", TERM_COLOR))
        self.trainer.fit(self.model, self.data, ckpt_path=ckpt_path)

    def validate(self):
        ckpt_path = self.cfg.val.checkpoint
        print(colored(f"Validating model from: `{ckpt_path}`.", TERM_COLOR))
        self.trainer.validate(self.model, self.data, ckpt_path=ckpt_path, verbose=True)

    def test(self):
        ckpt_path = self.cfg.test.checkpoint if ("train" not in self.tasks) else "best"
        print(colored(f"Testing model from: `{ckpt_path}`.", TERM_COLOR))
        self.trainer.test(self.model, self.data, ckpt_path=ckpt_path, verbose=True)

    def run(self):
        print(colored("Starting the experiment ...", TERM_COLOR))
        start = dt.datetime.now()

        if "train" in self.tasks:
            self.train()

        if ("val" in self.tasks) and (
            "train" not in self.tasks
        ):  # validation is already included in training
            self.validate()

        if "test" in self.tasks:
            self.test()

        end = dt.datetime.now()
        print(colored(f"Finished. The experiment took {end - start}.", TERM_COLOR))
