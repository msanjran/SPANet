from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ
import shutil
import json
import random
import numpy as np
import sys
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loggers.wandb import _WANDB_AVAILABLE, WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar
)

from spanet import JetReconstructionModel, Options

def set_global_seed(level: str, seed: int) -> None:
    ''' Set seed for reproducibility '''
    print(f"Setting global seed for training")
    print(f" - {level}: {seed}")

    if level == "pl_everything":
        pl.seed_everything(seed)
        # apparently these aren't set automatically
        # by seed_everything
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif level == "manual_everything":

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        raise NotImplementedError(f"{level} not recognised seed setting")

# def set_seed(seed: int = 42) -> None:
#     """Set seeds for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(0)

def clean_fpath(fpath):
    fpath = fpath.replace("'","")
    fpath = fpath.replace(",","")
    return fpath

def main(
        event_file: str,
        training_file: str,
        validation_file: str,
        options_file: Optional[str],
        checkpoint: Optional[str],
        state_dict: Optional[str],
        freeze_state_dict: bool,

        log_dir: str,
        name: str,

        torch_script: bool,
        fp16: bool,
        verbose: bool,
        full_events: bool,

        profile: bool,
        gpus: Optional[int],
        epochs: Optional[int],
        time_limit: Optional[str],
        batch_size: Optional[int],
        limit_dataset: Optional[float],
        random_seed: int,
        custom_mask_train: Optional[str],
        custom_mask_val: Optional[str],
        rand_control: Optional[str],
        rand_control_seed: Optional[int],
        clip_train: Optional[str],
        clip_val: Optional[str],
        val_shuffle: bool,
        val_include_last: bool,
        save_top_X: int,
        notebook_mode: int,
        dont_limit_index_sort: bool,
        shuffle_by_sample: bool
    ):

    # args_dict = locals()
    # print(args_dict)

    # sys.exit()

    if rand_control is not None and rand_control_seed is not None:
        set_global_seed(rand_control, rand_control_seed)

    ## clean because trainer_submit.py being annoying as f
    event_file = clean_fpath(event_file)
    training_file = clean_fpath(training_file)
    validation_file = clean_fpath(validation_file)
    if options_file is not None:
        options_file = clean_fpath(options_file)

    random_seed = int(random_seed)

    # Whether or not this script version is the master run or a worker
    master = True
    if "NODE_RANK" in environ:
        master = False

    # -------------------------------------------------------------------------------------------------------
    # Create options file and load any optional extra information.
    # -------------------------------------------------------------------------------------------------------
    options = Options(event_file, training_file, validation_file)
    
    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file), update_datasets=False)

    # -------------------------------------------------------------------------------------------------------
    # Command line overrides for common option values.
    # -------------------------------------------------------------------------------------------------------
    options.verbose_output = verbose
    if master and verbose:
        print(f"Verbose output activated.")

    if full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = epochs
    
    if custom_mask_train is not None:
        if master:
            print(f"Overriding 'train_mask'")
        options.train_custom_mask = custom_mask_train
    
    if custom_mask_val is not None:
        if master:
            print(f"Overriding 'val_mask'")
        options.val_custom_mask = custom_mask_val
    
    if clip_train is not None:
        if master:
            print(f"Overriding 'clip_train'")
        options.clip_train = clip_train
    
    if clip_val is not None:
        if master:
            print(f"Overriding 'clip_val'")
        options.clip_val = clip_val

    # bookkeeping
    if rand_control is not None and rand_control_seed is not None:
        options.global_seed_method = rand_control
        options.global_seed_number = rand_control_seed
    
    # val dataloader options (mainly for debugging)
    if val_shuffle:
        if master:
            print(f"Overriding 'val_dataloader_shuffle' {options.val_dataloader_shuffle} to {val_shuffle}")
        options.val_dataloader_shuffle = val_shuffle
    if val_include_last:
        if master:
            print(f"Overriding 'val_dataloader_drop_last' {options.val_dataloader_drop_last} to {not val_include_last}")
        options.val_dataloader_drop_last = not val_include_last

    if random_seed > 0:
        options.dataset_randomization = random_seed
    
    # settings on loading in datasets
    if dont_limit_index_sort:
        if master:
            print(f"Overriding 'limit_index_sorting' {options.limit_index_sorting} to {not dont_limit_index_sort}")
        options.limit_index_sorting = not dont_limit_index_sort
    if shuffle_by_sample:
        if master:
            print(f"Overriding 'shuffle_by_sample' {options.shuffle_by_sample} to {shuffle_by_sample}")
        options.shuffle_by_sample = shuffle_by_sample

    # -------------------------------------------------------------------------------------------------------
    # Print the full hyperparameter list
    # -------------------------------------------------------------------------------------------------------
    if master:
        options.display()

    # -------------------------------------------------------------------------------------------------------
    # Begin the training loop
    # -------------------------------------------------------------------------------------------------------

    # Create the initial model on the CPU
    model = JetReconstructionModel(options, torch_script)

    if state_dict is not None:
        if master:
            print(f"Loading state dict from: {state_dict}")

        state_dict = torch.load(state_dict, map_location="cpu")["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if master:
            print(f"Missing Keys: {missing_keys}")
            print(f"Unexpected Keys: {unexpected_keys}")

        if freeze_state_dict:
            for pname, parameter in model.named_parameters():
                if pname in state_dict:
                    parameter.requires_grad_(False)

    # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True) # idk
    log_dir = getcwd() if log_dir is None else log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=name)
    # logger = (
    #     WandbLogger(name=name, save_dir=log_dir)
    #     if _WANDB_AVAILABLE else
    #     TensorBoardLogger(save_dir=log_dir, name=name)
    # )

    # Create the checkpoint for this training run. We will save the best validation networks based on 'accuracy'
    # Old: pre 16may25
    # callbacks = [
    #     ModelCheckpoint(
    #         verbose=options.verbose_output,
    #         monitor='validation_accuracy',
    #         save_top_k=-1,
    #         mode='max',
    #         save_last=True
    #     ),
    #     ...
    # ]
    # New: post 16may25 --> but I've kept save_top_k=-1 instead of save_top_k=3
    callbacks = [
        ModelCheckpoint(
            verbose=options.verbose_output,
            filename='{epoch}-{step}-{validation_average_jet_accuracy:.3f}',
            monitor='validation_average_jet_accuracy',
            save_top_k=save_top_X,
            mode='max',
            save_last=True
        ),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        RichProgressBar() if (_RICH_AVAILABLE==True and notebook_mode==False) else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if (_RICH_AVAILABLE==True and notebook_mode==False) else ModelSummary(max_depth=1)
    ]

    epochs = options.epochs
    profiler = None
    if profile:
        epochs = 1
        profiler = PyTorchProfiler(emit_nvtx=True)

    # Create the final pytorch-lightning manager
    trainer = pl.Trainer(
        accelerator="gpu" if options.num_gpu > 0 else "auto",
        devices=options.num_gpu if options.num_gpu > 0 else "auto",
        strategy="ddp" if options.num_gpu > 1 else "auto",
        precision="16-mixed" if fp16 else "32-true",

        gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
        max_epochs=epochs,
        max_time=time_limit,

        logger=logger,
        profiler=profiler,
        callbacks=callbacks
    )

    # Save the current hyperparameters to a json file in the checkpoint directory
    if master:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)

        with open(f"{trainer.logger.log_dir}/options.json", 'w') as json_file:
            json.dump(options.__dict__, json_file, indent=4)

        shutil.copy2(options.event_info_file, f"{trainer.logger.log_dir}/event.yaml")

        # with open(f"{trainer.logger.log_dir}/args.json", "w") as json_file:
        #     json.dump(args_dict, json_file, indent=4)

        # copy the arguments into an 'args' file so easier to track...

        # save indices if we're getting val split from train split
        # for bookkeeping
        saved_train = model.training_dataset.save_indices_to_file(
            os.path.join(trainer.logger.log_dir, f"train_split_idx.npy"))
        saved_val = model.validation_dataset.save_indices_to_file(
            os.path.join(trainer.logger.log_dir, f"val_split_idx.npy"))
        if saved_train and saved_val:
            print(f"Saved train/val split indices")
        

    trainer.fit(model, ckpt_path=checkpoint)
    # -------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-ef", "--event_file", type=str, default="",
                        help="Input file containing event symmetry information.")

    parser.add_argument("-tf", "--training_file", type=str, default="",
                        help="Input file containing training data.")

    parser.add_argument("-vf", "--validation_file", type=str, default="",
                        help="Input file containing Validation data. If not provided, will use training data split.")

    parser.add_argument("-of", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load the training state from. "
                             "Fully restores model weights and optimizer state.")

    parser.add_argument("-sf", "--state_dict", type=str, default=None,
                        help="Load from checkpoint but only the model weights. "
                             "Can be partial as the weights don't have to match one-to-one.")

    parser.add_argument("-fsf", "--freeze_state_dict", action='store_true',
                        help="Freeze any weights that were loaded from the state dict. "
                             "Used for finetuning new layers.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")

    parser.add_argument("-n", "--name", type=str, default="spanet_output",
                        help="The sub-directory to create for this run and an identifier for WANDB.")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override number of epochs to train for")
    
    parser.add_argument("-t", "--time_limit", type=str, default=None,
                        help="Time limit for training, in the format DD:HH:MM:SS.")

    parser.add_argument("-g", "--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-f", "--full_events", action='store_true',
                        help="Limit training to only full events.")

    parser.add_argument("-p", "--limit_dataset", type=float, default=None,
                        help="Limit dataset to only the first L percent of the data (0 - 100).")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Torch AMP for training.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-r", "--random_seed", default=0,
                        help="Set random seed for cross-validation.")

    parser.add_argument("-ts", "--torch_script", action='store_true',
                        help="Compile the neural network using torchscript.")

    parser.add_argument("--profile", action='store_true',
                        help="Profile network for a single training epoch.")

    parser.add_argument("--custom_mask_train", default=None,
                        help="Path to a 'npy' file containing boolean event mask (train)")
    
    parser.add_argument("--custom_mask_val", default=None, type=str, 
                        help="Path to a 'npy' file containing boolean event mask (val)")

    parser.add_argument("--rand_control", type=str, default=None, choices=["pl_everything", "manual_everything"],
                        help="What seed control level to use --> please only use if 'random_seed' is 0")
    
    parser.add_argument("--rand_control_seed", type=int, default=None,
                        help="What seed to use for the 'rand_control' argument"
                        " (must be given if you wanna use the above)")
    
    parser.add_argument("--clip_train", default=None,
                        help="Path to file describing inputs to clip for training dataset")
    
    parser.add_argument("--clip_val", default=None,
                        help="Path to file describing inputs to clip for validation dataset")

    parser.add_argument("--val_shuffle", default=False, action='store_true',
                        help="Flag to shuffle the validation dataset each epoch during val step")
    
    parser.add_argument("--val_include_last", default=False, action='store_true',
                        help="Flag to include last batch of validation split each epoch during val step")

    parser.add_argument("--save_top_X", default=-1, type=int,
                        help="How many checkpoints to save, default=-1 (all)")

    parser.add_argument("--notebook_mode", default=False, action='store_true',
                        help="Basically RICH doesn't give us progress bars in notebooks..")
        
    parser.add_argument("--dont_limit_index_sort", default=False, action='store_true',
                        help="If flagged -> will not sort the limit indices (not recommended for large files)")
    
    parser.add_argument("--shuffle_by_sample", default=False, action='store_true',
                        help="If flagged -> will sort indices of dataset per-sample (equal n. A,B,C -> equal n. A,B,C)")

    main(**parser.parse_args().__dict__)
