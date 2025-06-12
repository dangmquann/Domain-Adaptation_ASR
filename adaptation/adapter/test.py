
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import shutil
import logging
import argparse
from hyperpyyaml import load_hyperpyyaml


parser = argparse.ArgumentParser(
        description="Train a SpeechGPT experiment",
    )

parser.add_argument(
    "config",
    type=str,
    help="A yaml-formatted file using the extended YAML syntax. "
    "defined by SpeechGPT.",
)

parser.add_argument( 
    "--flash",
    action='store_true',
    help="Enables or disables flash scaled dot product attention.",
)

parser.add_argument(
    "--compile",
    action='store_true',
    help="Enables or disables compile torch 2.0.",
)

args = parser.parse_args()

if args.flash:
    torch.backends.cuda.enable_flash_sdp(True)

with open(args.config) as fin:
    modules = load_hyperpyyaml(fin)
    trainer = modules['trainer']
    model = modules['model']
    if args.compile:
        model = torch.compile(model)
    ckpt_path = modules['resume_path']
    test_dataloader = modules['datamodule'].test_dataloader()
    trainer.test(
        model,
        ckpt_path=ckpt_path,
        dataloaders=test_dataloader,
        verbose=True,
    )
