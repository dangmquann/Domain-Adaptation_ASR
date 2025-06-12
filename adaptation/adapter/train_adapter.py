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
    description="Train a adapter experiment",
)

parser.add_argument(
    "config",
    type=str,
    help="A yaml-formatted file using the extended YAML syntax.",
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

parser.add_argument(
    "--use_lora",
    action='store_true',
    help="Enable LoRA fine-tuning (only LoRA params trainable).",
)

parser.add_argument(
    "--use_adapter",
    action='store_true',
    help="Enable Adapter fine-tuning (only Adapter params trainable).",
)

args = parser.parse_args()

if args.flash:
    torch.backends.cuda.enable_flash_sdp(True)

if args.use_lora and args.use_adapter:
    raise ValueError("Choose only one: --use_lora or --use_adapter.")

with open(args.config) as fin:
    modules = load_hyperpyyaml(fin)
    trainer = modules['trainer']
    model = modules['model']

    print(model)
    for name, param in model.named_parameters():
        print(name, param.size())

    # Đóng băng/thaw tham số theo lựa chọn
    if args.use_lora:
        num_param = sum(p.numel() for p in model.parameters())
        num_trainable = 0
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
                num_trainable += p.numel()
            else:
                p.requires_grad = False
        print(f"A total of {num_trainable} trainable parameters ({num_trainable / num_param * 100:.3f}% of the whole model) [LoRA]")
    elif args.use_adapter:
        num_param = sum(p.numel() for p in model.parameters())
        num_trainable = 0
        for name, p in model.named_parameters():
            if "adapter" in name:
                p.requires_grad = True
                num_trainable += p.numel()
            else:
                p.requires_grad = False
        print(f"A total of {num_trainable} trainable parameters ({num_trainable / num_param * 100:.3f}% of the whole model) [Adapter]")
    else:
        print("Không sử dụng LoRA hoặc Adapter, tất cả tham số sẽ được train.")

    if args.compile:
        model = torch.compile(model)

    os.makedirs(trainer.log_dir, exist_ok=True)
    shutil.copy(args.config, trainer.log_dir)
    datamodule = modules['datamodule']

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=modules['resume_path'],
    )
    test_dataloader = modules['datamodule'].test_dataloader()
    trainer.test(
        dataloaders=test_dataloader,
        verbose=True,
    )