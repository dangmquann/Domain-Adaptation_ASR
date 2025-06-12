import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import os
import shutil
import logging
import argparse
from hyperpyyaml import load_hyperpyyaml
from humming.layers.export_onnx import make_pad_mask


class EncoderModel(nn.Module):
    """A wrapper for encoder and encoder_embed"""

    def __init__(self, encoder: nn.Module, encoder_embed: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed

    def forward(
        self, features: Tensor, feature_lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: (N, T, C)
            feature_lengths: (N,)
        """
        x, x_lens = self.encoder_embed(features, feature_lengths)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        return encoder_out, encoder_out_lens


class StreamingEncoderModel(nn.Module):
    """A wrapper for encoder and encoder_embed"""

    def __init__(self, encoder: nn.Module, encoder_embed: nn.Module) -> None:
        super().__init__()
        assert len(encoder.chunk_size) == 1, encoder.chunk_size
        assert len(encoder.left_context_frames) == 1, encoder.left_context_frames
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]

        # The encoder_embed subsample features (T - 7) // 2
        # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
        self.pad_length = 7 + 2 * 3

        self.encoder = encoder
        self.encoder_embed = encoder_embed

    def forward(
        self, features: Tensor, feature_lengths: Tensor, states: List[Tensor]
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Streaming forward for encoder_embed and encoder.

        Args:
            features: (N, T, C)
            feature_lengths: (N,)
            states: a list of Tensors

        Returns encoder outputs, output lengths, and updated states.
        """
        chunk_size = self.chunk_size
        left_context_len = self.left_context_len

        cached_embed_left_pad = states[-2]
        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=features,
            x_lens=feature_lengths,
            cached_left_pad=cached_embed_left_pad,
        )
        assert x.size(1) == chunk_size, (x.size(1), chunk_size)

        src_key_padding_mask = make_pad_mask(x_lens)

        # processed_mask is used to mask out initial states
        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )
        processed_lens = states[-1]  # (batch,)
        # (batch, left_context_size)
        processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
        # Update processed lengths
        new_processed_lens = processed_lens + x_lens

        # (batch, left_context_size + chunk_size)
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        encoder_states = states[:-2]

        (
            encoder_out,
            encoder_out_lens,
            new_encoder_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_states,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        new_states = new_encoder_states + [
            new_cached_embed_left_pad,
            new_processed_lens,
        ]
        return encoder_out, encoder_out_lens, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[torch.Tensor]:
        """
        Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        states[-2] is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs)
        states[-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
        """
        states = self.encoder.get_init_states(batch_size, device)

        embed_states = self.encoder_embed.get_init_states(batch_size, device)
        states.append(embed_states)

        processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        states.append(processed_lens)

        return states
    
def main():
    parser = argparse.ArgumentParser(
        description="Export a trained RNNT (Zipformer) model to ONNX."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Đường dẫn file YAML config, sử dụng định dạng extended YAML (hyperpyyaml)."
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Bật/tắt flash scaled dot product attention (nếu GPU hỗ trợ)."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Bật/tắt torch.compile trong PyTorch 2.0."
    )

    args = parser.parse_args()

    # Tuỳ chọn flash attention
    if args.flash:
        torch.backends.cuda.enable_flash_sdp(True)

    # Đọc config
    with open(args.config) as fin:
        modules = load_hyperpyyaml(fin)

    trainer = modules["trainer"]
    model = modules["model"]

    # Tuỳ chọn compile (PyTorch 2.0)
    if args.compile:
        model = torch.compile(model)

    os.makedirs(trainer.log_dir, exist_ok=True)
    shutil.copy(args.config, trainer.log_dir)

    # Có thể khởi tạo datamodule nếu bạn cần (tùy yêu cầu):
    # datamodule = modules['datamodule']

    export_config = modules["export_config"]

    if export_config["enable_export"]:
        logging.info("Starting torch jit script export process ...")
        print("Starting torch jit script export process ...")

        export_dir = export_config.get(
            "export_dir",
            os.path.join(trainer.log_dir, "export")
        )
        os.makedirs(export_dir, exist_ok=True)

        # Tải checkpoint
        checkpoint_path = modules["resume_path"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        model.eval()


        if export_config["jit"]:
            if export_config["casual"]:
                model.encoder = StreamingEncoderModel(model.encoder, model.encoder_embed)
                chunk_size = model.encoder.chunk_size
                left_context_len = model.encoder.left_context_len
                filename = f"jit_script_chunk_{chunk_size}_left_{left_context_len}.pt"
            else:
                model.encoder = EncoderModel(model.encoder, model.encoder_embed)
                filename = "jit_script.pt"

            logging.info("Using torch.jit.script")
            model = torch.jit.script(model)
            model.save(os.path.join(export_dir, filename))
            logging.info(f"Saved to {filename}")
        else:
            logging.info("Not using torchscript. Export model.state_dict() instead.")
            filename = "model_state_dict.pt"
            torch.save(model.state_dict(), os.path.join(export_dir, filename))
            logging.info(f"Saved to {filename}")

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
            



