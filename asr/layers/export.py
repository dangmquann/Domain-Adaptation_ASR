from .zipformer_adapter import Zipformer2, AdapterModule
import torch
import torch.nn as nn
import onnx
import k2
from .decoder import Decoder
from torch import Tensor
import logging
from typing import List, Optional, Tuple, Union, Dict


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


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
    
