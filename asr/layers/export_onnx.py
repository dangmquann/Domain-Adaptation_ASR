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


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


class OnnxEncoder(nn.Module):
    """A wrapper for Zipformer and the encoder_proj from the joiner"""

    def __init__(
        self, encoder: Zipformer2, encoder_embed: nn.Module#, encoder_proj: nn.Linear
    ):
        """
        Args:
          encoder:
            A Zipformer encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        #self.encoder_proj = encoder_proj
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = 7 + 2 * 3

    def forward(
        self,
        x: torch.Tensor,
        cached_att: torch.Tensor,
        cached_cnn: torch.Tensor,
        cached_embed: torch.Tensor,
        processed_lens: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
        ]:
        """
        cached_att:
            The cached key tensors of the first attention modules.
            The cached value tensors of the first attention modules.
            The cached value tensors of the second attention modules.
        cached_cnn:
            The cached left contexts of the first convolution modules.
            The cached left contexts of the second convolution modules.
        cached_embed:
            The cached embeddings of the encoder.
        processed_lens: 
            The processed lengths of utterance.
        """
        N = x.size(0)
        T = self.chunk_size * 2 + self.pad_length
        x_lens = torch.tensor([T] * N, device=x.device)
        left_context_len = self.left_context_len

        x, x_lens, new_cached_embed_left_pad = self.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cached_left_pad=cached_embed,
        )
        assert x.size(1) == self.chunk_size, (x.size(1), self.chunk_size)

        src_key_padding_mask = torch.zeros(N, self.chunk_size, dtype=torch.bool)

        # processed_mask is used to mask out initial states
        processed_mask = torch.arange(left_context_len, device=x.device).expand(
            x.size(0), left_context_len
        )

        # (batch, left_context_size)
        processed_mask = (processed_lens <= processed_mask).flip(1)
        # Update processed lengths
        new_processed_lens = processed_lens + x_lens.unsqueeze(1)
        # (batch, left_context_size + chunk_size)
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        x = x.permute(1, 0, 2)
        
        encoder_state = []
        index_att = 0
        index_cnn = 0
        for i, module in enumerate(self.encoder.encoders):
            num_layers = module.num_layers
            embed_dim = self.encoder.encoder_dim[i]
            ds = self.encoder.downsampling_factor[i]
            num_heads = self.encoder.num_heads[i]
            key_dim = self.encoder.query_head_dim[i] * num_heads
            value_dim = self.encoder.value_head_dim[i] * num_heads
            downsample_left = self.encoder.left_context_frames[0] // ds
            nonlin_attn_head_dim = 3 * embed_dim // 4
            conv_left_pad = self.encoder.cnn_module_kernel[i] // 2 
            for layer in range(num_layers):
                cached_key = cached_att[:,index_att:index_att+downsample_left*key_dim].reshape(
                    N, downsample_left, key_dim
                ).transpose(0,1)
                logging.info(f"cached_key_{i}: {cached_key.shape}")
                index_att = index_att + downsample_left*key_dim

                cached_nonlin_attn = cached_att[:,index_att:index_att+downsample_left*nonlin_attn_head_dim].reshape(
                    N,1,downsample_left,nonlin_attn_head_dim
                ).transpose(0,1)
                logging.info(f"cached_nonlin_attn_{i}: {cached_nonlin_attn.shape}")
                index_att = index_att + downsample_left*nonlin_attn_head_dim

                cached_val1 = cached_att[:,index_att:index_att+downsample_left*value_dim].reshape(
                    N, downsample_left, value_dim
                ).transpose(0,1)
                logging.info(f"cached_val1_{i}: {cached_val1.shape}")
                index_att = index_att + downsample_left*value_dim

                cached_val2 = cached_att[:,index_att:index_att+downsample_left*value_dim].reshape(
                    N, downsample_left, value_dim
                ).transpose(0,1)
                logging.info(f"cached_val2_{i}: {cached_val2.shape}")
                index_att = index_att + downsample_left*value_dim

                cached_conv1 = cached_cnn[:,index_cnn:index_cnn+embed_dim*conv_left_pad].reshape(
                    N, embed_dim, conv_left_pad
                )
                logging.info(f"cached_conv1_{i}: {cached_conv1.shape}")
                index_cnn = index_cnn + embed_dim*conv_left_pad

                cached_conv2 = cached_cnn[:,index_cnn:index_cnn+embed_dim*conv_left_pad].reshape(
                    N, embed_dim, conv_left_pad
                )
                logging.info(f"cached_conv2_{i}: {cached_conv2.shape}")
                index_cnn = index_cnn + embed_dim*conv_left_pad

                encoder_state += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]

        (
            encoder_out,
            encoder_out_lens,
            new_encoder_states,
        ) = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=encoder_state,
            src_key_padding_mask=src_key_padding_mask,
        )
        encoder_out = encoder_out.permute(1, 0, 2)
        #encoder_out = self.encoder_proj(encoder_out)
        # Now encoder_out is of shape (N, T, jointer_dim)
        new_cached_att = []
        new_cached_cnn = []
        for i, tensor in enumerate(new_encoder_states):
            if (i % 6 == 0) or (i % 6 == 1) or (i % 6 == 2) or (i % 6 == 3):
                new_cached_att.append(tensor.transpose(0,1).reshape(N, -1))
            elif (i % 6 == 4) or (i % 6 == 5):
                new_cached_cnn.append(tensor.reshape(N, -1))

        new_cached_att = torch.cat(new_cached_att, dim=1)
        new_cached_cnn = torch.cat(new_cached_cnn, dim=1)            

        return (encoder_out, 
                new_cached_att, 
                new_cached_cnn, 
                new_cached_embed_left_pad, 
                new_processed_lens)

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
        
        initial_cached_att = []
        initial_cached_cnn = []
        for i, tensor in enumerate(states):
            if (i % 6 == 0) or (i % 6 == 1) or (i % 6 == 2) or (i % 6 == 3):
                initial_cached_att.append(tensor.transpose(0,1).reshape(batch_size, -1))
            elif (i % 6 == 4) or (i % 6 == 5):
                initial_cached_cnn.append(tensor.reshape(batch_size, -1))
        initial_cached_att = torch.cat(initial_cached_att, dim=1)
        initial_cached_cnn = torch.cat(initial_cached_cnn, dim=1)   
            
        embed_states = self.encoder_embed.get_init_states(batch_size, device)

        processed_lens = torch.zeros(batch_size,1, dtype=torch.int64, device=device)

        return initial_cached_att, initial_cached_cnn, embed_states, processed_lens

class OnnxDecoder(nn.Module):
    """A wrapper for Decoder and the decoder_proj from the joiner"""

    def __init__(self, decoder: Decoder, decoder_proj: nn.Linear):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, context_size).
        Returns
          Return a 2-D tensor of shape (N, jointer_dim)
        """
        need_pad = False
        decoder_output = self.decoder(y, need_pad=need_pad)
        decoder_output = decoder_output.squeeze(1)
        output = self.decoder_proj(decoder_output)

        return output


class OnnxJointer(nn.Module):
    """A wrapper for the joiner"""

    def __init__(self, output_linear: nn.Linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, jointer_dim)
          decoder_out:
            A 2-D tensor of shape (N, jointer_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        logit = encoder_out + decoder_out
        logit = self.output_linear(torch.tanh(logit))
        return logit
    
# ONNX wrapper for the adapter module
class OnnxAdapter(nn.Module):
    '''A wrapper for the adapter module and encoder_proj from the joiner'''
    def __init__(self, adapter_module: nn.Module, encoder_proj: nn.Linear):
        super(OnnxAdapter, self).__init__()
        self.adapter = adapter_module
        self.encoder_proj = encoder_proj

    def forward(self, x):
        return self.encoder_proj(self.adapter(x))


# Function to export the adapter model to ONNX.
def export_adapter_onnx(
    adapter_model: OnnxAdapter,
    adapter_filename: str,
    opset_version: int = 11,
) -> None:
    # Create a dummy input.
    # We assume the adapter expects a tensor of shape (batch_size, embed_dim)
    embed_dim = adapter_model.adapter.embed_dim
    T = 45
    dummy_input = torch.rand(1, T, embed_dim, dtype=torch.float32)

    # Prepare meta data
    meta_data = {
        "model_type": "adapter",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "ONNX exported adapter module",
        "input_shape": str(list(dummy_input.shape)),
        "embed_dim": str(embed_dim),
        "bottleneck_dim": str(adapter_model.adapter.bottleneck_dim),
    }
    logging.info(f"meta_data: {meta_data}")

    torch.onnx.export(
        adapter_model,
        dummy_input,
        adapter_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["adapter_out"],
        dynamic_axes={
            "encoder_out": {0: "N",1: "T", 2: "embed_dim"},
            "adapter_out": {0: "N"},
        },
    )

    add_meta_data(filename=adapter_filename, meta_data=meta_data)

def export_encoder_model_onnx(
    encoder_model: OnnxEncoder,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    encoder_model.encoder.__class__.forward = (
        encoder_model.encoder.__class__.streaming_forward
    )

    decode_chunk_len = encoder_model.chunk_size * 2
    # The encoder_embed subsample features (T - 7) // 2
    # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
    T = decode_chunk_len + encoder_model.pad_length

    x = torch.rand(1, T, 80, dtype=torch.float32)
    init_state = encoder_model.get_init_states()
    num_encoders = len(encoder_model.encoder.encoder_dim)
    logging.info(f"num_encoders: {num_encoders}")

    inputs = {}
    input_names = ["x"]

    outputs = {}
    output_names = ["encoder_out"]

    num_encoder_layers = ",".join(map(str, encoder_model.encoder.num_encoder_layers))
    encoder_dims = ",".join(map(str, encoder_model.encoder.encoder_dim))
    cnn_module_kernels = ",".join(map(str, encoder_model.encoder.cnn_module_kernel))
    ds = encoder_model.encoder.downsampling_factor
    left_context_len = encoder_model.left_context_len
    left_context_len = [left_context_len // k for k in ds]
    left_context_len = ",".join(map(str, left_context_len))
    query_head_dims = ",".join(map(str, encoder_model.encoder.query_head_dim))
    value_head_dims = ",".join(map(str, encoder_model.encoder.value_head_dim))
    num_heads = ",".join(map(str, encoder_model.encoder.num_heads))

    meta_data = {
        "model_type": "zipformer2",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "streaming zipformer2",
        "decode_chunk_len": str(decode_chunk_len),  # 32
        "T": str(T),  # 32+7+2*3=45
        "num_encoder_layers": num_encoder_layers,
        "encoder_dims": encoder_dims,
        "cnn_module_kernels": cnn_module_kernels,
        "left_context_len": left_context_len,
        "query_head_dims": query_head_dims,
        "value_head_dims": value_head_dims,
        "num_heads": num_heads,
    }
    logging.info(f"meta_data: {meta_data}")

     # attention cached
    cached_att = init_state[0]
    name = "cached_att"
    logging.info(f"{name}.shape: {cached_att.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # CNN cached
    cached_cnn = init_state[1]
    name = "cached_cnn"
    logging.info(f"{name}.shape: {cached_cnn.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (batch_size, channels, left_pad, freq)
    embed_states = init_state[-2]
    name = "embed_states"
    logging.info(f"{name}.shape: {embed_states.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    # (batch_size,)
    processed_lens = init_state[-1]
    name = "processed_lens"
    logging.info(f"{name}.shape: {processed_lens.shape}")
    inputs[name] = {0: "N"}
    outputs[f"new_{name}"] = {0: "N"}
    input_names.append(name)
    output_names.append(f"new_{name}")

    logging.info(inputs)
    logging.info(outputs)
    logging.info(input_names)
    logging.info(output_names)

    torch.onnx.export(
        encoder_model,
        (x, cached_att, cached_cnn, embed_states, processed_lens),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "x",
            "cached_att",
            "cached_cnn",
            "embed_states",
            "processed_lens",
        ],
        output_names=[
            "encoder_out",
            "new_cached_att",
            "new_cached_cnn",
            "new_embed_states",
            "new_processed_lens",
        ],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "encoder_out": {0: "N"},  
            **inputs,
            **outputs,
        },
    )

    add_meta_data(filename=encoder_filename, meta_data=meta_data)


def export_decoder_model_onnx(
    decoder_model: OnnxDecoder,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, decoder_model.context_size)

    and has one output:

        - decoder_out: a torch.float32 tensor of shape (N, jointer_dim)

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size

    y = torch.zeros(10, context_size, dtype=torch.int64)
    decoder_model = torch.jit.script(decoder_model)
    torch.onnx.export(
        decoder_model,
        y,
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )

    meta_data = {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    }
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def export_jointer_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported joiner model has two inputs:

        - encoder_out: a tensor of shape (N, jointer_dim)
        - decoder_out: a tensor of shape (N, jointer_dim)

    and produces one output:

        - logit: a tensor of shape (N, vocab_size)
    """
    jointer_dim = joiner_model.output_linear.weight.shape[1]
    logging.info(f"joiner dim: {jointer_dim}")

    projected_encoder_out = torch.rand(11, jointer_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(11, jointer_dim, dtype=torch.float32)

    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "encoder_out", #adapter_out
            "decoder_out",
        ],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"}, #adapter_out
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    meta_data = {
        "jointer_dim": str(jointer_dim),
    }
    add_meta_data(filename=joiner_filename, meta_data=meta_data)

class OnnxEncoderProj(nn.Module):

    def __init__(self, encoder_proj: nn.Linear):
        super(OnnxEncoderProj, self).__init__()
        self.encoder_proj = encoder_proj
    
    def forward(self, x):
        return self.encoder_proj(x)

def export_encoder_proj_onnx(
    encoder_proj: OnnxEncoderProj,
    encoder_proj_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the encoder_proj model to ONNX format.

    The exported model has one input:

        - encoder_out: a torch.float32 tensor of shape (N, jointer_dim)

    and has one output:

        - adapter_out: a torch.float32 tensor of shape (N, jointer_dim)

    Args:
      encoder_proj:
        The encoder_proj model to be exported.
      encoder_proj_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    jointer_dim = encoder_proj.encoder_proj.weight.shape[1]
    logging.info(f"jointer dim: {jointer_dim}")
    T = 45
    encoder_out = torch.rand(11, T,jointer_dim, dtype=torch.float32)

    torch.onnx.export(
        encoder_proj,
        encoder_out,
        encoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["adapter_out"],
        dynamic_axes={
            "encoder_out": {0: "N", 1: "T", 2: "embed_dim"},
            "adapter_out": {0: "N"},
        },
    )
    meta_data = {
        "jointer_dim": str(jointer_dim),
    }
    add_meta_data(filename=encoder_proj_filename, meta_data=meta_data)