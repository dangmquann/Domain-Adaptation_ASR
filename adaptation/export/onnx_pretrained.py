import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import k2
import kaldifeat
from kaldifeat import FbankOptions, OnlineFbank, OnlineFeature

import onnxruntime as ort
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from hyperpyyaml import load_hyperpyyaml


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = load_hyperpyyaml(f)
    return config


class OnnxModel:
    def __init__(
        self,
        encoder_model_filename: str,
        decoder_model_filename: str,
        joiner_model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.init_encoder(encoder_model_filename)
        self.init_decoder(decoder_model_filename)
        self.init_joiner(joiner_model_filename)

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        self.init_encoder_states()

    def init_encoder_states(self, batch_size: int = 1):
        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        logging.info(f"encoder_meta={encoder_meta}")

        model_type = encoder_meta["model_type"]
        assert model_type == "zipformer2", model_type

        decode_chunk_len = int(encoder_meta["decode_chunk_len"])
        T = int(encoder_meta["T"])

        num_encoder_layers = encoder_meta["num_encoder_layers"]
        encoder_dims = encoder_meta["encoder_dims"]
        cnn_module_kernels = encoder_meta["cnn_module_kernels"]
        left_context_len = encoder_meta["left_context_len"]
        query_head_dims = encoder_meta["query_head_dims"]
        value_head_dims = encoder_meta["value_head_dims"]
        num_heads = encoder_meta["num_heads"]

        def to_int_list(s):
            return list(map(int, s.split(",")))

        num_encoder_layers = to_int_list(num_encoder_layers)
        encoder_dims = to_int_list(encoder_dims)
        cnn_module_kernels = to_int_list(cnn_module_kernels)
        left_context_len = to_int_list(left_context_len)
        query_head_dims = to_int_list(query_head_dims)
        value_head_dims = to_int_list(value_head_dims)
        num_heads = to_int_list(num_heads)

        logging.info(f"decode_chunk_len: {decode_chunk_len}")
        logging.info(f"T: {T}")
        logging.info(f"num_encoder_layers: {num_encoder_layers}")
        logging.info(f"encoder_dims: {encoder_dims}")
        logging.info(f"cnn_module_kernels: {cnn_module_kernels}")
        logging.info(f"left_context_len: {left_context_len}")
        logging.info(f"query_head_dims: {query_head_dims}")
        logging.info(f"value_head_dims: {value_head_dims}")
        logging.info(f"num_heads: {num_heads}")

        num_encoders = len(num_encoder_layers)

        self.states = []
        for i in range(num_encoders):
            num_layers = num_encoder_layers[i]
            key_dim = query_head_dims[i] * num_heads[i]
            embed_dim = encoder_dims[i]
            nonlin_attn_head_dim = 3 * embed_dim // 4
            value_dim = value_head_dims[i] * num_heads[i]
            conv_left_pad = cnn_module_kernels[i] // 2

            for layer in range(num_layers):
                cached_key = torch.zeros(
                    left_context_len[i], batch_size, key_dim
                ).numpy()
                cached_nonlin_attn = torch.zeros(
                    1, batch_size, left_context_len[i], nonlin_attn_head_dim
                ).numpy()
                cached_val1 = torch.zeros(
                    left_context_len[i], batch_size, value_dim
                ).numpy()
                cached_val2 = torch.zeros(
                    left_context_len[i], batch_size, value_dim
                ).numpy()
                cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).numpy()
                cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).numpy()
                self.states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]
        embed_states = torch.zeros(batch_size, 128, 3, 19).numpy()
        self.states.append(embed_states)
        processed_lens = torch.zeros(batch_size, dtype=torch.int64).numpy()
        self.states.append(processed_lens)

        self.num_encoders = num_encoders

        self.segment = T
        self.offset = decode_chunk_len

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

        logging.info(f"context_size: {self.context_size}")
        logging.info(f"vocab_size: {self.vocab_size}")

    def init_joiner(self, joiner_model_filename: str):
        self.joiner = ort.InferenceSession(
            joiner_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        joiner_meta = self.joiner.get_modelmeta().custom_metadata_map
        self.joiner_dim = int(joiner_meta["joiner_dim"])

        logging.info(f"joiner_dim: {self.joiner_dim}")

    def _build_encoder_input_output(
        self,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        encoder_input = {"x": x.numpy()}
        encoder_output = ["encoder_out"]

        def build_inputs_outputs(tensors, i):
            assert len(tensors) == 6, len(tensors)

            # (downsample_left, batch_size, key_dim)
            name = f"cached_key_{i}"
            encoder_input[name] = tensors[0]
            encoder_output.append(f"new_{name}")

            # (1, batch_size, downsample_left, nonlin_attn_head_dim)
            name = f"cached_nonlin_attn_{i}"
            encoder_input[name] = tensors[1]
            encoder_output.append(f"new_{name}")

            # (downsample_left, batch_size, value_dim)
            name = f"cached_val1_{i}"
            encoder_input[name] = tensors[2]
            encoder_output.append(f"new_{name}")

            # (downsample_left, batch_size, value_dim)
            name = f"cached_val2_{i}"
            encoder_input[name] = tensors[3]
            encoder_output.append(f"new_{name}")

            # (batch_size, embed_dim, conv_left_pad)
            name = f"cached_conv1_{i}"
            encoder_input[name] = tensors[4]
            encoder_output.append(f"new_{name}")

            # (batch_size, embed_dim, conv_left_pad)
            name = f"cached_conv2_{i}"
            encoder_input[name] = tensors[5]
            encoder_output.append(f"new_{name}")

        for i in range(len(self.states[:-2]) // 6):
            build_inputs_outputs(self.states[i * 6 : (i + 1) * 6], i)

        # (batch_size, channels, left_pad, freq)
        name = "embed_states"
        embed_states = self.states[-2]
        encoder_input[name] = embed_states
        encoder_output.append(f"new_{name}")

        # (batch_size,)
        name = "processed_lens"
        processed_lens = self.states[-1]
        encoder_input[name] = processed_lens
        encoder_output.append(f"new_{name}")

        return encoder_input, encoder_output

    def _update_states(self, states: List[np.ndarray]):
        self.states = states

    def run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        Returns:
          Return a 3-D tensor of shape (N, T', joiner_dim) where
          T' is usually equal to ((T-7)//2-3)//2
        """
        encoder_input, encoder_output_names = self._build_encoder_input_output(x)

        out = self.encoder.run(encoder_output_names, encoder_input)

        self._update_states(out[1:])

        return torch.from_numpy(out[0])

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          decoder_input:
            A 2-D tensor of shape (N, context_size)
        Returns:
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]

        return torch.from_numpy(out)

    def run_joiner(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, joiner_dim)
          decoder_out:
            A 2-D tensor of shape (N, joiner_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]

        return torch.from_numpy(out)


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0].contiguous())
    return ans


def create_streaming_feature_extractor() -> OnlineFeature:
    """Create a CPU streaming feature extractor.

    At present, we assume it returns a fbank feature extractor with
    fixed options. In the future, we will support passing in the options
    from outside.

    Returns:
      Return a CPU streaming feature extractor.
    """
    opts = FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    return OnlineFbank(opts)


def greedy_search(
    model: OnnxModel,
    encoder_out: torch.Tensor,
    context_size: int,
    decoder_out: Optional[torch.Tensor] = None,
    hyp: Optional[List[int]] = None,
) -> List[int]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        A 3-D tensor of shape (1, T, joiner_dim)
      context_size:
        The context size of the decoder model.
      decoder_out:
        Optional. Decoder output of the previous chunk.
      hyp:
        Decoding results for previous chunks.
    Returns:
      Return the decoded results so far.
    """

    blank_id = 0

    if decoder_out is None:
        assert hyp is None, hyp
        hyp = [blank_id] * context_size
        decoder_input = torch.tensor([hyp], dtype=torch.int64)
        decoder_out = model.run_decoder(decoder_input)
    else:
        assert hyp is not None, hyp

    encoder_out = encoder_out.squeeze(0)
    T = encoder_out.size(0)
    for t in range(T):
        cur_encoder_out = encoder_out[t : t + 1]
        joiner_out = model.run_joiner(cur_encoder_out, decoder_out).squeeze(0)
        y = joiner_out.argmax(dim=0).item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = hyp[-context_size:]
            decoder_input = torch.tensor([decoder_input], dtype=torch.int64)
            decoder_out = model.run_decoder(decoder_input)

    return hyp, decoder_out



def token_ids_to_words(token_ids: List[int], token_table: k2.SymbolTable) -> str:
    """
    Convert a list of token IDs to words using the token table.

    Args:
        token_ids: List of token IDs.
        token_table: The symbol table mapping token IDs to tokens.

    Returns:
        A string representing the decoded words.
    """
    text = ""
    for i in token_ids:
        text += token_table[i]
    return text.replace("▁", " ").strip()


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Tải cấu hình từ file YAML
    config = load_config('decode_config.yaml')  # Thay 'decode_config.yaml' bằng đường dẫn thực tế

    logging.info("Loaded configuration:")
    logging.info(config)

    # Khởi tạo mô hình ONNX
    model = OnnxModel(
        encoder_model_filename=config["encoder_model_filename"],
        decoder_model_filename=config["decoder_model_filename"],
        jointer_model_filename=config["jointer_model_filename"],
    )

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = config["sample_rate"]
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {config['sound_files']}")
    waves = read_sound_files(
        filenames=config["sound_files"],
        expected_sample_rate=config["sample_rate"],
    )

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=math.log(1e-10),
    )

    feature_lengths = torch.tensor(feature_lengths, dtype=torch.int64)
    encoder_out, encoder_out_lens = model.run_encoder(features, feature_lengths)

    hyps = greedy_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
    )

    s = "\n"

    token_table = k2.SymbolTable.from_file(config["tokens"])

    for filename, hyp in zip(config["sound_files"], hyps):
        words = token_ids_to_words(hyp, token_table)
        s += f"{filename}:\n{words}\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()

    # how to run this script:
