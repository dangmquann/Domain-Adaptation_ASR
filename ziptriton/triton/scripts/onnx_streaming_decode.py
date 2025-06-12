#!/usr/bin/env python3
# Copyright 2022 Xiaomi Corporation (Authors: Wei Kang, Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
./pruned_transducer_stateless7_streaming/streaming_decode.py \
  --epoch 28 \
  --avg 15 \
  --decode-chunk-len 32 \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --decoding-method greedy_search \
  --num-decode-streams 2000
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import onnxruntime as ort
import k2
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet
from beam_search import Hypothesis, HypothesisList, get_hyps_shape
from torch.nn.utils.rnn import pad_sequence
from train import  get_params
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
import warnings
LOG_EPS = math.log(1e-10)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--encoder-model-filename",
        type=str,
        required=True,
        help="Path to the encoder onnx model. ",
    )

    parser.add_argument(
        "--decoder-model-filename",
        type=str,
        required=True,
        help="Path to the decoder onnx model. ",
    )

    parser.add_argument(
        "--joiner-model-filename",
        type=str,
        required=True,
        help="Path to the joiner onnx model. ",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Supported decoding methods are:
        greedy_search
        modified_beam_search
        fast_beam_search
        """,
    )

    parser.add_argument(
        "--num_active_paths",
        type=int,
        default=4,
        help="""An interger indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded parallel.",
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )
    return parser
def stack_states(states: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    stacked_state = []
    for i in range(len(states[0])):
        stacked_state.append(torch.cat([state[i] for state in states]))
    return stacked_state

def unstack_states(stacked_states: List[torch.Tensor]) -> List[List[torch.Tensor]]:
    unstacked_states = []
    batch_size = stacked_states[0].size(0)
    for i in range(batch_size):
        states = [state[i].unsqueeze(0) for state in stacked_states]
        unstacked_states.append(states)
    return unstacked_states

class DecodeStream(object):
    def __init__(
        self,
        samples,
        params: AttributeDict,
        cut_id: str,
        initial_states: List[torch.Tensor],
        decoding_graph: Optional[k2.Fsa] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Args:
          initial_states:
            Initial decode states of the model, e.g. the return value of
            `get_init_state` in conformer.py
          decoding_graph:
            Decoding graph used for decoding, may be a TrivialGraph or a HLG.
            Used only when decoding_method is fast_beam_search.
          device:
            The device to run this stream.
        """
        if params.decoding_method == "fast_beam_search":
            assert decoding_graph is not None
            assert device == decoding_graph.device
        self.samples = samples
        # self.samples = torch.cat([self.samples, torch.zeros(39*16000//100, dtype=torch.float32, device=device)])
        self.params = params
        self.cut_id = cut_id

        self.states = initial_states
        self.device = device
        self.sample_rate = 16000
        self.frames_length = 25
        self.frame_shift = 10
        self.frames_stride = 32
        self.min_seg = self.frames_length * self.sample_rate // 1000

        opts = FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.frame_length_ms = 25
        self.fbank = Fbank(opts)

        self.off_set = 240
        self.wav = torch.zeros(self.off_set, dtype=torch.float32, device=device)
        self.feature = torch.full((13,80), LOG_EPS, dtype=torch.float32, device=device)

        self.num_samples: int = len(samples)
        # how many frames have been processed. (before subsampling).
        # we only modify this value in `func:get_feature_frames`.
        # self.num_processed_frames: int = 0
        self.num_processed_samples: int = 0

        self._done: bool = False

        # The transcript of current utterance.
        self.ground_truth: str = ""

        # The decoding result (partial or final) of current utterance.
        self.hyp: List = []

        # how many frames have been processed, after subsampling (i.e. a
        # cumulative sum of the second return value of
        # encoder.streaming_forward
        self.done_frames: int = 0

        # It has two steps of feature subsampling in zipformer: out_lens=((x_lens-7)//2+1)//2
        # 1) feature embedding: out_lens=(x_lens-7)//2
        # 2) output subsampling: out_lens=(out_lens+1)//2
        self.pad_length = 13

        if params.decoding_method == "greedy_search":
            self.hyp = params.context_size * [params.blank_id]
        elif params.decoding_method == "modified_beam_search":
            self.hyps = HypothesisList()
            self.hyps.add(
                Hypothesis(
                    ys=[-1] * (params.context_size - 1) + [params.blank_id],
                    log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                )
            )
        elif params.decoding_method == "fast_beam_search":
            # The rnnt_decoding_stream for fast_beam_search.
            self.rnnt_decoding_stream: k2.RnntDecodingStream = k2.RnntDecodingStream(
                decoding_graph
            )
        else:
            raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    @property
    def done(self) -> bool:
        """Return True if all the features are processed."""
        return self._done

    @property
    def id(self) -> str:
        return self.cut_id

    def get_feature_frames(self, chunk_size: int) -> Tuple[torch.Tensor, int]:
        """Consume chunk_size frames of features"""
        chunk_length = chunk_size * 10 * self.sample_rate // 1000 
        chunk_sample = self.samples[self.num_processed_samples:self.num_processed_samples+chunk_length]
        
        self.wav = torch.cat([self.wav, chunk_sample])
        seg_wav = self.wav
        self.wav = self.wav[-240:]

        if len(seg_wav) < self.min_seg:
            tmp = torch.zeros(self.min_seg, dtype=torch.float32, device=self.device)
            tmp[0:len(seg_wav)] = seg_wav
            seg_wav = tmp

        feature = self.fbank(seg_wav)
        self.feature = torch.cat([self.feature, feature])
        ret_feature = self.feature
        self.feature = self.feature[-self.pad_length:]
        
        self.num_processed_samples += chunk_length
        if self.num_processed_samples >= self.num_samples:
            self._done = True

        return ret_feature, ret_feature.size(0)

    def decoding_result(self) -> List[int]:
        """Obtain current decoding result."""
        if self.params.decoding_method == "greedy_search":
            return self.hyp[self.params.context_size :]  # noqa
        elif self.params.decoding_method == "modified_beam_search":
            best_hyp = self.hyps.get_most_probable(length_norm=True)
            return best_hyp.ys[self.params.context_size:]  # noqa
        else:
            assert self.params.decoding_method == "fast_beam_search"
            return self.hyp

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

    def init_encoder_states(self, batch_size: int = 1):
        encoder_meta = self.encoder.get_modelmeta().custom_metadata_map
        # logging.info(f"encoder_meta={encoder_meta}")

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

        num_encoders = len(num_encoder_layers)
        cached_att = torch.zeros(batch_size,144640)
        cached_cnn = torch.zeros(batch_size,72704)
        embed_states = torch.zeros(batch_size, 128, 3, 19)
        processed_lens = torch.zeros(batch_size,1, dtype=torch.int64)
        states = [cached_att, cached_cnn, embed_states, processed_lens]
        self.num_encoders = num_encoders

        self.segment = T
        self.offset = decode_chunk_len
        return states

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
        states: List[torch.Tensor],
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        cached_att, cached_cnn, embed_states, processed_lens = states

        encoder_input = {"x": x.numpy()}
        encoder_output = ["encoder_out"]

        name = "cached_att"
        encoder_input[name] = cached_att.numpy()
        encoder_output.append(f"new_{name}")

        # CNN cached
        name = "cached_cnn"
        encoder_input[name] = cached_cnn.numpy()
        encoder_output.append(f"new_{name}")
        
        name = "embed_states"
        encoder_input[name] = embed_states.numpy()
        encoder_output.append(f"new_{name}")

        # (batch_size,)
        name = "processed_lens"
        encoder_input[name] = processed_lens.numpy()
        encoder_output.append(f"new_{name}")

        return encoder_input, encoder_output

    def run_encoder(self, x: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        Returns:
          Return a 3-D tensor of shape (N, T', joiner_dim) where
          T' is usually equal to ((T-7)//2-3)//2
        """
        encoder_input, encoder_output_names = self._build_encoder_input_output(x, states)

        out = self.encoder.run(encoder_output_names, encoder_input)
        encoder_out = torch.from_numpy(out[0])
        encoder_outlen = torch.stack([torch.tensor([e.shape[0]]) for e in encoder_out])
        return (encoder_out, encoder_outlen, [torch.from_numpy(i) for i in out[1:]])

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
    
def greedy_search(
    model: OnnxModel,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
) -> None:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      streams:
        A list of Stream objects.
    """
    assert len(streams) == encoder_out.size(0)
    assert encoder_out.ndim == 3

    blank_id = 0
    context_size = 2
    device = torch.device('cpu')
    T = encoder_out.size(1)

    decoder_input = torch.tensor(
        [stream.hyp[-context_size:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
   
    decoder_out = model.run_decoder(decoder_input)
    

    for t in range(T):
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        current_encoder_out = encoder_out[:, t : t + 1, :].squeeze(1) # noqa
        logits = model.run_joiner(
            current_encoder_out,
            decoder_out
        )
        # logits'shape (batch_size,  vocab_size)
        # logits = logits.squeeze(1).squeeze(1)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                streams[i].hyp.append(v)
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = torch.tensor(
                [stream.hyp[-context_size:] for stream in streams],
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.run_decoder(decoder_input)

def modified_beam_search(
    model: OnnxModel,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
    num_active_paths: int = 4,
    blank_penalty: float = 0.0,
) -> None:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The RNN-T model.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      streams:
        A list of stream objects.
      num_active_paths:
        Number of active paths during the beam search.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert len(streams) == encoder_out.size(0)

    blank_id = 0
    context_size = 2
    device = torch.device("cpu")
    batch_size = len(streams)
    T = encoder_out.size(1)

    B = [stream.hyps for stream in streams]

    for t in range(T):
        current_encoder_out = encoder_out[:, t]
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.stack(
            [hyp.log_prob.reshape(1) for hyps in A for hyp in hyps], dim=0
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        
        decoder_out = model.run_decoder(decoder_input)
        # decoder_out is of shape (num_hyps, decoder_output_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        ).squeeze(1).squeeze(1)  # (num_hyps, encoder_out_dim)

        # logits = model.joiner(current_encoder_out, decoder_out, project_input=False)
        logits = model.run_joiner(current_encoder_out, decoder_out)
        # logits is of shape (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0.0:
            logits[:, 0] -= blank_penalty

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(num_active_paths)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                if new_token != blank_id:
                    new_ys.append(new_token)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
                B[i].add(new_hyp)

    for i in range(batch_size):
        streams[i].hyps = B[i]

def decode_one_chunk(
    params: AttributeDict,
    model: OnnxModel,
    decode_streams: List[DecodeStream],
) -> List[int]:
    """Decode one chunk frames of features for each decode_streams and
    return the indexes of finished streams in a List.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      decode_streams:
        A List of DecodeStream, each belonging to a utterance.
    Returns:
      Return a List containing which DecodeStreams are finished.
    """
    device = "cpu"

    features = []
    feature_lens = []
    states = []
    processed_lens = []

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(32)
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    tail_length = 45
    if features.size(1) < tail_length:
        pad_length = tail_length - features.size(1)
        feature_lens += pad_length
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, pad_length),
            mode="constant",
            value=LOG_EPS,
        )

    states = stack_states(states)
    (encoder_out, encoder_out_lens, new_states) = model.run_encoder(
                                                        features,
                                                        states
                                                    )

    if params.decoding_method == "greedy_search":
        greedy_search(model=model, encoder_out=encoder_out, streams=decode_streams)
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            streams=decode_streams,
            encoder_out=encoder_out,
            num_active_paths=params.num_active_paths,
        )
    states = unstack_states(new_states)
    finished_streams = []
    for i in range(len(decode_streams)):
        decode_streams[i].states = states[i]
        decode_streams[i].done_frames += encoder_out_lens[i]
        if decode_streams[i].done:
            finished_streams.append(i)

    return finished_streams


def decode_dataset(
    cuts: CutSet,
    params: AttributeDict,
    model: OnnxModel,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      cuts:
        Lhotse Cutset containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    device = torch.device('cpu')

    log_interval = 50

    decode_results = []
    # Contain decode streams currently running.
    decode_streams = []
    for num, cut in enumerate(cuts):
        # each utterance has a DecodeStream.

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        assert audio.max() <= 1, "Should be normalized to [-1, 1])"

        samples = torch.from_numpy(audio).squeeze(0)

        initial_states = model.init_encoder_states()
        decode_stream = DecodeStream(
            samples=samples,
            params=params,
            cut_id=cut.id,
            initial_states=initial_states,
            decoding_graph=decoding_graph,
            device='cpu',
        )

        decode_stream.ground_truth = cut.supervisions[0].text

        decode_streams.append(decode_stream)

        while len(decode_streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                params=params, model=model, decode_streams=decode_streams
            )
            for i in sorted(finished_streams, reverse=True):
                decode_results.append(
                    (
                        decode_streams[i].id,
                        decode_streams[i].ground_truth.split(),
                        sp.decode(decode_streams[i].decoding_result()).split(),
                    )
                )
                del decode_streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    # decode final chunks of last sequences
    while len(decode_streams):
        finished_streams = decode_one_chunk(
            params=params, model=model, decode_streams=decode_streams
        )
        for i in sorted(finished_streams, reverse=True):
            decode_results.append(
                (
                    decode_streams[i].id,
                    decode_streams[i].ground_truth.split(),
                    sp.decode(decode_streams[i].decoding_result()).split(),
                )
            )
            del decode_streams[i]

    if params.decoding_method == "greedy_search":
        key = "greedy_search"
    elif params.decoding_method == "fast_beam_search":
        key = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}"
        )
    elif params.decoding_method == "modified_beam_search":
        key = f"num_active_paths_{params.num_active_paths}"
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")
    return {key: decode_results}


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.res_dir + f"/recogs-{test_set_name}" + params.suffix + ".txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir + f"/errs-{test_set_name}" + params.suffix + ".txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.res_dir + f"/wer-summary-{test_set_name}" + params.suffix+ ".txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    params.res_dir = f"{params.exp_dir}/streaming/onnx/{params.decoding_method}"

    # for streaming
    params.suffix = f"-streaming-chunk-{params.chunk_size}-left-"
    params.suffix += params.left_context_frames[:2]
    # for fast_beam_search
    if params.decoding_method == "fast_beam_search":
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    decoding_graph = None
    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    model = OnnxModel(
        encoder_model_filename=args.encoder_model_filename,
        decoder_model_filename=args.decoder_model_filename,
        joiner_model_filename=args.joiner_model_filename,
    )

    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.test_clean_cuts()
    test_other_cuts = librispeech.test_other_cuts()

    test_sets = ["test-clean", "test-other"]
    test_cuts = [test_clean_cuts, test_other_cuts]

    for test_set, test_cut in zip(test_sets, test_cuts):
        results_dict = decode_dataset(
            cuts=test_cut,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
