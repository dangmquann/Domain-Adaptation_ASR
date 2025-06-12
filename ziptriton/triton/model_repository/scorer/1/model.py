# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import torch
import sentencepiece as spm
from typing import Union, List
import time
from icefall.lexicon import Lexicon
from icefall import ContextGraph, ContextState, NgramLm, NgramLmStateCost
import k2
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import warnings
from loguru import logger
import sys
import os

# Loại bỏ tất cả các handler mặc định
logger.remove()

# Thêm handler in ra console
logger.add(sys.stdout, colorize=True, enqueue=True, level="DEBUG")

# Thêm handler ghi log vào file, ví dụ: lưu tại /logs/triton_model.log
# Bạn có thể thay đổi đường dẫn, kích thước xoay file, hoặc mức lưu log tùy ý.
log_file_path = os.environ.get("TRITON_LOG_FILE", "/data/quandm8/ziptriton/triton/log_triton.log")
logger.add(log_file_path, rotation="10 MB", compression="zip", enqueue=True, level="DEBUG")
# --- KẾT THÚC CẤU HÌNH LOGGER ---
@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # the lm score for next token given the current ys
    lm_score: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # N-gram LM state
    state_cost: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


def get_hyps_shape(hyps: List[HypothesisList]) -> k2.RaggedShape:
    """Return a ragged shape with axes [utt][num_hyps].

    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return a ragged shape with 2 axes [utt][num_hyps]. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    ans = k2.ragged.create_ragged_shape2(
        row_splits=row_splits, cached_tot_size=row_splits[-1].item()
    )
    return ans


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        logger.info("Initialize TritonPythonModel with args: {}".format(args))
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)
        self.enable_adapter = (self.model_config['parameters'].get('enable_adapter', {}).get('string_value', 'True') == 'True')

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Get INPUT configuration
        if "GPU" in args['model_instance_kind']:
            self.device = f"cuda:{args['model_instance_device_id']}"
        else:
            self.device = "cpu"
        logger.info("Device: {}".format(self.device))

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out") # adapter_out
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.encoder_dim = encoder_config['dims'][-1]
        logger.info("Encoder dim: {}".format(self.encoder_dim))
        
        self.init_sentence_piece(self.model_config['parameters'])

        self.decoding_method = self.model_config['parameters']['decoding_method']
        logger.info("Decoding method: {}".format(self.decoding_method))

        # parameters for fast beam search
        if self.decoding_method == 'fast_beam_search':
            self.temperature = float(self.model_config['parameters']['temperature'])

            self.beam = int(self.model_config['parameters']['beam'])
            self.max_contexts = int(self.model_config['parameters']['max_contexts'])
            self.max_states = int(self.model_config['parameters']['max_states'])
            
            self.fast_beam_config = k2.RnntDecodingConfig(
                vocab_size=self.vocab_size,
                decoder_history_len=self.context_size,
                beam=self.beam,
                max_contexts=self.max_contexts,
                max_states=self.max_states,
            )

            self.decoding_graph = k2.trivial_graph(
                    self.vocab_size - 1, device=self.device
            )
        
        if self.decoding_method == 'modified_beam_search':
            self.num_active_paths = int(self.model_config['parameters']['num_active_paths'])
        # use to record every sequence state
        self.seq_states = {}
        print("Finish Init")

    def init_sentence_piece(self, parameters):
        for key,value in parameters.items():
            parameters[key] = value["string_value"]
        self.context_size = int(parameters['context_size'])
        if 'bpe' in parameters['tokenizer_file']:
            sp = spm.SentencePieceProcessor()
            sp.load(parameters['tokenizer_file'])
            self.blank_id = sp.piece_to_id("<blk>")
            self.unk_id = sp.piece_to_id("<unk>")
            self.vocab_size = sp.get_piece_size()
            self.tokenizer = sp
        else:
            assert 'char' in parameters['tokenizer_file']
            lexicon = Lexicon(parameters['tokenizer_file'])
            self.unk_id = lexicon.token_table["<unk>"]
            self.blank_id = lexicon.token_table["<blk>"]
            self.vocab_size = max(lexicon.tokens) + 1
            self.tokenizer = lexicon

    def forward_adapter(self, encoder_out):
        logger.debug("Running adapter inference")
        
        in_adapter_tensor = pb_utils.Tensor("encoder_out", encoder_out.as_numpy())
        
        inference_request = pb_utils.InferenceRequest(
            model_name='adapter',
            requested_output_names=['adapter_out'],
            inputs=[in_adapter_tensor]
        )
        
        inference_response = inference_request.exec()
        if inference_response.has_error():
            error_msg = inference_response.error().message()
            logger.error("Adapter inference error: {}", error_msg)
            raise pb_utils.TritonModelException(error_msg)
        
        adapter_out_tensor = pb_utils.get_output_tensor_by_name(inference_response, 'adapter_out')
        logger.info("Adapter inference done")
        return adapter_out_tensor


    

    def forward_joiner(self, cur_encoder_out, decoder_out):
        logger.debug("Running joiner inference")
        in_joiner_tensor_0 = pb_utils.Tensor("encoder_out", cur_encoder_out.cpu().numpy()) #encoder_out
        in_joiner_tensor_1 = pb_utils.Tensor("decoder_out", decoder_out.cpu().numpy())

        inference_request = pb_utils.InferenceRequest(
            model_name='joiner',
            requested_output_names=['logit'],
            inputs=[in_joiner_tensor_0, in_joiner_tensor_1])
        inference_response = inference_request.exec()
        if inference_response.has_error():
            error_msg = inference_response.error().message()
            logger.error("Joiner inference error: {}", error_msg)
            raise pb_utils.TritonModelException(inference_response.error().message())
        
        # Extract the output tensors from the inference response.
        logits = pb_utils.get_output_tensor_by_name(inference_response,'logit')
        logits = torch.utils.dlpack.from_dlpack(logits.to_dlpack()).cpu()
        assert len(logits.shape) == 2, logits.shape
        logger.info("Joiner inference done")
        return logits


    def forward_decoder(self,decoder_input):
        logger.debug("Running decoder inference")
        if self.decoding_method == 'greedy_search':
            decoder_input = np.asarray(decoder_input, dtype=np.int64)
        if self.decoding_method == 'fast_beam_search':
            decoder_input = decoder_input.cpu().numpy()
        in_decoder_input_tensor = pb_utils.Tensor("y", decoder_input)

        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['decoder_out'],
            inputs=[in_decoder_input_tensor])

        inference_response = inference_request.exec()
        if inference_response.has_error():
            error_msg = inference_response.error().message()
            logger.error("Decoder inference error: {}", error_msg)
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            decoder_out = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'decoder_out')
            decoder_out = torch.utils.dlpack.from_dlpack(decoder_out.to_dlpack()).cpu()
            logger.debug("Decoder inference completed")
            return decoder_out

    def greedy_search(self, encoder_out, hyps_list):
        logger.info("Starting greedy search decoding")
        emitted = False
        # add blank_id as prefix
        hyps_list = [[self.blank_id] * self.context_size + h for h in hyps_list]
        contexts = [h[-self.context_size:] for h in hyps_list]
        decoder_out = self.forward_decoder(contexts)
        assert encoder_out.shape[0] == decoder_out.shape[0]

        for t in range(encoder_out.shape[1]):
            if emitted:
                contexts = [h[-self.context_size:] for h in hyps_list]
                decoder_out = self.forward_decoder(contexts)

            cur_encoder_out = encoder_out[:,t]
            logits = self.forward_joiner(cur_encoder_out, decoder_out)

            assert logits.ndim == 2, logits.shape
            y = logits.argmax(dim=1).tolist()
            for i, v in enumerate(y):
                if v not in (self.blank_id, self.unk_id):
                    hyps_list[i].append(v)
                    emitted = True
        # remove prefix blank_id
        hyps_list = [h[self.context_size:] for h in hyps_list]
        logger.info("Greedy search decoding finished")
        return hyps_list

    # From k2 utils.py
    def get_texts(self, 
        best_paths: k2.Fsa, return_ragged: bool = False
    ) -> Union[List[List[int]], k2.RaggedTensor]:
        """Extract the texts (as word IDs) from the best-path FSAs.
        Args:
          best_paths:
            A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
            containing multiple FSAs, which is expected to be the result
            of k2.shortest_path (otherwise the returned values won't
            be meaningful).
          return_ragged:
            True to return a ragged tensor with two axes [utt][word_id].
            False to return a list-of-list word IDs.
        Returns:
          Returns a list of lists of int, containing the label sequences we
          decoded.
        """
        logger.debug("Extracting texts from best paths")
        if isinstance(best_paths.aux_labels, k2.RaggedTensor):
            # remove 0's and -1's.
            aux_labels = best_paths.aux_labels.remove_values_leq(0)
            # TODO: change arcs.shape() to arcs.shape
            aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)

            # remove the states and arcs axes.
            aux_shape = aux_shape.remove_axis(1)
            aux_shape = aux_shape.remove_axis(1)
            aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
        else:
            # remove axis corresponding to states.
            aux_shape = best_paths.arcs.shape().remove_axis(1)
            aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
            # remove 0's and -1's.
            aux_labels = aux_labels.remove_values_leq(0)

        assert aux_labels.num_axes == 2
        logger.debug("Text extraction complete")
        if return_ragged:
            return aux_labels
        else:
            return aux_labels.tolist()

    def fast_beam_search(self, encoder_out, states_list):
        logger.info("Starting fast beam search decoding")
        streams_list = [state[0] for state in states_list]
        processed_lens_list = [state[1] for state in states_list]

        decoding_streams = k2.RnntDecodingStreams(streams_list, self.fast_beam_config)
        encoder_out = encoder_out.to(self.device)
        for t in range(encoder_out.shape[1]):
            shape, contexts = decoding_streams.get_contexts()
            contexts = contexts.to(torch.int64)

            decoder_out = self.forward_decoder(contexts)

            cur_encoder_out = torch.index_select(
                encoder_out[:, t, :], 0, shape.row_ids(1).to(torch.int64)
            )

            logits = self.forward_joiner(cur_encoder_out,
                decoder_out)

            logits = logits.squeeze(1).squeeze(1).float()
            log_probs = (logits / self.temperature).log_softmax(dim=-1)
            decoding_streams.advance(log_probs.to(self.device))
        decoding_streams.terminate_and_flush_to_streams()
        lattice = decoding_streams.format_output(processed_lens_list)

        best_path = k2.shortest_path(lattice, use_double_scores=True)
        hyps_list = self.get_texts(best_path)
        logger.info("Fast beam search decoding finished")
        return hyps_list

    def modified_beam_search(self, encoder_out, states_list, blank_penalty=0):
        logger.info("Starting modified beam search decoding")
        T = encoder_out.size(1)
        batch_size = encoder_out.size(0)
        for t in range(T):
            current_encoder_out = encoder_out[:, t].to(self.device) # current_encoder_out's shape: (batch_size, encoder_out_dim)
            
            hyps_shape = get_hyps_shape(states_list).to(self.device)

            A = [list(b) for b in states_list]
            states_list = [HypothesisList() for _ in range(batch_size)]

            ys_log_probs = torch.stack(
                [hyp.log_prob.reshape(1) for hyps in A for hyp in hyps], dim=0
            )  # (num_hyps, 1)

            decoder_input = torch.tensor(
                [hyp.ys[-self.context_size:] for hyps in A for hyp in hyps],
                device=self.device,
                dtype=torch.int64,
            )  # (num_hyps, context_size)

            decoder_out = self.forward_decoder(decoder_input.cpu().numpy()) # decoder_out is of shape (num_hyps, decoder_output_dim)

            current_encoder_out = torch.index_select(
                current_encoder_out,
                dim=0,
                index=hyps_shape.row_ids(1).to(torch.int64),
            ) # (num_hyps, encoder_out_dim)

            # logits = model.joiner(current_encoder_out, decoder_out, project_input=False)
            logits = self.forward_joiner(current_encoder_out, decoder_out)
            # logits is of shape (num_hyps, 1, 1, vocab_size)

            logits = logits.squeeze(1).squeeze(1).to(self.device)

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
                topk_log_probs, topk_indexes = ragged_log_probs[i].topk(self.num_active_paths)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                    topk_token_indexes = (topk_indexes % vocab_size).tolist()

                for k in range(len(topk_hyp_indexes)):
                    hyp_idx = topk_hyp_indexes[k]
                    hyp = A[i][hyp_idx]

                    new_ys = hyp.ys[:]
                    new_token = topk_token_indexes[k]
                    if new_token != self.blank_id:
                        new_ys.append(new_token)

                    new_log_prob = topk_log_probs[k]
                    new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
                    states_list[i].add(new_hyp)
        logger.info("Modified beam search decoding finished")
        return states_list
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        # start_time = time.time()
        logger.info("Exeuting inference for {} requests".format(len(requests))) 
        batch_encoder_out_list = []
        batch_encoder_lens_list = []
        batch_idx = 0
        encoder_max_len = 0

        batch_idx2_corrid = {}
    
        states_list = []
        end_idx = set()

        for request in requests:
            # Perform inference on the request and append it to responses list...
            logger.debug("======Processing request index {} =========", batch_idx)
            # in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            # logger.info("Enabling adapter: {}", self.enable_adapter)
               # Nếu enable_adapter=True, nhận input tensor "adapter_out"
            
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            logger.info("in_0 {}", in_0)
            logger.info("Encoder output sample: {}", in_0.as_numpy().flatten()[:10])
            in_0 = self.forward_adapter(in_0)
            logger.info("Adapter output sample : {}", in_0.as_numpy().flatten()[:10])


            
            # TODO: directly use torch tensor from_dlpack(in_0.to_dlpack())
            # Convert GPU tensor to a torch.Tensor via DLPack,
            # then move to CPU and convert to a NumPy array.
            # torch_tensor = torch.utils.dlpack.from_dlpack(in_0.to_dlpack())
            # logger.info("Adapter output shape: {}", torch_tensor.shape)
            # logger.debug("Adapter output sample: {}", torch_tensor.cpu().numpy().flatten()[:10])

            batch_encoder_out_list.append(in_0.as_numpy())
            # For streaming ASR, assert each request sent from client has batch size 1.
            assert batch_encoder_out_list[-1].shape[0] == 1            
            encoder_max_len = max(encoder_max_len, batch_encoder_out_list[-1].shape[1])
            cur_b_lens = np.array([in_0.as_numpy().shape[1]])#   torch_tensor.cpu().numpy().shape[1]
            batch_encoder_lens_list.append(cur_b_lens)
            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]

            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]

            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]

            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            logger.debug("START value: {}", start)
            logger.debug("READY value: {}", ready)


            if start and ready:
                logger.info("Initializing state for corrid: {}", corrid)
                # intialize states
                if self.decoding_method == 'fast_beam_search':
                    processed_lens = cur_b_lens
                    state = [k2.RnntDecodingStream(self.decoding_graph), processed_lens]
                elif self.decoding_method == 'modified_beam_search':
                    state = HypothesisList()
                    state.add(
                        Hypothesis(
                            ys=[-1] * (self.context_size - 1) + [self.blank_id],
                            log_prob=torch.zeros(1, dtype=torch.float32, device=self.device),
                        )
                    )
                else:
                    assert self.decoding_method == 'greedy_search'
                    state = []
                self.seq_states[corrid] = state
                logger.debug("State initialized for corrid: {}", corrid)

            if end and ready:
                end_idx.add(batch_idx)
                logger.debug("Marking request {} for finalization", batch_idx)
    
            if ready:
                state = self.seq_states[corrid]
                batch_idx2_corrid[batch_idx] = corrid
                states_list.append(state)

            batch_idx += 1

        encoder_out_array = np.zeros((batch_idx, encoder_max_len, self.encoder_dim),
                                  dtype=self.data_type)
        
        
        logger.info("Total requests processed: {}", batch_idx)
        logger.info("Max encoder length: {}", encoder_max_len)


        for i, t in enumerate(batch_encoder_out_list):
            encoder_out_array[i, 0:t.shape[1]] = t
    
        encoder_out = torch.from_numpy(encoder_out_array)
        logger.info("Encoder output prepared with shape: {}", encoder_out.shape)
        # if self.enable_adapter:
        #     encoder_out = self.forward_adapter(encoder_out)

        if self.decoding_method == "fast_beam_search":
            hyps_list = self.fast_beam_search(encoder_out, states_list)
        elif self.decoding_method == "modified_beam_search":
            hyps_list = self.modified_beam_search(encoder_out, states_list)
        else:
            hyps_list = self.greedy_search(encoder_out, states_list)

        logger.info("Decoding method: {}", self.decoding_method)
        logger.info("Number of hypotheses returned: {}", len(hyps_list))

        responses = []
        for i in range(len(hyps_list)):
            if self.decoding_method == 'modified_beam_search':
                best_hyp = hyps_list[i].get_most_probable(length_norm=True)
                hyp = best_hyp.ys[self.context_size:] 
            else:
                hyp = hyps_list[i]
            if hasattr(self.tokenizer, 'token_table'):
                sent = [self.tokenizer.token_table[idx] for idx in hyp]
            else:
                sent = self.tokenizer.decode(hyp).split()
            sent = np.array(sent)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", sent.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
            # update states
            corr = batch_idx2_corrid[i]
            if i in end_idx:
                del self.seq_states[corr]
            else:
                if self.decoding_method == 'fast_beam_search':
                    self.seq_states[corr][1] += batch_encoder_lens_list[i] # stream decoding state is updated in fast_beam_search
                elif self.decoding_method == 'modified_beam_search':
                    self.seq_states[corr] = hyps_list[i]
                else:
                    self.seq_states[corr] = hyp
        logger.info("Inference execution complete for {} request(s)", len(requests))
        assert len(requests) == len(responses)
        # processing_time = time.time() - start_time
        # print("processing time in scorer", processing_time)
        return responses
    
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        logger.info("Finalizing TritonPythonModel and cleaning up resources")
        print('Cleaning up...')