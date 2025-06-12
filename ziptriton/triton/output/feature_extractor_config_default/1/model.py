# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from torch.utils.dlpack import from_dlpack
import torch
import kaldifeat
import _kaldifeat
from typing import List
import json
import numpy as np
import time
class Fbank(torch.nn.Module):
    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)
    
class Feat(object):
    def __init__(self, seqid, sample_rate,frame_stride, device='cpu'):
        self.seqid = seqid
        self.sample_rate = sample_rate
        self.wav = torch.zeros(240, dtype=torch.float32, device=device)
        self.offset = 240
        self.frames = torch.full((13,80),-23.025850929940457, dtype=torch.float32, device=device)
        self.frame_stride = int(frame_stride)
        self.device = device

    def add_wavs(self, wav: torch.tensor):
        wav = wav.to(self.device)
        self.wav = torch.cat([self.wav, wav], axis=0)

    def get_seg_wav(self):
        seg = self.wav
        self.wav = self.wav[-self.offset:]
        return seg

    def add_frames(self, frames: torch.tensor):
        """
        frames: seq_len x feat_sz
        """
        self.frames = torch.cat([self.frames, frames], axis=0)

    def get_frames(self):
        seg = self.frames
        self.frames = self.frames[self.frame_stride:]
        return seg

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
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "x")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        if self.output0_dtype == np.float32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self.feature_size = output0_config['dims'][-1]
        self.decoding_window = output0_config['dims'][-2]

        parameters = model_config["parameters"]
        sample_rate = int(parameters["sample_rate"]['string_value'])
        frame_length_ms = int(parameters["frame_length_ms"]['string_value'])
        frame_shift_ms = int( parameters["frame_shift_ms"]['string_value'])
        decode_chunk_size = int(parameters["decode_chunk_size"]['string_value'])

        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = self.feature_size
        opts.frame_opts.frame_length_ms = frame_length_ms
        opts.frame_opts.frame_shift_ms = frame_shift_ms
        opts.frame_opts.samp_freq = sample_rate
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)

        self.seq_feat = {}

        self.frame_stride = int(decode_chunk_size*2)
        self.sample_rate = sample_rate
        self.min_seg = frame_length_ms * sample_rate // 100

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
        # start_time = time.time()
        total_waves = []
        responses = []
        batch_seqid = []
        end_seqid = {}
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            #  wavs = input0.as_numpy()[0]
            wavs = from_dlpack(input0.to_dlpack())[0]
            # print(wavs)
            input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")
            #  wav_lens = input1.as_numpy()[0][0]
            wav_lens = from_dlpack(input1.to_dlpack())[0]
            # print(len(wavs))
            # print(wav_lens)
            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]
            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]
            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]
            # print('corrid', corrid)
            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]
            if start:
                self.seq_feat[corrid] = Feat(corrid,
                                             self.sample_rate,
                                             self.frame_stride,
                                             self.device)
            if ready:
                self.seq_feat[corrid].add_wavs(wavs[0:wav_lens])

            batch_seqid.append(corrid)
            if end:
                end_seqid[corrid] = 1
            wavs = self.seq_feat[corrid].get_seg_wav()
            if len(wavs) < self.min_seg:
                temp = torch.zeros(self.min_seg, dtype=torch.float32,
                                   device=self.device)
                temp[0:len(wavs)] = wavs[:]
                wavs = temp
        
            total_waves.append(wavs)
        
        features = self.feature_extractor(total_waves)
        batch_size = len(batch_seqid)
        batch_speech = torch.zeros((batch_size, self.decoding_window,
                                    self.feature_size), dtype=self.dtype)
        batch_speech_lens = torch.zeros((batch_size, 1), dtype=torch.int64)
        i = 0
        for corrid, frames in zip(batch_seqid, features):
            self.seq_feat[corrid].add_frames(frames)
            r_frames = self.seq_feat[corrid].get_frames()
            speech = batch_speech[i: i + 1]
            speech_lengths = batch_speech_lens[i: i + 1]
            i += 1
            speech_lengths[0] = r_frames.size(0)
            speech[0][0:r_frames.size(0)] = r_frames.to(speech.device)
            # out_tensor0 = pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech))
            # out_tensor1 = pb_utils.Tensor.from_dlpack("speech_lengths",
            #                                            to_dlpack(speech_lengths))
            out_tensor0 = pb_utils.Tensor("x", speech.numpy())
            
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor0])
            responses.append(response)
            if corrid in end_seqid:
                del self.seq_feat[corrid]
        # process_time = time.time() - start_time
        # print("process time feature extractor", process_time)
        return responses

    def finalize(self):
        print("Remove feature extractor!")
