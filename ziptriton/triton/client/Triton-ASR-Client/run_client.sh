# python3 client_copy.py \
#     --manifest-dir /data/quandm8/ziptriton/triton/client/Triton-ASR-Client/datasets/mini_en \
#     --audio-path /data/quandm8/ziptriton/triton/client/Triton-ASR-Client/datasets/mini_en/wav/1089-134686-0001.wav \
#     --model-name transducer \
#     --simulate-streaming \
#     --encoder_right_context 0 \
#     --chunk_size 16 \
#     --subsampling 2 \
#     --num-tasks 100

# python3 client_copy.py \
#     --manifest-dir /data/quangnv53/data_hub/librispeech/kaldi_manifest/test-clean \
#     --model-name transducer \
#     --simulate-streaming \
#     --encoder_right_context 0 \
#     --chunk_size 16 \
#     --subsampling 2 \
#     --num-tasks 100


### Vietnamese ### 

# python3 client_copy.py \
#     --manifest-dir /data/quandm8/ziptriton/triton/client/Triton-ASR-Client/datasets/mini_vn_dhscope \
#     --audio-path /data/quandm8/ziptriton/triton/client/Triton-ASR-Client/datasets/mini_vn_dhscope/wav/speaker_040-000277.wav \
#     --model-name transducer \
#     --simulate-streaming \
#     --encoder_right_context 0 \
#     --chunk_size 16 \
#     --subsampling 2 \
#     --num-tasks 1

# python3 client_copy.py \
#     --manifest-dir /data/quandm8/ziptriton/triton/client/Triton-ASR-Client/datasets/mini_g7 \
#     --model-name transducer \
#     --simulate-streaming \
#     --encoder_right_context 0 \
#     --chunk_size 16 \
#     --subsampling 2 \
#     --num-tasks 50

python3 client_explicit.py \
    --manifest-dir /data/quandm8/ziptriton/triton/client/Triton-ASR-Client/datasets/mini_vnpost \
    --model-name transducer \
    --domain vnpost \
    --simulate-streaming \
    --encoder_right_context 0 \
    --chunk_size 16 \
    --subsampling 2 \
    --num-tasks 50
