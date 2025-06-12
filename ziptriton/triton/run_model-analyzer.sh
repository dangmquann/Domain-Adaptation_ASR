# model-analyzer profile \
#     --model-repository /workspace/model_repository \
#     --profile-models encoder \
#     --triton-launch-mode=docker \
#     --output-model-repository-path /workspace/output/ \
#     --override-output-model-repository \
#     -f perf.yaml
    # --run-config-search-max-concurrency 2 \
    # --run-config-search-max-model-batch-size 2 \
    # --run-config-search-max-instance-count 2 \
    # --export-path profile_results


sudo docker run -it --gpus all --shm-size 1G -v /var/run/docker.sock:/var/run/docker.sock -v $PWD:/workspace/triton \
    --net=host nvcr.io/nvidia/tritonserver:24.12-py3-sdk
    #-v /data/quandm8/ziptriton/triton:workspace/triton \

model-analyzer profile \
    --model-repository /data/quandm8/ziptriton/triton/model_repository \
    --profile-models transducer \
    --triton-launch-mode=docker --triton-docker-shm-size=1G \
    --triton-docker-image=sherpa_triton_server \
    --output-model-repository-path /data/quandm8/ziptriton/triton/output/ \
    --export-path /data/quandm8/ziptriton/triton/profile_results \
    --run-config-search-max-concurrency 2 \
    --override-output-model-repository \
 

    