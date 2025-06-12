sudo docker run --gpus all --rm -v $PWD:/workspace/sherpa --name sherpa_server --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it sherpa_triton_server

# sudo docker run -it --gpus "device=1" -v /var/run/docker.sock:/var/run/docker.sock -v /data/quandm8/ziptriton/triton:/data/quandm8/ziptriton/triton --net=host nvcr.io/nvidia/tritonserver:23.02-py3-sdk    