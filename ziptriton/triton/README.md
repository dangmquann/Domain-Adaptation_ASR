# Inference Serving Best Practice for Transducer ASR based on Icefall <!-- omit in toc -->
## Prepare Environment
Build the server docker image:
```
cd triton
docker build . -f Dockerfile/Dockerfile.server -t sherpa_triton_server:latest --network host
```
Start the docker container:
```bash
docker run --gpus all -rm -v $PWD:/workspace/sherpa --name sherpa_server --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it sherpa_triton_server
```
Now, you should enter into the container successfully. Inside the container,run server:
```bash
bash run_server.sh
```
## Test client
```bash
cd triton/client/Triton-ASR-client
bash run_client.sh
```


