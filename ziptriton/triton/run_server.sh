tritonserver --model-repository=model_repository \
    --http-port 10086 --grpc-port 7001 --metrics-port 10087 \
   --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000 \
   --model-control-mode=explicit --strict-model-config=false #--log-verbose=1
