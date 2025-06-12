# CUDA_VISIBLE_DEVICES=0 python train.py /data/quandm8/zipformer/egs/adapter/config/dora.yaml # Train lại với rank = 8
# CUDA_VISIBLE_DEVICES=0 python train.py /data/quandm8/zipformer/egs/adapter/config/rslora.yaml # Train lai với rank cao hơn 
# CUDA_VISIBLE_DEVICES=0 python train_adapter.py /data/quandm8/zipformer/egs/adapter/config/zipformer_adapter.yaml
# CUDA_VISIBLE_DEVICES=1 python train.py /data/quandm8/zipformer/egs/adapter/config/olora.yaml
# CUDA_VISIBLE_DEVICES=1 python train.py /data/quandm8/zipformer/egs/adapter/config/zipformer_lora_k2.yaml


