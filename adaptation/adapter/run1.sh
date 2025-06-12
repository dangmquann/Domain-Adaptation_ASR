cd /data/quandm8/zipformer/egs/adapter
# CUDA_VISIBLE_DEVICES=1 python train.py /data/quandm8/zipformer/egs/adapter/config/zipformer_lora.yaml   # Adapter + LoRA (r=8)
CUDA_VISIBLE_DEVICES=1 python train_adapter.py /data/quandm8/zipformer/egs/adapter/config/zipformer_adapter.yaml
# CUDA_VISIBLE_DEVICES=1 python train.py /data/quandm8/zipformer/egs/adapter/config/adaplora.yaml   # LoRA for G7
# CUDA_VISIBLE_DEVICES=1 python train.py /data/quandm8/zipformer/egs/adapter/config/hra.yaml
# CUDA_VISIBLE_DEVICES=1 python train.py /data/quandm8/zipformer/egs/adapter/config/eva.yaml


