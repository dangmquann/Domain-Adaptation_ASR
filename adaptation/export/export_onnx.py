import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import shutil
import logging
import argparse
from typing import Dict, Tuple
from hyperpyyaml import load_hyperpyyaml

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16

from humming.layers.scaling_converter import convert_scaled_to_non_scaled
from humming.layers.zipformer import Zipformer2
from humming.layers.decoder import Decoder
from humming.layers.export_onnx import (
    OnnxEncoder,
    OnnxDecoder,
    OnnxJointer,
    OnnxAdapter,
    OnnxEncoderProj,
    export_encoder_proj_onnx,
    export_encoder_model_onnx,
    export_decoder_model_onnx,
    export_jointer_model_onnx,
    export_adapter_onnx,
    make_pad_mask,
    add_meta_data
)

import k2  
import torch.nn as nn

## How to run: /data/quandm8/zipformer/egs/export$ python -m export_onnx /data/quandm8/zipformer/egs/export/config/export-onnx.yaml


def main():
    parser = argparse.ArgumentParser(
        description="Export a trained RNNT (Zipformer) model to ONNX."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Đường dẫn file YAML config, sử dụng định dạng extended YAML (hyperpyyaml)."
    )
    parser.add_argument(
        "--flash",
        action="store_true",
        help="Bật/tắt flash scaled dot product attention (nếu GPU hỗ trợ)."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Bật/tắt torch.compile trong PyTorch 2.0."
    )

    args = parser.parse_args()

    # Tuỳ chọn flash attention
    if args.flash:
        torch.backends.cuda.enable_flash_sdp(True)

    # Đọc config
    with open(args.config) as fin:
        modules = load_hyperpyyaml(fin)

    trainer = modules["trainer"]
    model = modules["model"]

    # Tuỳ chọn compile (PyTorch 2.0)
    if args.compile:
        model = torch.compile(model)

    os.makedirs(trainer.log_dir, exist_ok=True)
    shutil.copy(args.config, trainer.log_dir)

    # Có thể khởi tạo datamodule nếu bạn cần (tùy yêu cầu):
    # datamodule = modules['datamodule']

    export_config = modules["export_config"]
    if export_config["enable_export"]:
        logging.info("Starting ONNX export process ...")
        print("Starting ONNX export process ...")

        export_dir = export_config.get(
            "export_dir",
            os.path.join(trainer.log_dir, "export")
        )
        os.makedirs(export_dir, exist_ok=True)

        # Tải checkpoint
        checkpoint_path = modules["resume_path"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load state_dict vào model
        model.load_state_dict(checkpoint["state_dict"])

        # print(model.encoder)
        # print(model.last_adapter)

        model.to("cpu")
        model.eval()


        # Convert mô hình đã được scale -> non-scale (nếu cần)
        convert_scaled_to_non_scaled(model, inplace=True)

        # Tạo các "wrapper" OnnxEncoder, OnnxDecoder, OnnxJointer
        
        
        if export_config["use_adapters"]:
            encoder = OnnxEncoder(
            encoder=model.encoder,
            encoder_embed=model.encoder_embed,
            # encoder_proj=model.last_adapter.embed_dim,
            )
            adapter = OnnxAdapter(
                adapter_module=model.last_adapter,
                encoder_proj= model.jointer.encoder_proj,
            )
        else:
            encoder = OnnxEncoder(
            encoder=model.encoder,
            encoder_embed=model.encoder_embed,
            # encoder_proj=model.jointer.encoder_proj,
            )
            encoder_proj = OnnxEncoderProj(
                encoder_proj=model.jointer.encoder_proj,
            )

        decoder = OnnxDecoder(
            decoder=model.decoder,
            decoder_proj=model.jointer.decoder_proj,
        )

        jointer = OnnxJointer(output_linear=model.jointer.output_linear)

        # Đếm tham số
        encoder_num_param = sum(p.numel() for p in encoder.parameters())
        decoder_num_param = sum(p.numel() for p in decoder.parameters())
        jointer_num_param = sum(p.numel() for p in jointer.parameters())
        # adapter_num_param = sum(p.numel() for p in adapter.parameters()) if model.last_adapter is not None else 0
        # total_num_param = encoder_num_param + decoder_num_param + jointer_num_param + adapter_num_param

        logging.info(f"encoder parameters: {encoder_num_param}")
        # logging.info(f"adapter parameters: {adapter_num_param}")
        logging.info(f"decoder parameters: {decoder_num_param}")
        logging.info(f"jointer parameters: {jointer_num_param}")
        # logging.info(f"total parameters: {total_num_param}")

        # Tạo suffix cho tên file ONNX
        epoch = export_config.get("epoch", 1)
        avg = export_config.get("avg", 1)
        suffix = f"epoch-{epoch}-avg-{avg}"

        opset_version = export_config.get("opset_version", 11)

        # 1) Export Encoder
        logging.info("Exporting encoder ...")
        encoder_filename = os.path.join(export_dir, f"encoder-{suffix}.onnx")
        export_encoder_model_onnx(
            encoder,
            encoder_filename,
            opset_version=opset_version
        )
        logging.info(f"Exported encoder to {encoder_filename}")

        # 2) Export Adapter or EncoderProj
        if export_config["use_adapters"]:
            logging.info("Exporting adapter ...")
            adapter_filename = os.path.join(export_dir, f"adapter-{suffix}.onnx")
            export_adapter_onnx(
                adapter,
                adapter_filename,
                opset_version=opset_version
            )
            logging.info(f"Exported adapter to {adapter_filename}")
        else:
            logging.info("Exporting encoder_proj ...")
            encoder_proj_filename = os.path.join(export_dir, f"encoder_proj-{suffix}.onnx")
            export_encoder_proj_onnx(
                encoder_proj,
                encoder_proj_filename,
                opset_version=opset_version
            )
            logging.info(f"Exported encoder_proj to {encoder_proj_filename}")

        # 3) Export Decoder
        logging.info("Exporting decoder ...")
        decoder_filename = os.path.join(export_dir, f"decoder-{suffix}.onnx")
        export_decoder_model_onnx(
            decoder,
            decoder_filename,
            opset_version=opset_version
        )
        logging.info(f"Exported decoder to {decoder_filename}")

        # 4) Export Jointer
        logging.info("Exporting jointer ...")
        jointer_filename = os.path.join(export_dir, f"jointer-{suffix}.onnx")
        export_jointer_model_onnx(
            jointer,
            jointer_filename,
            opset_version=opset_version
        )
        logging.info(f"Exported jointer to {jointer_filename}")

        # Tuỳ chọn xuất FP16
        if export_config.get("save_fp16", False):
            logging.info("Generating FP16 models ...")

            # FP16 Encoder
            enc_fp16 = float16.convert_float_to_float16(
                onnx.load(encoder_filename), keep_io_types=True
            )
            enc_fp16_name = os.path.join(export_dir, f"encoder-{suffix}.fp16.onnx")
            onnx.save(enc_fp16, enc_fp16_name)
            logging.info(f"Exported FP16 encoder to {enc_fp16_name}")

            # FP16 Adapter
            if export_config["use_adapters"]:
                adapter_fp16 = float16.convert_float_to_float16(
                    onnx.load(adapter_filename), keep_io_types=True
                )
                adapter_fp16_name = os.path.join(export_dir, f"adapter-{suffix}.fp16.onnx")
                onnx.save(adapter_fp16, adapter_fp16_name)
                logging.info(f"Exported FP16 adapter to {adapter_fp16_name}")
            else:
                encoder_proj_fp16 = float16.convert_float_to_float16(
                    onnx.load(encoder_proj_filename), keep_io_types=True
                )
                encoder_proj_fp16_name = os.path.join(export_dir, f"encoder_proj-{suffix}.fp16.onnx")
                onnx.save(encoder_proj_fp16, encoder_proj_fp16_name)
                logging.info(f"Exported FP16 encoder_proj to {encoder_proj_fp16_name}")

            # FP16 Decoder
            dec_fp16 = float16.convert_float_to_float16(
                onnx.load(decoder_filename), keep_io_types=True
            )
            dec_fp16_name = os.path.join(export_dir, f"decoder-{suffix}.fp16.onnx")
            onnx.save(dec_fp16, dec_fp16_name)
            logging.info(f"Exported FP16 decoder to {dec_fp16_name}")

            # FP16 Jointer
            join_fp16 = float16.convert_float_to_float16(
                onnx.load(jointer_filename), keep_io_types=True
            )
            join_fp16_name = os.path.join(export_dir, f"jointer-{suffix}.fp16.onnx")
            onnx.save(join_fp16, join_fp16_name)
            logging.info(f"Exported FP16 jointer to {join_fp16_name}")

        # Tuỳ chọn xuất INT8
        if export_config.get("save_int8", False):
            logging.info("Generating INT8 quantization models ...")

            # Quantize Encoder
            enc_int8 = os.path.join(export_dir, f"encoder-{suffix}.int8.onnx")
            quantize_dynamic(
                model_input=encoder_filename,
                model_output=enc_int8,
                op_types_to_quantize=["MatMul"],
                weight_type=QuantType.QInt8,
            )
            logging.info(f"Exported INT8 encoder to {enc_int8}")

            # Quantize Adapter
            if export_config["use_adapters"]:
                adapter_int8 = os.path.join(export_dir, f"adapter-{suffix}.int8.onnx")
                quantize_dynamic(
                    model_input=adapter_filename,
                    model_output=adapter_int8,
                    op_types_to_quantize=["MatMul"],
                    weight_type=QuantType.QInt8,
                )
                logging.info(f"Exported INT8 adapter to {adapter_int8}")

            # Quantize Decoder
            dec_int8 = os.path.join(export_dir, f"decoder-{suffix}.int8.onnx")
            quantize_dynamic(
                model_input=decoder_filename,
                model_output=dec_int8,
                op_types_to_quantize=["MatMul", "Gather"],
                weight_type=QuantType.QInt8,
            )
            logging.info(f"Exported INT8 decoder to {dec_int8}")

            # Quantize Jointer
            join_int8 = os.path.join(export_dir, f"jointer-{suffix}.int8.onnx")
            quantize_dynamic(
                model_input=jointer_filename,
                model_output=join_int8,
                op_types_to_quantize=["MatMul"],
                weight_type=QuantType.QInt8,
            )
            logging.info(f"Exported INT8 jointer to {join_int8}")

        logging.info("ONNX export process completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
