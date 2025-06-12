import argparse 
import logging 
import torch
from .onnx_pretrained import OnnxModel
from hyperpyyaml import load_hyperpyyaml


def load_config(config_path):
    with open(config_path, "r") as fin:
        config = load_hyperpyyaml(fin)
    return config

def test_encoder(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    C = 80
    for i in range(3):
        N = torch.randint(low=1, high=20, size=(1,)).item()
        T = torch.randint(low=30, high=50, size=(1,)).item()
        logging.info(f"test_encoder: iter {i}, N={N}, T={T}")

        x = torch.rand(N, T, C)
        x_lens = torch.randint(low=30, high=T + 1, size=(N,))
        x_lens[0] = T

        torch_encoder_out, torch_encoder_out_lens = torch_model.encoder(x, x_lens)
        torch_encoder_out = torch_model.joiner.encoder_proj(torch_encoder_out)

        onnx_encoder_out, onnx_encoder_out_lens = onnx_model.run_encoder(x, x_lens)

        assert torch.allclose(torch_encoder_out, onnx_encoder_out, atol=1e-05), (
            (torch_encoder_out - onnx_encoder_out).abs().max()
        )


def test_decoder(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    context_size = onnx_model.context_size
    vocab_size = onnx_model.vocab_size
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_decoder: iter {i}, N={N}")
        x = torch.randint(
            low=1,
            high=vocab_size,
            size=(N, context_size),
            dtype=torch.int64,
        )
        torch_decoder_out = torch_model.decoder(x, need_pad=torch.tensor([False]))
        torch_decoder_out = torch_model.joiner.decoder_proj(torch_decoder_out)
        torch_decoder_out = torch_decoder_out.squeeze(1)

        onnx_decoder_out = onnx_model.run_decoder(x)
        assert torch.allclose(torch_decoder_out, onnx_decoder_out, atol=1e-4), (
            (torch_decoder_out - onnx_decoder_out).abs().max()
        )


def test_joiner(
    torch_model: torch.jit.ScriptModule,
    onnx_model: OnnxModel,
):
    encoder_dim = torch_model.joiner.encoder_proj.weight.shape[1]
    decoder_dim = torch_model.joiner.decoder_proj.weight.shape[1]
    for i in range(10):
        N = torch.randint(1, 100, size=(1,)).item()
        logging.info(f"test_joiner: iter {i}, N={N}")
        encoder_out = torch.rand(N, encoder_dim)
        decoder_out = torch.rand(N, decoder_dim)

        projected_encoder_out = torch_model.joiner.encoder_proj(encoder_out)
        projected_decoder_out = torch_model.joiner.decoder_proj(decoder_out)

        torch_joiner_out = torch_model.joiner(encoder_out, decoder_out)
        onnx_joiner_out = onnx_model.run_joiner(
            projected_encoder_out, projected_decoder_out
        )

        assert torch.allclose(torch_joiner_out, onnx_joiner_out, atol=1e-4), (
            (torch_joiner_out - onnx_joiner_out).abs().max()
        )

@torch.no_grad()
def main():
    config = load_config("decode_config.yaml")

    logging.info("Loaded configuration:")
    logging.info(config)

    torch_model = torch.jit.load(config["torch_model_filename"])
        # Khởi tạo mô hình ONNX
    onnx_model = OnnxModel(
        encoder_model_filename=config["encoder_model_filename"],
        decoder_model_filename=config["decoder_model_filename"],
        jointer_model_filename=config["jointer_model_filename"],
    )

    logging.info("Test encoder")
    test_encoder(torch_model, onnx_model)

    logging.info("Test decoder")
    test_decoder(torch_model, onnx_model)

    logging.info("Test joiner")
    test_joiner(torch_model, onnx_model)
    logging.info("Finished checking ONNX models")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# See https://github.com/pytorch/pytorch/issues/38342
# and https://github.com/pytorch/pytorch/issues/33354
#
# If we don't do this, the delay increases whenever there is
# a new request that changes the actual batch size.
# If you use `py-spy dump --pid <server-pid> --native`, you will
# see a lot of time is spent in re-compiling the torch script model.
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    torch.manual_seed(20220727)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
