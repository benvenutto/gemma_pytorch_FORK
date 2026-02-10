import contextlib
import random
from typing import Tuple

import numpy as np
import torch
from torch.export import Dim

# from executorch.backends.apple.coreml.partition import CoreMLPartitioner
# from executorch.exir import ExecutorchProgram, to_edge_transform_and_lower

import coremltools as ct

from gemma import config
from gemma import model as gemma_model
from gemma.model import GemmaModel, Gemma2DecoderLayer


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main():
    # Flags set here
    variant = '1b'
    quant = False
    seed = 42
    compute_device = 'mps'
    ckpt = './model/1b/model.ckpt'
    tokenizer = './model/1b/tokenizer.model'

    # Construct the model config.
    model_config = config.get_model_config(variant)
    model_config.dtype = "float32"
    model_config.quant = quant
    model_config.tokenizer = tokenizer

    # Seed random.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Setup compute device
    device = torch.device(compute_device)

    # Create the model and load the weights.
    with _set_default_tensor_type(model_config.get_dtype()):
        torch_causal_model = gemma_model.GemmaForCausalLM(model_config)
        torch_causal_model.load_weights(ckpt)
        torch_causal_model = torch_causal_model.to(device)
    torch_causal_model.eval()

    # Pick a decoder layer
    torch_model: GemmaModel = torch_causal_model.model
    torch_decoder_layer: Gemma2DecoderLayer = torch_model.layers[0]
    print(f"First decoder layer: \n{torch_decoder_layer}")

    # Setup dummy layer params
    hidden_states: torch.Tensor = torch.rand((1, 7, 1152),
                                             dtype=model_config.get_dtype(),
                                             device=device)
    freqs_cis: torch.Tensor = torch.rand((7, 128),
                                         dtype=model_config.get_dtype(),
                                         device=device)
    kv_write_indices: torch.Tensor = torch.rand((7, ),
                                                dtype=model_config.get_dtype(),
                                                device=device)
    kv_cache: Tuple[torch.Tensor, torch.Tensor] = (
        torch.rand((1, 107, 1, 256),
                   dtype=model_config.get_dtype(),
                   device=device),
        torch.rand((1, 107, 1, 256),
                   dtype=model_config.get_dtype(),
                   device=device),
    )
    mask: torch.Tensor = torch.rand((1, 1, 7, 107),
                                    dtype=model_config.get_dtype(),
                                    device=device)
    local_mask: torch.Tensor =torch.rand((1, 1, 7, 107),
                                    dtype=model_config.get_dtype(),
                                    device=device)

    # Setup model parameters & dynamic shapes
    decoder_layer_inputs = (
        hidden_states,
        freqs_cis,
        kv_write_indices,
        kv_cache,
        mask,
        local_mask,
    )
    batch_size_dim = Dim.AUTO
    seq_length_dim = Dim('seq_length')
    max_seq_len_dim = Dim('max_seq_len')
    dynamic_shapes = {
        "hidden_states": {0: batch_size_dim, 1: seq_length_dim},
        "freqs_cis": {0: seq_length_dim},
        "kv_write_indices": {0: seq_length_dim},
        "kv_cache": (
            {0: batch_size_dim, 1: max_seq_len_dim, 2: Dim.STATIC, 3: Dim.STATIC},
            {0: batch_size_dim, 1: max_seq_len_dim, 2: Dim.STATIC, 3: Dim.STATIC},
        ),
        "mask": (Dim.AUTO, Dim.AUTO, Dim.AUTO, Dim.AUTO),
        "local_mask": (Dim.AUTO, Dim.AUTO, Dim.AUTO, Dim.AUTO),
    }

    # # Export the pyTorch model, functionally simplify for inference & convert to a CoreML model
    # with torch.inference_mode():
    #     exported_program = torch.export.export(
    #         torch_decoder_layer,
    #         decoder_layer_inputs,
    #         dynamic_shapes=dynamic_shapes,
    #         strict=False,
    #     )

    exported_program = torch.export.export(torch_decoder_layer, decoder_layer_inputs)
    exported_program_inference = exported_program.run_decompositions(decomp_table={})
    print(exported_program_inference)

    # with torch.no_grad():
    #     exported_program = exported_program.run_decompositions(decomp_table={})

    # with torch.no_grad():
    #     exported_edge_program: EdgeProgramManager = to_edge(exported_program)

    # edge_program: ExecutorchProgram = to_edge_transform_and_lower(
    #     exported_program,
    #     partitioner=[CoreMLPartitioner]
    # )

    ml_model = ct.convert(
        exported_program_inference,
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS26,
        compute_units=ct.ComputeUnit.ALL,
        states=[
            ct.StateType(
                wrapped_type=[
                    ct.TensorType(), ct.TensorType(),
                ],
                name='kv_cache',
            )
        ],
    )

    print(f">> CoreML model:\n{ml_model}")
    # print(logits.shape)


if __name__ == "__main__":
    main()
