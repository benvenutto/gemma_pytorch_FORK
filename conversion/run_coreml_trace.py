import contextlib
import random

import numpy as np
import torch
from torch.export import Dim

import coremltools as ct

from gemma import config
from gemma import model as gemma_model

from ops.tensor_transformation import index_copy, multinomial

from util.generation import TorchPredictor, CoreMlPredictor, generate


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
        torch_model = gemma_model.GemmaForCausalLM(model_config)
        torch_model.load_weights(ckpt)
        torch_model = torch_model.to(device).eval()

    torch_model.eval()
    print("Model loading done")

    # Setup prompt and generation parameters
    prompts = [
        "What are large language models?",
        "How do neurons work?",
        "Can the brain burn ketones instead of glucose?",
    ]
    max_new_tokens = 100

    # Run generation using TorchPredictor
    input_ids = [torch_model.tokenizer.encode(p) for p in prompts]

    output_ids = generate(
        pred_model=TorchPredictor(torch_model),
        input_ids=input_ids,
        config=model_config,
        pad_id=torch_model.tokenizer.pad_id,
        eos_id=torch_model.tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    results = [torch_model.tokenizer.decode(ids) for ids in output_ids]
    print(f"PyTorch model result is: {results}")

    # Prepare example inputs for torch.export (needed for CoreML conversion)
    batch_size = len(prompts)
    min_prompt_len = min(len(p) for p in input_ids)
    max_prompt_len = max(len(p) for p in input_ids)
    max_seq_len = max_prompt_len + max_new_tokens

    # Re-initialize cache for export
    torch_model.model.initialise_cache(batch_size, max_seq_len, device=device)

    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        torch_model.tokenizer.pad_id,
                                        dtype=torch.int64)
    for i, p in enumerate(input_ids):
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len])
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                          dtype=torch.int64).to(device)

    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    local_mask_tensor = mask_tensor + torch.tril(
        torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
        diagonal=-model_config.sliding_window_size,
    ) if model_config.sliding_window_size else None
    curr_mask_tensor = mask_tensor[:, :, input_positions_tensor]
    curr_local_mask_tensor = local_mask_tensor[:, :, input_positions_tensor] \
        if local_mask_tensor is not None else None

    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)

    # Setup model parameters & dynamic shapes (no sampling params — sampling is external)
    model_params = (
        input_token_ids_tensor,
        input_positions_tensor,
        curr_mask_tensor,
        output_positions_tensor,
        curr_local_mask_tensor,
    )

    # Dynamic dims: during decode steps, sequence-length dims shrink to 1.
    # Dim.AUTO infers the lower bound from the example, which is too large
    # for single-token decode. Use explicit Dim(min=1) for those axes.
    seq_dim = Dim("seq", min=1)
    dynamic_shapes = {
        'input_token_ids': (None, seq_dim, ),
        'input_positions': (seq_dim, ),
        'mask': (None, None, seq_dim, None, ),
        'output_positions': (None,),
        'local_mask': (None, None, seq_dim, None, ),
    }

    # Export the PyTorch model, functionally simplify for inference & convert to a CoreML model
    with torch.inference_mode():
        aten_program = torch.export.export(
            torch_model,
            model_params,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )
        simplified_aten_program = aten_program.run_decompositions(decomp_table={})

    cache_size = torch_model.model.get_kv_cache().get_cache_size()
    num_caches = torch_model.model.get_kv_cache().get_num_caches()
    expected_prefix = 'model.layers.25.self_attn.kv_caches'
    state_model = [
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_size), name=f'{expected_prefix}.{kv_name}_{layer_index}')
        for layer_index in range(num_caches) for kv_name in ['k_cache', 'v_cache']
    ]
    coreml_model = ct.convert(
        simplified_aten_program,
        source='pytorch',
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        states=state_model,
        outputs=[ct.TensorType(name='logits')]
    )

    print(f">> CoreML model:\n{coreml_model}")

    # Run generation using CoreMlPredictor
    coreml_output_ids = generate(
        pred_model=CoreMlPredictor(coreml_model),
        input_ids=input_ids,
        config=model_config,
        pad_id=torch_model.tokenizer.pad_id,
        eos_id=torch_model.tokenizer.eos_id,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    coreml_results = [torch_model.tokenizer.decode(ids) for ids in coreml_output_ids]
    print(f"CoreML model result is: {coreml_results}")


if __name__ == "__main__":
    main()
