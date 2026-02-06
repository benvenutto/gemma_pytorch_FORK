import contextlib
import random

import numpy as np
import torch
from torch.export import Dim

import coremltools as ct

from gemma import config
from gemma import model as gemma_model


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
        # print(torch_model)
    torch_model.eval()
    print("Model loading done")

    # Setup prompt and generation parameters
    prompts = [
        "What are large language models?",
        "How do neurons work?",
        "Can the brain burn ketones instead of glucose?",
    ]
    batch_size = len(prompts)
    output_len = 100
    temperature = 1.0
    top_p = 0.95
    top_k = 64

    # Tokenize the prompts
    prompt_tokens = [torch_model.tokenizer.encode(prompt) for prompt in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len

    ### # build KV caches
    ### kv_caches = []
    ### k_caches = []
    ### v_caches = []
    ### kv_cache_size = (batch_size, max_seq_len, model_config.num_key_value_heads, model_config.head_dim)
    ### for layer_num in range(model_config.num_hidden_layers):
    ###     size = (batch_size, max_seq_len, model_config.num_key_value_heads, model_config.head_dim)
    ###     dtype = model_config.get_dtype()
    ###     k_cache = torch.zeros(size=size, dtype=dtype, device=device)
    ###     v_cache = torch.zeros(size=size, dtype=dtype, device=device)
    ###     kv_caches.append((k_cache, v_cache))

    torch_model.model.initialise_cache(batch_size, max_seq_len, device=device)

    # Prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                  torch_model.tokenizer.pad_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        torch_model.tokenizer.pad_id,
                                        dtype=torch.int64)
    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, :len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != torch_model.tokenizer.pad_id
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                          dtype=torch.int64).to(device)

    # Create mask
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    local_mask_tensor = mask_tensor + torch.tril(
        torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
        diagonal=-model_config.sliding_window_size,
    ) if model_config.sliding_window_size else None
    # curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    curr_mask_tensor = mask_tensor[:, :, input_positions_tensor]
    # curr_local_mask_tensor = local_mask_tensor.index_select(
    #     2, input_positions_tensor
    # ) if local_mask_tensor is not None else None
    curr_local_mask_tensor = local_mask_tensor[:, :, input_positions_tensor] \
        if local_mask_tensor is not None else None

    # Sampling params
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor(
        [temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)

    # Run the PyTorch model
    gen_input_token_ids_tensor = input_token_ids_tensor.clone()
    gen_input_positions_tensor = input_positions_tensor.clone()
    gen_output_positions_tensor = output_positions_tensor.clone()
    gen_curr_mask_tensor = curr_mask_tensor.clone()
    gen_curr_local_mask_tensor = curr_local_mask_tensor.clone()

    output_index = torch.tensor([min_prompt_len], dtype=torch.int64).to(device)
    for i in range(max_seq_len - min_prompt_len):
        next_token_ids, logits = torch_model(
            input_token_ids=gen_input_token_ids_tensor,
            input_positions=gen_input_positions_tensor,
            # kv_write_indices=None,
            mask=gen_curr_mask_tensor,
            output_positions=gen_output_positions_tensor,
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
            local_mask=gen_curr_local_mask_tensor,
        )
        # curr_prompt_mask = prompt_mask_tensor.index_select(
        #     1, output_index).squeeze(dim=1)
        curr_prompt_mask = prompt_mask_tensor[:, output_index].squeeze(dim=1)
        # curr_token_ids = token_ids_tensor.index_select(
        #     1, output_index).squeeze(dim=1)
        curr_token_ids = token_ids_tensor[:, output_index].squeeze(dim=1)
        output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                       next_token_ids).unsqueeze(dim=1)
        token_ids_tensor.index_copy_(1, output_index, output_token_ids)

        gen_input_token_ids_tensor = output_token_ids
        gen_input_positions_tensor = output_index
        # gen_curr_mask_tensor = mask_tensor.index_select(2,
        #                                             gen_input_positions_tensor)
        gen_curr_mask_tensor = mask_tensor[:, :, gen_input_positions_tensor]
        # gen_curr_local_mask_tensor = local_mask_tensor.index_select(
        #     2, gen_input_positions_tensor
        # ) if local_mask_tensor is not None else None
        gen_curr_local_mask_tensor = local_mask_tensor[:, :, gen_input_positions_tensor] \
            if local_mask_tensor is not None else None
        gen_output_positions_tensor = torch.tensor([0], dtype=torch.int64).to(
            device)
        output_index = output_index + 1

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
      trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
      if torch_model.tokenizer.eos_id in trimmed_output:
        eos_index = trimmed_output.index(torch_model.tokenizer.eos_id)
        trimmed_output = trimmed_output[:eos_index]
      results.append(torch_model.tokenizer.decode(trimmed_output))

    print(f"PyTorch model result is: {results}")

    # Setup model parameters & dynamic shapes
    kv_write_indices_tensor = input_positions_tensor    ### will get overwritten
    model_params = (
        input_token_ids_tensor,
        input_positions_tensor,
        # kv_write_indices_tensor,
        curr_mask_tensor,
        output_positions_tensor,
        temperatures_tensor,
        top_ps_tensor,
        top_ks_tensor,
        curr_local_mask_tensor,
    )

    dynamic_shapes = {
        "input_token_ids": (Dim.AUTO, Dim.AUTO, ),
        "input_positions": (Dim.AUTO, ),
        # "kv_write_indices": None,
        "mask": (Dim.AUTO, Dim.AUTO, Dim.AUTO, Dim.AUTO, ),
        "output_positions": (Dim.AUTO,),
        "temperatures": None,
        "top_ps": None,
        "top_ks": None,
        "local_mask": (Dim.AUTO, Dim.AUTO, Dim.AUTO, Dim.AUTO, ),
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
        # edge_program: EdgeProgramManager = to_edge(aten_program, compile_config=EdgeCompileConfig(_check_ir_validity=False))

    # with torch.no_grad():
    #     exported_program = exported_program.run_decompositions(decomp_table={})

    # with torch.no_grad():
    #     exported_edge_program: EdgeProgramManager = to_edge(exported_program)

    # edge_program: ExecutorchProgram = to_edge_transform_and_lower(
    #     exported_program,
    #     partitioner=[CoreMLPartitioner]
    # )


    # print(exported_program)
    cache_size = torch_model.model.get_kv_cache().get_cache_size()
    num_caches = torch_model.model.get_kv_cache().get_num_caches()
    expected_prefix = 'model.layers.25.self_attn.kv_caches'
    state_model = [
        ct.StateType(wrapped_type=ct.TensorType(shape=cache_size), name=f'{expected_prefix}.{kv_name}_{layer_index}')
        for layer_index in range(num_caches) for kv_name in ['k_cache', 'v_cache']
    ]
    ml_model = ct.convert(
        simplified_aten_program,
        source='pytorch',
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        states=state_model,
    )

    print(f">> CoreML model:\n{ml_model}")
    # print(logits.shape)


if __name__ == "__main__":
    main()
