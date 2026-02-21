from abc import ABC, abstractmethod
from typing import Union, Tuple, Sequence, Any, List

import torch
from gemma.model import GemmaForCausalLM

from coremltools.models.model import MLState


class PredictorInterface(ABC):
    @abstractmethod
    def __call__(
            self,
            input_token_ids: torch.Tensor,
            input_positions: torch.Tensor,
            mask: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class TorchPredictor(PredictorInterface):
    def __init__(self, torch_model):
        self.torch_model = torch_model

    def __call__(
            self,
            input_token_ids: torch.Tensor,
            input_positions: torch.Tensor,
            mask: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.torch_model(
            input_token_ids,
            input_positions,
            mask,
            output_positions,
            temperatures,
            top_ps,
            top_ks,
            local_mask,
        )

class CoreMlPredictor(PredictorInterface):
    def __init__(self, coreml_model):
        self.coreml_model = coreml_model

    def __call__(
            self,
            input_token_ids: torch.Tensor,
            input_positions: torch.Tensor,
            mask: torch.Tensor,
            output_positions: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
            local_mask: torch.Tensor | None = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch_dtype_coreml_int = torch.int32
        torch_dtype_coreml_float = torch.float32

        input_token_ids_tensor_val = input_token_ids.cpu().to(dtype=torch_dtype_coreml_int).numpy()
        input_positions_tensor_val = input_positions.cpu().to(dtype=torch_dtype_coreml_int).numpy()
        mask_tensor_val = mask.cpu().to(dtype=torch_dtype_coreml_float).numpy()
        output_positions_tensor_val = output_positions.cpu().to(dtype=torch_dtype_coreml_int).numpy()
        temperatures_tensor_val = temperatures.cpu().to(dtype=torch_dtype_coreml_float).numpy()
        top_ps_tensor_val = top_ps.cpu().to(dtype=torch_dtype_coreml_float).numpy()
        top_ks_tensor_val = top_ks.cpu().to(dtype=torch_dtype_coreml_int).numpy()
        local_mask_tensor_val = local_mask.cpu().to(dtype=torch_dtype_coreml_float).numpy()

        coreml_inputs = {
            'input_token_ids': input_token_ids_tensor_val,
            'input_positions': input_positions_tensor_val,
            'mask': mask_tensor_val,
            'output_positions': output_positions_tensor_val,
            'temperatures': temperatures_tensor_val,
            'top_ps': top_ps_tensor_val,
            'top_ks': top_ks_tensor_val,
            'local_mask': local_mask_tensor_val,
        }
        model_state: MLState = self.coreml_model.make_state()
        next_tokens, logits = self.coreml_model.predict(coreml_inputs, state=model_state)
        return torch.from_numpy(next_tokens), torch.from_numpy(logits)


def tokenize_prompts(
        torch_model: GemmaForCausalLM,
        prompts: Union[str, Sequence[str]],
        output_len=int,
        device=Any
) -> dict:
    """Return tokenized prompt as padded tensors.
    """
    # Tokenize batch
    if isinstance(prompts, str):
        prompts = [prompts]
    prompt_tokens = [torch_model.tokenizer.encode(prompt) for prompt in prompts]

    # Get token lengths
    batch_size = len(prompt_tokens)
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len

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
        diagonal=-torch_model.config.sliding_window_size,
    ) if torch_model.config.sliding_window_size else None
    # curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    curr_mask_tensor = mask_tensor[:, :, input_positions_tensor]
    # curr_local_mask_tensor = local_mask_tensor.index_select(
    #     2, input_positions_tensor
    # ) if local_mask_tensor is not None else None
    curr_local_mask_tensor = local_mask_tensor[:, :, input_positions_tensor] \
        if local_mask_tensor is not None else None

    # Sampling params
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)

    return {
        'input_token_ids': input_token_ids_tensor,
        'input_positions': input_positions_tensor,
        'mask': curr_mask_tensor,
        'output_positions': output_positions_tensor,
        'local_mask': curr_local_mask_tensor,
        'min_prompt_len':min_prompt_len,
        'max_seq_len': max_seq_len,
    }


def generate(
        torch_model: GemmaForCausalLM,
        pred_model: PredictorInterface,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        local_mask: torch.Tensor,
        min_prompt_len: torch.Tensor,
        max_seq_len: torch.Tensor,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        device=Any,
) -> torch.Tensor:

    # Initialize cache for expected outputs
    torch_model.model.initialise_cache(batch_size, max_seq_len, device=device)


    pass