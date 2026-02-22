from abc import ABC, abstractmethod
from typing import Union, Tuple, Any

import torch
from gemma.config import GemmaConfig


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

    @abstractmethod
    def reset(self, batch_size: int, max_seq_len: int, device: Any) -> None:
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

    def reset(self, batch_size: int, max_seq_len: int, device: Any) -> None:
        self.torch_model.model.initialise_cache(batch_size, max_seq_len, device=device)


class CoreMlPredictor(PredictorInterface):
    def __init__(self, coreml_model):
        from coremltools.models.model import MLState
        self.coreml_model = coreml_model
        self.model_state: MLState = self.coreml_model.make_state()

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
        top_ps_tensor_val = top_ps.cpu().to(dtype=torch_dtype_coreml_float).numpy()
        top_ks_tensor_val = top_ks.cpu().to(dtype=torch_dtype_coreml_int).numpy()

        coreml_inputs = {
            'input_token_ids': input_token_ids_tensor_val,
            'input_positions': input_positions_tensor_val,
            'mask': mask_tensor_val,
            'output_positions': output_positions_tensor_val,
            'top_ps': top_ps_tensor_val,
            'top_ks': top_ks_tensor_val,
        }

        if temperatures is not None:
            coreml_inputs['temperatures'] = temperatures.cpu().to(
                dtype=torch_dtype_coreml_float).numpy()

        if local_mask is not None:
            coreml_inputs['local_mask'] = local_mask.cpu().to(
                dtype=torch_dtype_coreml_float).numpy()

        device = input_token_ids.device
        preds = self.coreml_model.predict(coreml_inputs, state=self.model_state)
        next_tokens = torch.from_numpy(preds['next_tokens']).to(dtype=torch.int64, device=device)
        logits = torch.from_numpy(preds['logits']).to(device)
        return next_tokens, logits

    def reset(self, batch_size: int, max_seq_len: int, device: Any) -> None:
        self.model_state = self.coreml_model.make_state()


def generate(
        pred_model: PredictorInterface,
        input_ids: list[list[int]],
        config: GemmaConfig,
        pad_id: int,
        eos_id: int,
        max_new_tokens: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        device: Any = None,
) -> list[list[int]]:
    """Generates token sequences using an autoregressive loop.

    Args:
        pred_model: A PredictorInterface wrapping either a torch or CoreML model.
        input_ids: Pre-tokenized prompts as a list of token ID lists.
        config: GemmaConfig for the model (used for sliding_window_size).
        pad_id: Padding token ID from the tokenizer.
        eos_id: End-of-sequence token ID from the tokenizer.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (None or 0 for greedy).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        device: Torch device for tensor allocation.

    Returns:
        List of generated token ID lists (one per prompt, prompt stripped,
        truncated at EOS).
    """
    batch_size = len(input_ids)
    min_prompt_len = min(len(p) for p in input_ids)
    max_prompt_len = max(len(p) for p in input_ids)
    max_seq_len = max_prompt_len + max_new_tokens

    # Build padded token tensors
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                  pad_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                        pad_id, dtype=torch.int64)
    for i, p in enumerate(input_ids):
        token_ids_tensor[i, :len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
            p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != pad_id
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                          dtype=torch.int64).to(device)

    # Causal mask
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)

    # Sliding window mask
    local_mask_tensor = mask_tensor + torch.tril(
        torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
        diagonal=-config.sliding_window_size,
    ) if config.sliding_window_size else None

    curr_mask_tensor = mask_tensor[:, :, input_positions_tensor]
    curr_local_mask_tensor = local_mask_tensor[:, :, input_positions_tensor] \
        if local_mask_tensor is not None else None

    # Sampling parameters
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor(
        [temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)

    # Initialize predictor state (KV cache)
    pred_model.reset(batch_size, max_seq_len, device=device)

    # Autoregressive generation loop
    gen_input_token_ids = input_token_ids_tensor.clone()
    gen_input_positions = input_positions_tensor.clone()
    gen_output_positions = output_positions_tensor.clone()
    gen_curr_mask = curr_mask_tensor.clone()
    gen_curr_local_mask = curr_local_mask_tensor.clone() \
        if curr_local_mask_tensor is not None else None

    output_index = torch.tensor([min_prompt_len], dtype=torch.int64).to(device)
    for i in range(max_seq_len - min_prompt_len):
        next_token_ids, logits = pred_model(
            input_token_ids=gen_input_token_ids,
            input_positions=gen_input_positions,
            mask=gen_curr_mask,
            output_positions=gen_output_positions,
            temperatures=temperatures_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
            local_mask=gen_curr_local_mask,
        )

        curr_prompt_mask = prompt_mask_tensor[:, output_index].squeeze(dim=1)
        curr_token_ids = token_ids_tensor[:, output_index].squeeze(dim=1)
        output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                       next_token_ids).unsqueeze(dim=1)
        token_ids_tensor.index_copy_(1, output_index, output_token_ids)

        gen_input_token_ids = output_token_ids
        gen_input_positions = output_index
        gen_curr_mask = mask_tensor[:, :, gen_input_positions]
        gen_curr_local_mask = local_mask_tensor[:, :, gen_input_positions] \
            if local_mask_tensor is not None else None
        gen_output_positions = torch.tensor([0], dtype=torch.int64).to(device)
        output_index = output_index + 1

    # Extract generated tokens (strip prompt, truncate at EOS)
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
        trimmed_output = tokens[len(input_ids[i]):len(input_ids[i])
                                + max_new_tokens]
        if eos_id in trimmed_output:
            eos_index = trimmed_output.index(eos_id)
            trimmed_output = trimmed_output[:eos_index]
        results.append(trimmed_output)

    return results
