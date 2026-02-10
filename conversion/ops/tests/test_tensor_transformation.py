from typing import Final

import numpy as np
import torch
from torch import nn
from torch.export import Dim


import coremltools as ct

from .test_util import torch_export_model, coreml_convert_model, run_coreml_model


class TestTensorIndexCopySlice(nn.Module):
    SEQUENCE_DIM: Final[int] = 1

    def __init__(self, target_state):
        super().__init__()
        self.register_buffer('target_state', target_state, persistent=False)

    def forward(self, slice_indices: torch.Tensor, slice_values: torch.Tensor) -> torch.Tensor:
        self.target_state.index_copy_(TestTensorIndexCopySlice.SEQUENCE_DIM, slice_indices, slice_values)
        return self.target_state

    # def forward(self, slice_indices: torch.Tensor, slice_values: torch.Tensor) -> torch.Tensor:
    #     # slice_index = slice_indices[0]
    #     # slice_value = slice_values[:, slice_index]
    #     slice_value = slice_values[:, 0]
    #     slice_value =  slice_value.unsqueeze(TestTensorIndexCopySlice.SEQUENCE_DIM)
    #     self.target_state = self.target_state * slice_value     ### with broadcast
    #     return self.target_state


def test_index_copy():

    ###
    ### Setup PyTorch model
    ###

    torch_device = torch.device('mps')
    torch_data_dtype = torch.float32
    torch_state_dtype = torch.float16
    torch_index_dtype = torch.int64
    torch_coreml_int = torch.int32

    target_state = torch.rand((3, 5, 2, 4), dtype=torch_state_dtype, device=torch_device)
    torch_model = TestTensorIndexCopySlice(target_state).to(device=torch_device)

    ###
    ### Save the original target state, then update the relevant slices, and test
    ###
    saved_target_state = target_state.clone()
    slice_indices = torch.tensor([2, 3, 0], dtype=torch_index_dtype, device=torch_device)
    saved_slice_values = saved_target_state[:, slice_indices].clone()
    slice_values = torch.rand((3, 3, 2, 4), dtype=torch_state_dtype, device=torch_device)
    updated_target_state = torch_model(slice_indices, slice_values)
    updated_target_state = updated_target_state.clone()

    assert not torch.equal(updated_target_state, saved_target_state)
    assert torch.equal(updated_target_state[:, slice_indices], slice_values)

    ### Restore updated values
    restored_target_state = torch_model(slice_indices, saved_slice_values)
    assert torch.equal(restored_target_state, saved_target_state)
    assert not torch.equal(updated_target_state, restored_target_state)

    ###
    ### Convert to CoreML model
    ###

    model_inputs = (
        slice_indices,
        slice_values,
    )
    batch_size_dim = Dim.AUTO(min=1)
    indices_dim = Dim("indices", min=1)
    dynamic_shapes = {
        'slice_indices': (indices_dim, ),
        'slice_values': (batch_size_dim, indices_dim, Dim.AUTO, Dim.AUTO, ),
    }
    aten_program = torch_export_model(torch_model, model_inputs, dynamic_shapes)

    ###
    ### Convert the CoreML model
    ###

    state_desc = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=target_state.shape,
            ),
            name='target_state',
        ),
    ]
    coreml_model = coreml_convert_model(aten_program,
                                        states=state_desc,
                                        )

    # coreml_model.save('pytorch_to_coreml_model.mlpackage')

    ###
    ### Run the CoreML model to replicate PyTorch inference
    ###

    # coreml_model: MLModel = MLModel('pytorch_to_coreml_model.mlpackage')

    slice_indices_val = slice_indices.cpu().to(dtype=torch_coreml_int).numpy()
    slice_values_val = slice_values.cpu().to(dtype=torch_data_dtype).numpy()
    coreml_inputs = {
        'slice_indices': slice_indices_val,
        'slice_values': slice_values_val,
    }
    target_state_val = target_state.cpu().to(dtype=torch_data_dtype).numpy()          ### Bah! use float32 not float16 ?!?
    coreml_updated_target = run_coreml_model(
        coreml_model,
        coreml_inputs,
        state_kv=('target_state', target_state_val),
    )

    assert not np.array_equal(coreml_updated_target['index_copy'], saved_target_state.cpu().to(dtype=torch_data_dtype).numpy())
    assert np.array_equal(coreml_updated_target['index_copy'][:, slice_indices.cpu().numpy()],
                          slice_values.cpu().to(dtype=torch_data_dtype).numpy())


    print(coreml_updated_target)
