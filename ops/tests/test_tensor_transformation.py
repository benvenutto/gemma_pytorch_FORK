from typing import Final

import torch
from coremltools.models import MLModel
from torch import nn
from torch.export import Dim

import coremltools as ct

from .test_util import torch_export_model, coreml_convert_model, run_coreml_model


class TestTensorIndexCopySlice(nn.Module):
    SEQUENCE_DIM: Final[int] = 1

    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target, persistent=False)

    def forward(self, slice_indices: torch.Tensor, slice_values: torch.Tensor) -> torch.Tensor:
        self.target.index_copy_(TestTensorIndexCopySlice.SEQUENCE_DIM, slice_indices, slice_values)
        return self.target

    # def forward(self, slice_indices: torch.Tensor, slice_values: torch.Tensor) -> torch.Tensor:
    #     # slice_index = slice_indices[0]
    #     # slice_value = slice_values[:, slice_index]
    #     slice_value = slice_values[:, 0]
    #     slice_value =  slice_value.unsqueeze(TestTensorIndexCopySlice.SEQUENCE_DIM)
    #     self.target = self.target * slice_value     ### with broadcast
    #     return self.target


def test_index_copy():

    ###
    ### Setup PyTorch model
    ###

    torch_device = torch.device('mps')
    torch_data_dtype = torch.float32
    torch_state_dtype = torch.float16
    torch_index_dtype = torch.int64
    torch_coreml_int = torch.int32

    target = torch.rand((8, 24, 6, 4), dtype=torch_state_dtype, device=torch_device)
    torch_model = TestTensorIndexCopySlice(target).to(device=torch_device)

    # original_target = target.clone()
    slice_indices = torch.tensor([2, 3, 5], dtype=torch_index_dtype, device=torch_device)
    # original_target_slice = original_target[:, slice_indices]   ### dim=1
    slice_values = torch.rand((8, 3, 6, 4), dtype=torch_state_dtype, device=torch_device)

    ### Update with the slice
    updated_target = torch_model(slice_indices, slice_values)
    # assert not torch.equal(updated_target, original_target_slice)

    # ### Update with the original values that were in the slice
    # restored_target = torch_model(slice_indices, original_target_slice)
    # assert torch.equal(restored_target, original_target)

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
                shape=target.shape,
            ),
            name='target',
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

    coreml_inputs = {
        'slice_indices': slice_indices.cpu().to(dtype=torch_coreml_int).numpy(),
        'slice_values': slice_values.cpu().to(dtype=torch_data_dtype).numpy(),
    }
    target_state = target.cpu().to(dtype=torch_data_dtype).numpy()  ### Bah! use float32 not float16 ?!?
    coreml_updated_target = run_coreml_model(
        coreml_model,
        coreml_inputs,
        state_kv=('target', target_state),
    )

    print(coreml_updated_target)
