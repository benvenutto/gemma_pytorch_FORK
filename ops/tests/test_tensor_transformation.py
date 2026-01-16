from typing import Final
import torch
from torch import nn
from torch.export import Dim

from .test_util import torch_export_model, coreml_convert_model, run_coreml_model


class TestTensorIndexCopySlice(nn.Module):
    SEQUENCE_DIM: Final[int] = 1

    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    def forward(self, slice_indices: torch.Tensor, slice_values: torch.Tensor) -> torch.Tensor:
        self.target.index_copy_(TestTensorIndexCopySlice.SEQUENCE_DIM, slice_indices, slice_values)
        return self.target

def test_index_copy():

    ###
    ### Setup PyTorch model
    ###

    torch_device = torch.device('mps')
    torch_data_dtype = torch.float32
    torch_index_dtype = torch.int64

    target = torch.rand((8, 24, 6, 4), dtype=torch_data_dtype, device=torch_device)
    torch_model = TestTensorIndexCopySlice(target).to(device=torch_device, dtype=torch_data_dtype)

    original_target = target.clone()
    slice_indices = torch.tensor([2, 3, 5], dtype=torch_index_dtype, device=torch_device)
    original_target_slice = original_target[:, slice_indices]   ### dim=1
    slice_values = torch.rand((8, 3, 6, 4), dtype=torch_data_dtype, device=torch_device)

    ### Update with the slice
    updated_target = torch_model(slice_indices, slice_values)
    assert not torch.equal(updated_target, original_target_slice)

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
    batch_size_dim = Dim("batch_size", min=1, max=256)
    indices_dim = Dim("indices", min=1)
    dynamic_shapes = {
        'slice_indices': (indices_dim, ),
        'slice_values': (batch_size_dim, indices_dim, Dim.AUTO, Dim.AUTO, ),
    }
    aten_program = torch_export_model(torch_model, model_inputs, dynamic_shapes)

    ###
    ### Convert the CoreML model
    ###

    coreml_model = coreml_convert_model(aten_program, )

    ###
    ### Run the CoreML model to replicate PyTorch inference
    ###

    coreml_inputs = {
        'slice_values': slice_values,
        'slice_indices': slice_indices,
    }
    coreml_updated_target = run_coreml_model(coreml_model, coreml_inputs)

    print(coreml_updated_target)

