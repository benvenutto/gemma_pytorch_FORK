from typing import Tuple
import torch

import coremltools as ct
from coremltools.models.model import MLState

import numpy as np


@torch.inference_mode()
def torch_export_model(torch_model, model_inputs, dynamic_shapes):
    aten_program = torch.export.export(
        torch_model,
        model_inputs,
        dynamic_shapes=dynamic_shapes
    )
    simplified_aten_program = aten_program.run_decompositions(decomp_table={})
    return simplified_aten_program


def coreml_convert_model(aten_program, states=None):
    coreml_model = ct.convert(
        aten_program,
        source='pytorch',
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        states=states,
    )
    return coreml_model


def run_coreml_model(coreml_model, coreml_inputs, state_kv=None|Tuple[str, np._typing.NDArray[np.float16]]):
    if state_kv is not None:
        model_state: MLState = coreml_model.make_state()
        model_state.write_state(name=state_kv[0], value=state_kv[1])
    else:
        model_state: MLState = None

    preds = coreml_model.predict(coreml_inputs, state=model_state)
    return preds