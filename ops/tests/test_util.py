import copy
import torch
import coremltools as ct
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


def coreml_convert_model(aten_program):
    coreml_model = ct.convert(
        aten_program,
        source='pytorch',
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        # states=[
        #         ct.StateType(
        #             wrapped_type=ct.TensorType(
        #                 # shape=kv_cache_size,
        #                 dtype=np.float16,
        #             ),
        #             name='k_caches',
        #         ),
        # ],
    )
    return coreml_model


def run_coreml_model(coreml_model, coreml_inputs):
    coreml_model.predict(coreml_inputs)