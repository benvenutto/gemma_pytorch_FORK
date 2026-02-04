from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb

@register_torch_op
def index_copy(context, node):
    """ Update target tensor at the given indices into dimension `dim`, using the corresponding slices of values
    """
    x, dim, slice_indices, slice_values = _get_inputs(context, node, expected=4)
    x = mb.cast(x=x, dtype='fp16')
    x = mb.scatter(data=x, indices=slice_indices, updates=slice_values, axis=dim, mode='update', name=node.name)
    context.add(x)