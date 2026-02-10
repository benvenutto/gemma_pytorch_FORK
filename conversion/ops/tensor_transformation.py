from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op, _TORCH_OPS_REGISTRY
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb


del _TORCH_OPS_REGISTRY['multinomial']      ### Replace shipped version, which is broken getting parameters

@register_torch_op
def index_copy(context, node):
    """ Update target tensor at the given indices into dimension `dim`, using the corresponding slices of values
    """
    x, dim, slice_indices, slice_values = _get_inputs(context, node, expected=4)
    x = mb.cast(x=x, dtype='fp16')
    x = mb.scatter(data=x, indices=slice_indices, updates=slice_values, axis=dim, mode='update', name=node.name)
    context.add(x)

@register_torch_op
def multinomial(context, node):
    x, num_samples, replacement = _get_inputs(context, node, expected=3)
    if num_samples.val is None:
        raise ValueError("In torch.multinomial op, num_samples must be const")
    else:
        num_samples = num_samples.val
    if num_samples > 1:
        if replacement.val is None or replacement.val != True:
            raise ValueError("When num_samples is larger than 1, only replacement=True is supported.")
    # Based on PyTorch documentations, the input to `torch.multinomial` is probability, not logit.
    # Simon: actually the documentation says it doesn't need to sum to 0 across row, so either probs or weights
    x = mb.random_categorical(x=x, size=num_samples, mode="probs", name=node.name)
    context.add(x)

