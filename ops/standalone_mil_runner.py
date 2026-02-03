from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
import coremltools as ct


# def index_copy(context, node):
@mb.program(input_specs=[mb.TensorSpec(shape=(1, )), mb.TensorSpec(shape=(1, )), mb.TensorSpec(shape=(1, )), mb.TensorSpec(shape=(1, )), ])
def index_copy(target, dim, slice_indices, slice_values):
    """ Update target tensor at the given indices into dimension `dim`, using the corresponding slices of values
    """
    # node_name = node.name
    # inputs = _get_inputs(context, node, expected=4)
    # target = inputs[0]
    # dim = inputs[1]
    # slice_indices = inputs[2]
    # slice_values = inputs[3]

    all_true = mb.fill(shape=[target.rank], value=True)
    slice_mask = mb.slice_update(x=all_true, update=[False], begin=[0], end=[1])

    all_zero = mb.fill(shape=[target.rank], value=0)

    def _update_target_slice(target, dim, slice_index, slice_value):
        all_zero = mb.fill(shape=[target.rank], value=0)
        pure_index = mb.slice_update(x=all_zero, update=slice_index, begin=[0], end=[1])
        c_target = mb.cast(x=target, dtype="fp16")
        return mb.slice_update(
            x=c_target,
            update=slice_value,
            begin=pure_index,
            end=pure_index,
            begin_mask=slice_mask,
            end_mask=slice_mask,
            squeeze_mask=slice_mask,
            name='index_copy'
        )

    slice_index = mb.slice_by_size(x=slice_indices, begin=[0], size=[1])
    slice_value = mb.slice_by_index(
        x=slice_values,
        begin=all_zero,     ### HACK: first update slice
        end=all_zero,
        begin_mask=slice_mask,
        end_mask=slice_mask,
        squeeze_mask=slice_mask)

    x_ = _update_target_slice(target, dim, slice_index, slice_value)
    # context.add(x_)


###
### Hack - run tests without pytest, get normal stack traces
###

if __name__ == "__main__":
    model = ct.convert(index_copy, source='milinternal')
    print(model)