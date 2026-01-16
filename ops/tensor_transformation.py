from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb

@register_torch_op
def index_copy(context, node):
    """ Update target tensor at the given indices into dimension `dim`, using the corresponding slices of values
    """
    op_name = node.name
    inputs = _get_inputs(context, node, expected=4)
    target = inputs[0]
    dim = inputs[1]
    slice_indices = inputs[2]
    slice_values = inputs[3]

    all_true = mb.fill(shape=[target.rank], value=True)
    slice_mask = mb.slice_update(x=all_true, update=[False], begin=[dim.val], end=[dim.val + 1])

    all_zero = mb.fill(shape=[target.rank], value=0)

    def _update_target_slice(target, dim, slice_index, slice_value):
        all_zero = mb.fill(shape=[target.rank], value=0)
        pure_index = mb.slice_update(x=all_zero, update=slice_index, begin=[dim.val], end=[dim.val + 1])
        return mb.slice_update(
            x=target,
            update=slice_value,
            begin=pure_index,
            end=pure_index,
            begin_mask=slice_mask,
            end_mask=slice_mask,
            squeeze_mask=slice_mask)

    slice_index = mb.slice_by_size(x=slice_indices, begin=[0], size=[1])
    slice_value = mb.slice_by_index(
        x=slice_values,
        begin=all_zero,     ### HACK: first update slice
        end=all_zero,
        begin_mask=slice_mask,
        end_mask=slice_mask,
        squeeze_mask=slice_mask)

    x_ = _update_target_slice(target, dim, slice_index, slice_value)
    return x_

    # begin_mask = mb.fill(shape=[target.rank], value=True, name='begin_mask')
    # begin_mask = mb.slice_update(x=begin_mask, update=[False], begin=[dim.val], end=[dim.val + 1])
    # end_mask = mb.fill(shape=[target.rank], value=True, name='end_mask')
    # end_mask = mb.slice_update(x=end_mask, update=[False], begin=[dim.val], end=[dim.val + 1])
    # squeeze_mask = mb.fill(shape=[target.rank], value=False, name='squeeze_mask')
    # squeeze_mask = mb.slice_update(x=squeeze_mask, update=[True], begin=[dim.val], end=[dim.val + 1])
    #
    # first_slice = mb.slice_by_size(x=slice_values,
    #                 begin=index_from_zero,
    #                 size=index_to_end)
    # index_from_zero = mb.slice_update(x=index_from_zero, update=first_slice, begin=[dim.val], end=[dim.val + 1], name='current_index')  ### Hack
    # mb.slice_update(x=target,
    #                 update=slice_values,
    #                 begin=index_from_zero,
    #                 end=index_to_end,
    #                 begin_mask=begin_mask,
    #                 end_mask=end_mask,
    #                 squeeze_mask=squeeze_mask,
    #                 name='update_target')
    #
    # # def cond(slice_index, *loop_vars):
    # #     return mb.less(x=slice_index, y=num_slices)
    # #
    # # def body(slice_index, *loop_vars):
    # #     return [slice_index] + loop_vars
    #



    print(f">> in index-copy_() my version ...")
