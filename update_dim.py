from argparse import ArgumentParser

import onnx
from onnx import TensorProto, helper
from onnx.tools.update_model_dims import update_inputs_outputs_dims
from onnxconverter_common.float16 import convert_float_to_float16


def update_input_tensor_dtype(
    model: onnx.ModelProto,
    dtype: int,
    batch_name: str,
    size: int,
):
    input_name = model.graph.input[0].name
    new_input_tensor = helper.make_tensor_value_info(
        input_name,
        dtype,
        [batch_name, 3, size, size],  # Shape: [batch, channels, height, width]
    )
    model.graph.input.pop(0)
    model.graph.input.insert(0, new_input_tensor)

    return model


def insert_cast(model: onnx.ModelProto, dtype: int):
    input_name = model.graph.input[0].name
    cast_output = 'cast_0'
    cast_node = helper.make_node(
        'Cast',
        inputs=[input_name],
        outputs=[cast_output],
        to=dtype,
    )
    model.graph.node[0].input[0] = cast_output
    model.graph.node.insert(0, cast_node)

    return model


def insert_cast_divide_255(model: onnx.ModelProto, dtype: int):
    input_name = model.graph.input[0].name
    cast_output = 'cast_0'
    div_output = 'div_255_output'
    cast_node = helper.make_node(
        'Cast',
        inputs=[input_name],
        outputs=[cast_output],
        to=dtype,
    )
    const_name = 'const_255'
    const_tensor = helper.make_tensor(const_name, dtype, [], [255.0])
    model.graph.initializer.append(const_tensor)
    div_node = helper.make_node(
        'Div',
        inputs=[cast_output, const_name],
        outputs=[div_output],
    )
    model.graph.node[0].input[0] = div_output
    model.graph.node.insert(0, div_node)
    model.graph.node.insert(0, cast_node)

    return model


def main(args):
    print(f'Loading model from {args.input}...')

    if args.u8_head and args.no_cast:
        print('--u8-head with force create cast layer, --no-cast is ignored.')

    model = onnx.load(args.input)
    dtype = TensorProto.FLOAT

    if args.half:
        model = convert_float_to_float16(model)
        dtype = TensorProto.FLOAT16

    if args.u8_head:
        insert_cast_divide_255(model, dtype=dtype)
        dtype = TensorProto.UINT8
    elif args.half and not args.no_cast:
        # If model is converted to FP16, add Cast to FLOAT input
        insert_cast(model, dtype=dtype)
        dtype = TensorProto.FLOAT

    output_node = model.graph.output
    model = update_inputs_outputs_dims(
        model,
        {'images': [args.batch_name, 3, args.size, args.size]},
        {
            node.name: [args.batch_name, 'num_class+4', 'num_boxes']
            for node in output_node
        },
    )
    update_input_tensor_dtype(model, dtype, args.batch_name, args.size)
    onnx.save(model, args.output)
    print(f'Saved updated model to {args.output}')


if __name__ == '__main__':
    parser = ArgumentParser('update YOLO onnx dims')
    parser.add_argument('input', type=str, help='input onnx path')
    parser.add_argument('output', type=str, help='output onnx path')
    parser.add_argument(
        '--size',
        type=int,
        required=False,
        default=640,
        help='model size, default 640',
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='convert model to fp16?',
    )
    parser.add_argument(
        '--no-cast',
        action='store_true',
        help='do not add onnx cast layer if model input is FP16',
    )
    parser.add_argument(
        '--batch-name',
        required=False,
        default='batch',
        help='batch dim alias name, default `batch`',
    )
    parser.add_argument(
        '--u8-head',
        action='store_true',
        help='add uint8 input head with cast and division by 255',
    )

    args = parser.parse_args()
    main(args)
