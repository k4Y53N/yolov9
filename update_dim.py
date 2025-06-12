from argparse import ArgumentParser

import onnx
from onnx import TensorProto, helper
from onnx.tools.update_model_dims import update_inputs_outputs_dims
from onnxconverter_common.float16 import convert_float_to_float16


def add_cast_fp16_layer(model: onnx.ModelProto):
    # Assume the first node is the input layer and we want to remove it
    # In ONNX, rather than "removing" an input layer, you would adjust the graph to change inputs

    # Create a Cast node to cast input from float32 to float16
    # Note: You need to know the input name and shape
    input_name = model.graph.input[0].name
    output_name = 'cast_0'
    cast_node = helper.make_node(
        'Cast', inputs=[input_name], outputs=[output_name], to=TensorProto.FLOAT16
    )

    # Modify the graph's first node to take the output of the cast node as its input
    model.graph.node[0].input[0] = output_name

    # Add the cast node to the beginning of the graph
    model.graph.node.insert(0, cast_node)

    return model


def fp32_to_fp16_with_cast(model: onnx.ModelProto, size: int, batch_name: str):
    add_cast_fp16_layer(model)
    input_name = model.graph.input[0].name
    new_input_tensor = helper.make_tensor_value_info(
        input_name,  # Name of the new input
        TensorProto.FLOAT,  # Data type
        [batch_name, 3, size, size],  # Shape: [batch, channels, height, width]
    )
    # Replace the original input with the new one
    # Remove the original input
    model.graph.input.pop(0)
    # Add the new input
    model.graph.input.insert(0, new_input_tensor)

    return model


def main(args):
    model = onnx.load(args.input)
    output_node = model.graph.output
    model = update_inputs_outputs_dims(
        model,
        {'images': [args.batch_name, 3, args.size, args.size]},
        {
            node.name: [args.batch_name, 'num_class+4', 'num_boxes']
            for node in output_node
        },
    )

    if args.half:
        model = convert_float_to_float16(model)

        if not args.no_cast:
            fp32_to_fp16_with_cast(model, args.size, args.batch_name)

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

    args = parser.parse_args()
    main(args)
