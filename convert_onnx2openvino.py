import openvino as ov
import argparse
import os
import glob


def convert(core, onnx_path, output_path):
    model_onnx = core.read_model(model=onnx_path)
    compiled_model_onnx = core.compile_model(model=model_onnx, device_name="CPU")
    output_file = os.path.join(output_path, os.path.basename(onnx_path).split('.')[0] + '.xml')
    ov.save_model(model_onnx, output_model=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=str, help="Path to the ONNX file")
    parser.add_argument("--output-path", type=str, help="Output directory")
    args = parser.parse_args()
    onnx_path = args.onnx_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    core = ov.Core()
    if os.path.isdir(onnx_path):
        for sub_path in glob.glob(os.path.join(onnx_path, "*.onnx")):
            convert(core, sub_path, output_path)
    else:
        convert(core, onnx_path, output_path)
