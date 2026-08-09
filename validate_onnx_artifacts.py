"""Validate deployment ONNX files structurally and through runtime inference."""

import argparse
import numpy as np
import onnx
import onnxruntime as ort

from strategy_config import FEATURE_COLUMNS, MODEL


EXPECTED_OUTPUT_WIDTHS = {
    "best_model_a_live.onnx": 3,
    "best_model_b_live.onnx": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = "outputs/smoke/" if args.smoke else ""

    sample = np.zeros(
        (1, MODEL.lookback, len(FEATURE_COLUMNS)),
        dtype=np.float32,
    )
    for name, expected_width in EXPECTED_OUTPUT_WIDTHS.items():
        path = f"{root}{name}"
        model = onnx.load(path)
        onnx.checker.check_model(model)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: sample})[0]
        expected_shape = (1, expected_width)
        if output.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected output for {path}: {output.shape}; expected {expected_shape}"
            )
        print(f"{path}: valid, output shape={output.shape}")


if __name__ == "__main__":
    main()
