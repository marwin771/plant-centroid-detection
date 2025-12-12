import os
import onnx
from ultralytics import YOLO
from onnxconverter_common import float16


def get_file_size_mb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)


def export_and_quantize(model_path):
    print("Loading YOLOv8 model...\n")
    model = YOLO(model_path)

    # PT model info
    pt_size = get_file_size_mb(model_path)
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"PT model loaded: {model_path}")

    # Export ONNX (FP32)
    print("\nExporting ONNX FP32...")
    onnx_path = model.export(format="onnx")
    onnx_size = get_file_size_mb(onnx_path)

    print(f"FP32 ONNX saved: {onnx_path}")

    # Convert to FP16
    print("\nConverting to FP16 ONNX...")
    model_fp32 = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32)

    fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
    onnx.save(model_fp16, fp16_path)
    fp16_size = get_file_size_mb(fp16_path)

    # Final report
    print("\n==============================")
    print("        FINAL REPORT")
    print("==============================")
    print(f"PT Model (.pt):")
    print(f"  Path: {model_path}")
    print(f"  Size: {pt_size:.2f} MB")
    print(f"  Total Params: {total_params:,}")
    print(f"  Trainable Params: {trainable_params:,}")

    print("\nONNX FP32 Model:")
    print(f"  Path: {onnx_path}")
    print(f"  Size: {onnx_size:.2f} MB")

    print("\nONNX FP16 Model:")
    print(f"  Path: {fp16_path}")
    print(f"  Size: {fp16_size:.2f} MB")

    print("\nDone.")
    print("==============================")



if __name__ == "__main__":
    export_and_quantize("runs/detect/train/weights/best.pt")
