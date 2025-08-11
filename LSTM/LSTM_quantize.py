# pt_to_onnx_int8.py
import argparse, os, torch, torch.nn as nn, numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# use: python3 LSTM/LSTM_quantize.py models/[relative path to model .pt] --hidden-size 256 --num-classes [vocab size] --seq-len 20

# ---- Your model (must match training) ----
class ASLClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)   # [B, T, 2H]
        out = out[:, -1, :]     # last timestep
        return self.fc(out)     # [B, C]

def file_mb(path): return os.path.getsize(path) / (1024 * 1024)

def main():
    p = argparse.ArgumentParser("Convert PyTorch .pt to ONNX and INT8-quantized ONNX")
    p.add_argument("pt", help=".pt checkpoint (state_dict or full model)")
    p.add_argument("--onnx", default=None, help="Output ONNX path (default: infer from .pt)")
    p.add_argument("--onnx-int8", default=None, help="Output INT8 ONNX path (default: add _int8)")
    # Model hyperparams (only used if .pt is a state_dict)
    p.add_argument("--input-size", type=int, default=354)
    p.add_argument("--hidden-size", type=int, default=256)  # use 256 if we're standardizing
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-classes", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args()

    base = os.path.splitext(args.pt)[0]
    onnx_path = args.onnx or f"{base}.onnx"
    onnx_int8_path = args.onnx_int8 or f"{os.path.splitext(onnx_path)[0]}_int8.onnx"

    # 1) Load .pt
    obj = torch.load(args.pt, map_location="cpu")
    if isinstance(obj, nn.Module):
        model = obj
    else:
        # assume state_dict
        model = ASLClassifier(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
        )
        model.load_state_dict(obj)
    model.eval()

    # 2) Export to ONNX (dynamic batch and seq_len)
    dummy = torch.randn(1, args.seq_len, args.input_size)
    torch.onnx.export(
        model, dummy, onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"},
                      "logits": {0: "batch_size"}}
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"✅ ONNX exported: {onnx_path}  ({file_mb(onnx_path):.2f} MB)")

    # 3) Dynamic quantization (INT8 weights)
    quantize_dynamic(
        model_input=onnx_path,
        model_output=onnx_int8_path,
        per_channel=True,                    # better accuracy for Linear
        reduce_range=False,
        weight_type=QuantType.QInt8
    )
    print(f"✅ INT8 ONNX saved: {onnx_int8_path}  ({file_mb(onnx_int8_path):.2f} MB)")

    # 4) Quick sanity check with ORT
    sess = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(2, args.seq_len, args.input_size).astype(np.float32)
    y = sess.run(None, {"input": x})[0]
    print(f"🔎 ORT test: input {x.shape} → output {y.shape}")

if __name__ == "__main__":
    main()
