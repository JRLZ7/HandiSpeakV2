# pt_to_onnx_int8_transformer.py
import argparse, os, torch, torch.nn as nn, numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# python3 transformer/tsf_quantize.py models/[relative path to model.pt] \
#  --input-dim 354 --model-dim 256 --num-heads 8 --num-layers 4 --num-classes [vocab size] --seq-len 20

# ----- Your Transformer model -----
class HandiSpeakTransformer(nn.Module):
    def __init__(self, input_dim=354, model_dim=256, num_heads=8, num_layers=4, num_classes=50, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)     # [B, T, D]
        x = self.transformer(x)    # [B, T, D]
        x = x.mean(dim=1)          # global avg pool over time
        return self.classifier(x)  # [B, C]

def file_mb(path): return os.path.getsize(path) / (1024*1024)

def main():
    p = argparse.ArgumentParser("Convert Transformer .pt to ONNX + INT8 (dynamic) with ORT sanity check")
    p.add_argument("pt", help=".pt checkpoint (state_dict or full module)")
    p.add_argument("--onnx", default=None, help="Output ONNX path (default: infer from .pt)")
    p.add_argument("--onnx-int8", default=None, help="Output INT8 ONNX path (default: add _int8)")
    # Model hyperparams (needed if .pt is a state_dict)
    p.add_argument("--input-dim", type=int, default=354)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    base = os.path.splitext(args.pt)[0]
    onnx_path = args.onnx or f"{base}.onnx"
    onnx_int8_path = args.onnx_int8 or f"{os.path.splitext(onnx_path)[0]}_int8.onnx"

    # 1) Load .pt
    obj = torch.load(args.pt, map_location="cpu")
    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = HandiSpeakTransformer(
            input_dim=args.input_dim,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            dropout=args.dropout,
        )
        model.load_state_dict(obj)
    model.eval()

    # 2) Export to ONNX (dynamic batch + seq_len)
    dummy = torch.randn(1, args.seq_len, args.input_dim)
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

    # 3) Dynamic INT8 quantization
    # For Transformers, dynamic quantization mainly targets Linear/MatMul weights.
    quantize_dynamic(
        model_input=onnx_path,
        model_output=onnx_int8_path,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8
    )
    print(f"✅ INT8 ONNX saved: {onnx_int8_path}  ({file_mb(onnx_int8_path):.2f} MB)")

    # 4) ORT sanity check
    sess = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(2, args.seq_len, args.input_dim).astype(np.float32)
    y = sess.run(None, {"input": x})[0]
    print(f"🔎 ORT test: input {x.shape} → output {y.shape}")

if __name__ == "__main__":
    main()
