#!/usr/bin/env python3
"""
Quantize TSF-CTC model to ONNX FP32 and INT8 (dynamic).
Usage:
  python ctc50/ctc_quantize.py --ckpt models/TSF_CTC_best_WER19.68.pt --onnx_fp32 models/TSF_CTC_best_WER19.68.onnx --onnx_int8 models/TSF_CTC_best_WER19.68_int8.onnx --num_classes 51 --seq_len 20 --runs 50
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from ctc_model import TSFCTCEncoder

def build_model_from_ckpt(ckpt_path: str, num_classes: int):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    feature_dim   = cfg.get("feature_dim", 354)
    d_model       = cfg.get("d_model", 256)
    nhead         = cfg.get("nhead", 8)
    num_layers    = cfg.get("layers", 4)
    dim_ff        = cfg.get("ff", 512)
    dropout       = cfg.get("dropout", 0.1)

    model = TSFCTCEncoder(
        feature_dim=feature_dim,
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
    )
    sd = ckpt["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, feature_dim

class ExportWrapper(nn.Module):
    """Wraps (inputs[T,B,F], lengths[B]) → logits[B,T,C] to match your forward."""
    def __init__(self, model: TSFCTCEncoder):
        super().__init__()
        self.model = model
    def forward(self, x, input_lengths):
        # x: [B, T, F], input_lengths: [B]
        return self.model(x, input_lengths)

def export_onnx(model: nn.Module, feature_dim: int, seq_len: int, onnx_fp32: str):
    dummy_B = 2
    dummy_T = seq_len
    x = torch.randn(dummy_B, dummy_T, feature_dim, dtype=torch.float32)
    lengths = torch.full((dummy_B,), dummy_T, dtype=torch.int32)

    wrapper = ExportWrapper(model)
    torch.onnx.export(
        wrapper,
        (x, lengths),
        onnx_fp32,
        export_params=True,
        do_constant_folding=True,
        input_names=["inputs", "input_lengths"],
        output_names=["logits"],
        dynamic_axes={
            "inputs": {0: "batch", 1: "time"},
            "input_lengths": {0: "batch"},
            "logits": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )
    print(f"✅ ONNX FP32 exported: {onnx_fp32}")

def quantize_int8(onnx_fp32: str, onnx_int8: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=onnx_fp32,
        model_output=onnx_int8,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"✅ INT8 ONNX saved: {onnx_int8}")

def ort_sanity_and_timing(onnx_path: str, feature_dim: int, seq_len: int, runs: int):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    B, T, F = 2, seq_len, feature_dim
    x = np.random.randn(B, T, F).astype(np.float32)
    lengths = np.full((B,), T, dtype=np.int32)

    # Sanity
    out = sess.run(["logits"], {"inputs": x, "input_lengths": lengths})[0]
    print(f"🔎 ORT test: input ({B},{T},{F}) → output {tuple(out.shape)}")

    # Timing
    # Warmup
    for _ in range(10):
        sess.run(None, {"inputs": x, "input_lengths": lengths})
    t0 = time.time()
    for _ in range(runs):
        sess.run(None, {"inputs": x, "input_lengths": lengths})
    dt = (time.time() - t0) * 1000.0 / runs
    print(f"⏱  Avg latency over {runs} runs: {dt:.3f} ms")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx_fp32", required=True)
    ap.add_argument("--onnx_int8", required=True)
    ap.add_argument("--num_classes", type=int, default=51, help="vocab+blank")
    ap.add_argument("--seq_len", type=int, default=20, help="dummy T for export & ORT test")
    ap.add_argument("--runs", type=int, default=50, help="timing iterations per model")
    args = ap.parse_args()

    model, feature_dim = build_model_from_ckpt(args.ckpt, args.num_classes)
    export_onnx(model, feature_dim, args.seq_len, args.onnx_fp32)
    quantize_int8(args.onnx_fp32, args.onnx_int8)

    print("\n— ORT FP32 —")
    ort_sanity_and_timing(args.onnx_fp32, feature_dim, args.seq_len, args.runs)
    print("\n— ORT INT8 (dynamic) —")
    ort_sanity_and_timing(args.onnx_int8, feature_dim, args.seq_len, args.runs)

if __name__ == "__main__":
    main()
