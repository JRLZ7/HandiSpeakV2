import onnxruntime as ort, numpy as np, time

def bench(model_path, iters=50, B=16, T=20, F=354):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(B, T, F).astype(np.float32)
    # warmup
    for _ in range(5): sess.run(None, {"input": x})
    t0 = time.time()
    for _ in range(iters): sess.run(None, {"input": x})
    dt = (time.time() - t0) / iters
    return dt

fp32 = "models/TSF_200words_fp32quantized.onnx" # path to fp32 file
int8 = "models/TSF_200words_int8quantized.onnx" # path to int8 file
print("FP32 ms:", bench(fp32)*1000)
print("INT8 ms:", bench(int8)*1000)