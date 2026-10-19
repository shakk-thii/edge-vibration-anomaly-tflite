"""Evaluate FP32 vs INT8: anomaly detection quality, size, latency, memory, bandwidth."""
import os, time, json, platform, numpy as np, tensorflow as tf, psutil
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from dataset import make_window, band_features, FS, N

d = np.load("data.npz"); Xva, Xtn, Xtf = d["Xva"], d["Xtn"], d["Xtf"]
mu, sd, kinds, sev = d["mu"], d["sd"], d["kinds"], d["sev"]
print(f"Python {platform.python_version()} | TensorFlow {tf.__version__} | {platform.system()} {platform.release()} | CPU threads=1")
print("CPU:", [l.split(':')[1].strip() for l in open('/proc/cpuinfo') if 'model name' in l][0])

# ---- inference wrappers ------------------------------------------------------
keras_model = tf.keras.models.load_model("ae_fp32.keras")
keras_fn = tf.function(lambda x: keras_model(x, training=False))

def tflite_runner(path):
    it = tf.lite.Interpreter(model_path=path, num_threads=1); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    qi, qo = i["quantization"], o["quantization"]
    def run(x):                                       # x: (96,) float32 -> reconstruction float32
        if i["dtype"] == np.int8:
            x = np.clip(np.round(x / qi[0] + qi[1]), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], x[None]); it.invoke()
        y = it.get_tensor(o["index"])[0]
        return (y.astype(np.float32) - qo[1]) * qo[0] if o["dtype"] == np.int8 else y
    arena = sum(int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize for t in it.get_tensor_details())
    return run, arena

def keras_run(x): return keras_fn(tf.constant(x[None])).numpy()[0]
runners = {"FP32 Keras (baseline)": (keras_run, None),
           "FP32 TFLite": tflite_runner("ae_fp32.tflite"),
           "INT8 TFLite (edge)": tflite_runner("ae_int8.tflite")}
sizes = {"FP32 Keras (baseline)": os.path.getsize("ae_fp32.keras"),
         "FP32 TFLite": os.path.getsize("ae_fp32.tflite"), "INT8 TFLite (edge)": os.path.getsize("ae_int8.tflite")}

mse = lambda run, X: np.array([np.mean((x - run(x)) ** 2) for x in X])
res = {}
for name, (run, arena) in runners.items():
    ev, en, ef = mse(run, Xva), mse(run, Xtn), mse(run, Xtf)
    theta = float(np.percentile(ev, 99))                  # threshold from healthy validation data
    y = np.r_[np.zeros(len(en)), np.ones(len(ef))]; s = np.r_[en, ef]; p = (s > theta).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    # latency: warm-up 100, then 2000 timed single-window inferences
    x0 = Xtn[0]
    for _ in range(100): run(x0)
    lat = []
    for k in range(2000):
         t = time.perf_counter(); run(Xtn[k % len(Xtn)]); lat.append((time.perf_counter() - t) * 1e3)
    res[name] = dict(theta=theta, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                      acc=float((tp + tn) / len(y)), prec=float(precision_score(y, p)), rec=float(recall_score(y, p)),
                      f1=float(f1_score(y, p)), auc=float(roc_auc_score(y, s)), lat_mean=float(np.mean(lat)),
                      lat_p99=float(np.percentile(lat, 99)), fps=float(1e3 / np.mean(lat)),
                      size_kb=sizes[name] / 1024, arena_kb=None if arena is None else arena / 1024,
                      e_val=ev.tolist(), e_norm=en.tolist(), e_fault=ef.tolist())
    r = res[name]
    print(f"\n=== {name} ===\nsize {r['size_kb']:.2f} KB | theta(99th pct val MSE)={theta:.4f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp} | acc={r['acc']:.4f} prec={r['prec']:.4f} recall={r['rec']:.4f} F1={r['f1']:.4f} AUC={r['auc']:.4f}")
    print(f"latency mean={r['lat_mean']:.4f} ms p99={r['lat_p99']:.4f} ms throughput={r['fps']:.0f} inf/s (n=2000, after 100 warm-up)")
    if arena: print(f"estimated tensor arena (upper bound, no buffer reuse): {arena/1024:.2f} KB")


# per-fault recall at INT8 operating point
run8 = runners["INT8 TFLite (edge)"][0]; th8 = res["INT8 TFLite (edge)"]["theta"]; e8 = np.array(res["INT8 TFLite (edge)"]["e_fault"])
print("\nINT8 detection rate by fault type / severity:")
for k in ("outer", "inner", "ball"):
    print(f" {k:5s}", " ".join(f"sev {s}: {np.mean(e8[(kinds==k)&(sev==s)]>th8)*100:5.1f}%" for s in (0.5, 1.0, 1.5)))
agree = np.mean((np.array(res["FP32 TFLite"]["e_fault"]) > res["FP32 TFLite"]["theta"]) == (e8 > th8))
print(f"FP32 vs INT8 decision agreement on fault windows: {agree*100:.1f}%")

# ---- core interpreter latency (pre-quantised input; isolates the TFLite kernel cost) ----
print("\n=== CORE INTERPRETER LATENCY (set_tensor + invoke + get_tensor, n=2000, 100 warm-up) ===")
for name, path in (("FP32 TFLite", "ae_fp32.tflite"), ("INT8 TFLite (edge)", "ae_int8.tflite")):
    it = tf.lite.Interpreter(model_path=path, num_threads=1); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    x = (np.clip(np.round(Xtn[0] / i["quantization"][0] + i["quantization"][1]), -128, 127).astype(np.int8)
         if i["dtype"] == np.int8 else Xtn[0])[None]
    for _ in range(100): it.set_tensor(i["index"], x); it.invoke(); it.get_tensor(o["index"])
    lat = []
    for _ in range(2000):
        t = time.perf_counter(); it.set_tensor(i["index"], x); it.invoke(); it.get_tensor(o["index"])
        lat.append((time.perf_counter() - t) * 1e3)
    res[name]["core_ms"] = float(np.mean(lat)); res[name]["core_p99"] = float(np.percentile(lat, 99))
    print(f"{name:20s} mean {np.mean(lat)*1000:7.2f} us   p99 {np.percentile(lat,99)*1000:7.2f} us")
sc = lambda n: np.r_[res[n]["e_norm"], res[n]["e_fault"]]
print(f"Reconstruction-error score deviation INT8 vs FP32: mean |dMSE| = {np.mean(np.abs(sc('INT8 TFLite (edge)')-sc('FP32 TFLite'))):.4f}, Pearson r = {np.corrcoef(sc('INT8 TFLite (edge)'), sc('FP32 TFLite'))[0,1]:.5f}")

# ---- end-to-end pipeline latency (raw window -> FFT -> INT8 AE -> decision) -----
rng = np.random.default_rng(7); wins = [make_window(rng) for _ in range(300)]
for w in wins[:50]: run8((band_features(w) - mu) / sd)
tt = []
for w in wins:
    t = time.perf_counter(); f = ((band_features(w) - mu) / sd).astype(np.float32)
    e = np.mean((f - run8(f)) ** 2); _ = e > th8; tt.append((time.perf_counter() - t) * 1e3)
print(f"\nEnd-to-end (FFT 3x4096 + 32-band features + INT8 inference + decision): mean {np.mean(tt):.3f} ms   p99 {np.percentile(tt,99):.3f} ms")
res["e2e_ms"] = float(np.mean(tt))

# ---- bandwidth: stream raw vs. edge decision --------------------------------
raw_bps = 3 * FS * 2; win_s = N / FS; msg_b = 16
print(f"Raw 3-axis int16 stream: {raw_bps/1024:.1f} KB/s = {raw_bps*86400/1e9:.2f} GB/day | edge result msg ({msg_b} B / {win_s:.3f} s): {msg_b/win_s:.1f} B/s = {msg_b/win_s*86400/1e6:.2f} MB/day | reduction {100*(1-msg_b/win_s/raw_bps):.3f} %")
res["raw_bps"], res["edge_bps"] = raw_bps, msg_b / win_s

# ---- live-style anomaly classification log with millisecond timestamps ---------
print("\n=== STREAMING ANOMALY LOG (INT8 TFLite on edge node, theta=%.4f) ===" % th8)
print("timestamp(ms) win    MSE      decision ground-truth")
rng = np.random.default_rng(11); t0 = time.time(); seq = [None]*6 + ["outer"]*5 + [None]*3 + ["inner"]*5 + ["ball"]*4
for i, k in enumerate(seq):
    w = make_window(rng, k, 1.0); f = ((band_features(w) - mu) / sd).astype(np.float32)
    e = float(np.mean((f - run8(f)) ** 2))
    print(f"{int((t0 + i*N/FS)*1000)} {i:3d} {e:8.4f} {'ANOMALY' if e>th8 else 'normal ':8s} {k or 'healthy'}")
json.dump(res, open("results.json", "w"))
