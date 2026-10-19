"""Generate report figures from results.json (run after evaluate.py)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("results.json"))
d = np.load("data.npz"); kinds, sev = d["kinds"], d["sev"]
r8 = res["INT8 TFLite (edge)"]; th = r8["theta"]

# 1. Reconstruction-error distribution
bins = np.geomspace(0.3, 30, 80)
plt.figure(figsize=(8, 3.6))
plt.hist(r8["e_norm"], bins, color="#2e7d4f", alpha=.75, label=f"Healthy test windows (n={len(r8['e_norm'])})")
plt.hist(r8["e_fault"], bins, color="#c0392b", alpha=.7, label=f"Faulty test windows (n={len(r8['e_fault'])})")
plt.axvline(th, ls="--", c="k", label=f"threshold θ = {th:.3f} (99th pct of healthy val.)")
plt.xscale("log"); plt.xlabel("Reconstruction MSE (INT8 autoencoder, log scale)"); plt.ylabel("windows")
plt.title("Reconstruction-error distribution"); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("fig_error_hist.png", dpi=150); plt.close()

# 2. Confusion matrix
cm = np.array([[r8["tn"], r8["fp"]], [r8["fn"], r8["tp"]]])
plt.figure(figsize=(3.8, 3.4)); plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["normal", "anomaly"]); plt.yticks([0, 1], ["normal", "anomaly"])
plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("INT8 confusion matrix"); plt.tight_layout()
plt.savefig("fig_confusion.png", dpi=150); plt.close()

# 3. Size, arena and recall by severity
names = ["FP32 Keras (baseline)", "FP32 TFLite", "INT8 TFLite (edge)"]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
sz = [res[n]["size_kb"] for n in names]
ax[0].bar(["FP32 Keras", "FP32 TFLite", "INT8 TFLite"], sz, color=["#999", "#5b8fd6", "#2a9d5c"])
ax[0].axhline(80, ls="--", c="r"); ax[0].set_title("Model size"); ax[0].set_ylabel("KB")
ar = [res[n]["arena_kb"] for n in names[1:]]
ax[1].bar(["FP32 TFLite", "INT8 TFLite"], ar, color=["#5b8fd6", "#2a9d5c"]); ax[1].set_title("Est. tensor arena (KB)")
e8 = np.array(r8["e_fault"]); x = np.arange(3)
for k_i, (k, c) in enumerate((("outer", "#c0392b"), ("inner", "#8e44ad"), ("ball", "#e69f00"))):
    rec = [np.mean(e8[(kinds == k) & (sev == s)] > th) * 100 for s in (0.5, 1.0, 1.5)]
    ax[2].bar(x + (k_i - 1) * 0.25, rec, 0.25, color=c, label=k)
ax[2].set_xticks(x, ["mild", "medium", "severe"]); ax[2].set_ylabel("detection rate (%)")
ax[2].set_title("INT8 recall by fault severity"); ax[2].legend(fontsize=7)
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig_summary.png", dpi=150); plt.close()
print("saved fig_error_hist.png, fig_confusion.png, fig_summary.png")
