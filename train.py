"""Train a symmetric autoencoder on NORMAL vibration only, export FP32 + INT8 TFLite."""
import os, json, numpy as np, tensorflow as tf
from dataset import build

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)

# ---- data ingestion -------------------------------------------------------
Xtr, Xva, Xtn, Xtf, meta = build(rng)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6              # normalisation from healthy data only
norm = lambda x: ((x - mu) / sd).astype(np.float32)
Xtr, Xva, Xtn, Xtf = map(norm, (Xtr, Xva, Xtn, Xtf))
np.savez("data.npz", Xtr=Xtr, Xva=Xva, Xtn=Xtn, Xtf=Xtf, mu=mu, sd=sd,
         kinds=np.array([m[0] for m in meta]), sev=np.array([m[1] for m in meta]))
print(f"train(normal)={len(Xtr)} val(normal)={len(Xva)} test normal={len(Xtn)} test fault={len(Xtf)} features={Xtr.shape[1]}")

# ---- symmetric autoencoder 96-32-16-8-16-32-96 ------------------------------
L = tf.keras.layers
model = tf.keras.Sequential([
    L.Input((96,)),
    L.Dense(32, activation="relu"), L.Dense(16, activation="relu"), L.Dense(8, activation="relu"),
    L.Dense(16, activation="relu"), L.Dense(32, activation="relu"), L.Dense(96)])
model.compile("adam", "mse")
model.summary(print_fn=print)
hist = model.fit(Xtr, Xtr, validation_data=(Xva, Xva), epochs=80, batch_size=64, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
model.save("ae_fp32.keras")

# ---- TFLite export: FP32 and full-integer INT8 ------------------------------
conv = tf.lite.TFLiteConverter.from_keras_model(model)
open("ae_fp32.tflite", "wb").write(conv.convert())

def rep():                                             # calibration set for INT8 ranges
    for x in Xtr[:300]:
        yield [x[None]]
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
open("ae_int8.tflite", "wb").write(conv.convert())
for f in ("ae_fp32.keras", "ae_fp32.tflite", "ae_int8.tflite"):
    print(f"{f:16s} {os.path.getsize(f)/1024:8.2f} KB")
print("params:", model.count_params())
