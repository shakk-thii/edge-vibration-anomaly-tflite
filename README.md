# Industrial Anomaly Vibration Detection using TFLite Autoencoders

**20CS2E82 – AI for Edge Devices · Project 3 (Group I – TinyML, Model Compression & Accelerators)**
Shakthikumar Sadhanandham · 71812301232 · Department of CSE, Sri Ramakrishna Engineering College, Coimbatore

Bearing-fault detection on an edge node. 3-axis accelerometer windows (12 kHz, 4,096 samples) are turned into a
96-element FFT band-energy vector, and a 96-32-16-8-16-32-96 autoencoder trained only on healthy vibration flags
windows whose reconstruction error exceeds the 99th percentile of healthy validation error. The model is converted
to a fully integer-quantised (INT8) TensorFlow Lite flatbuffer.

## Results (from the report)

| Metric | FP32 Keras (baseline) | INT8 TFLite (edge) |
|---|---|---|
| Model size | 129.15 KB | 16.41 KB |
| F1 / AUC | 0.9327 / 0.9827 | 0.9327 / 0.9828 |
| Latency per inference | 0.287 ms | 0.0138 ms |
| Est. tensor arena | 30.94 KB (FP32 TFLite) | 8.32 KB |
| Uplink per sensor | 70.3 KB/s | 46.9 B/s |

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Synthetic CWRU-style 3-axis bearing vibration generator and FFT band features |
| `train.py` | Trains the autoencoder on healthy data, exports FP32 and INT8 TFLite models |
| `evaluate.py` | FP32 vs INT8: detection metrics, size, latency, tensor arena, bandwidth, streaming log |
| `make_figs.py` | Plots the error distribution, confusion matrix and size/recall charts |

## Run

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python make_figs.py
```

Works the same in Google Colab: upload the four `.py` files and run the commands with a leading `!`.

## Note on data

The CWRU and NASA bearing archives could not be downloaded in the build environment, so `dataset.py` generates
physics-based synthetic signals using CWRU constants (12 kHz, 1,730–1,797 rpm, 6205 bearing defect ratios).
Real CWRU windows can be passed to `band_features()` without changing the model code.
