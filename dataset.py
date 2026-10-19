"""Synthetic CWRU-style 3-axis bearing vibration generator + FFT band features.

Real CWRU / NASA files could not be downloaded in the build sandbox, so signals are
synthesised from bearing physics: shaft harmonics + broadband noise (healthy), plus
periodic resonance-ringing impacts at the bearing defect frequency (faulty).
Set FS, N and defect ratios below to match CWRU drive-end data (12 kHz, ~1750 rpm).
"""
import numpy as np

FS = 12000          # sampling rate (Hz), as in CWRU 12k drive-end data
N = 4096            # samples per window (~0.34 s)
N_BANDS = 32        # spectral energy bands per axis
BPFO, BPFI, BSF = 3.585, 5.415, 2.357   # defect-frequency ratios (x shaft speed), 6205 bearing
AXIS_GAIN = np.array([1.0, 0.8, 0.6])   # x, y, z sensitivity
F_RES = np.array([3200., 3000., 2800.]) # structural resonance excited by impacts (Hz)
T = np.arange(N) / FS


def _impacts(rng, f_defect, amp, f_res, mod_fr=None, fr=None):
    """Train of exponentially decaying resonance bursts at the defect frequency."""
    sig = np.zeros(N)
    klen = 240
    tk = np.arange(klen) / FS
    kernel = np.exp(-900 * tk) * np.sin(2 * np.pi * f_res * tk)
    t_i = rng.uniform(0, 1 / f_defect)
    while t_i < N / FS - klen / FS:
        a = amp * (1 + 0.3 * rng.standard_normal())
        if mod_fr is not None:                       # inner race: load-zone modulation
            a *= 1 + np.cos(2 * np.pi * fr * t_i)
        i0 = int(t_i * FS)
        sig[i0:i0 + klen] += a * kernel
        t_i += (1 / f_defect) * (1 + 0.01 * rng.standard_normal())   # ~1 % slip jitter
    return sig


def make_window(rng, fault=None, severity=1.0):
    """Return (3, N) accelerometer window. fault in {None,'outer','inner','ball'}."""
    rpm = rng.uniform(1730, 1797)                    # CWRU load 0..3 hp speed range
    fr = rpm / 60
    load = rng.uniform(0.9, 1.2)
    out = np.zeros((3, N))
    for a in range(3):
        s = sum(amp * np.sin(2 * np.pi * k * fr * T + rng.uniform(0, 6.28))
                for k, amp in ((1, .35), (2, .12), (3, .05)))
        s = AXIS_GAIN[a] * load * s + 0.08 * rng.standard_normal(N)
        s += 0.03 * np.sin(2 * np.pi * F_RES[a] * T + rng.uniform(0, 6.28)) * rng.standard_normal(N)
        if fault:
            if fault == "outer":
                s += _impacts(rng, BPFO * fr, 0.30 * severity * AXIS_GAIN[a], F_RES[a])
            elif fault == "inner":
                s += _impacts(rng, BPFI * fr, 0.25 * severity * AXIS_GAIN[a], F_RES[a], True, fr)
            else:
                s += _impacts(rng, 2 * BSF * fr, 0.20 * severity * AXIS_GAIN[a], F_RES[a])
        out[a] = s
    return out


_EDGES = np.geomspace(10, 5900, N_BANDS + 1)
_FREQ = np.fft.rfftfreq(N, 1 / FS)
_BAND_IDX = [(_FREQ >= _EDGES[i]) & (_FREQ < _EDGES[i + 1]) for i in range(N_BANDS)]
_HANN = np.hanning(N)

def band_features(win):
    """FFT -> log spectral energy in 32 log-spaced bands per axis -> 96-dim vector."""
    spec = np.abs(np.fft.rfft(win * _HANN, axis=1)) ** 2 / N
    feats = [[np.log10(spec[a][m].sum() + 1e-9) for m in _BAND_IDX] for a in range(3)]
    return np.asarray(feats, dtype=np.float32).ravel()


def build(rng, n_train=3000, n_val=500, n_test_norm=500, n_test_fault=600):
    f = lambda w: band_features(w)
    Xtr = np.stack([f(make_window(rng)) for _ in range(n_train)])
    Xva = np.stack([f(make_window(rng)) for _ in range(n_val)])
    Xtn = np.stack([f(make_window(rng)) for _ in range(n_test_norm)])
    kinds, sev, Xtf, meta = ["outer", "inner", "ball"], [0.5, 1.0, 1.5], [], []
    for i in range(n_test_fault):
        k, s = kinds[i % 3], sev[(i // 3) % 3]
        Xtf.append(f(make_window(rng, k, s))); meta.append((k, s))
    return Xtr, Xva, Xtn, np.stack(Xtf), meta
