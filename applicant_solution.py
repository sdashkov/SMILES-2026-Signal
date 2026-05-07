import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):

    del tx_n

    score_filter = helpers["score_filter"]
    fit_tx_prediction = helpers["fit_tx_prediction"]

    def filt_matrix(x):
        y = np.empty_like(x)
        for ch in range(x.shape[1]):
            y[:, ch] = score_filter(x[:, ch])
        return y

    def rank1_from_band_matrix(x):
        cov = x.conj().T @ x / x.shape[0]
        _, vecs = np.linalg.eigh(cov)

        v = vecs[:, -1]
        shared = x @ v

        denom = np.vdot(shared, shared) + 1e-30
        out = np.empty_like(x)

        for ch in range(x.shape[1]):
            coef = np.vdot(shared, x[:, ch]) / denom
            out[:, ch] = coef * shared

        return out

    def inverse_score_filter_target(target, n_iter=3):

        z = target.copy()

        for _ in range(n_iter):
            fz = filt_matrix(z)
            z += target - fz

        return z

    rx_band = filt_matrix(rx)

    tx_band = fit_tx_prediction(rx)

    for _ in range(2):
        ext_band = rank1_from_band_matrix(rx_band - tx_band)
        ext_time = inverse_score_filter_target(ext_band, n_iter=2)
        tx_band = fit_tx_prediction(rx - ext_time)

    ext_band = rank1_from_band_matrix(rx_band - tx_band)

    tx_time = inverse_score_filter_target(tx_band, n_iter=3)
    ext_time = inverse_score_filter_target(ext_band, n_iter=3)

    tx_eff = filt_matrix(tx_time)
    ext_eff = filt_matrix(ext_time)

    rx_hat = rx.copy()


    for ch in range(rx.shape[1]):
        a0 = tx_eff[:, ch]
        a1 = ext_eff[:, ch]
        y = rx_band[:, ch]

        G = np.array(
            [
                [np.vdot(a0, a0), np.vdot(a0, a1)],
                [np.vdot(a1, a0), np.vdot(a1, a1)],
            ],
            dtype=np.complex128,
        )

        b = np.array(
            [
                np.vdot(a0, y),
                np.vdot(a1, y),
            ],
            dtype=np.complex128,
        )

        G += 1e-9 * np.trace(G).real * np.eye(2)

        coef = np.linalg.solve(G, b)

        for k in range(2):
            mag = np.abs(coef[k])
            if mag > 1.8:
                coef[k] *= 1.8 / mag

        rx_hat[:, ch] -= coef[0] * tx_time[:, ch] + coef[1] * ext_time[:, ch]

    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
