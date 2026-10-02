"""
Digital Modulation BER Simulator (BPSK, QPSK, 16-QAM, 64-QAM) over AWGN.

Simulates bit error rate vs Eb/N0 and overlays the theoretical curves.
Also plots constellation diagrams at a chosen SNR.

Run:  python modulation_ber.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def gray_pam_levels(k):
    """Gray-coded PAM levels for k bits per dimension. Returns array indexed by bit-pattern integer."""
    M = 2 ** k
    levels = np.arange(-(M - 1), M, 2)           # -M+1 ... M-1
    gray = np.arange(M) ^ (np.arange(M) >> 1)    # gray code of index
    mapping = np.zeros(M)
    mapping[gray] = levels
    return mapping


def modulate(bits, M):
    """Map a bit array to unit-average-energy symbols. M = 2 (BPSK) or square QAM."""
    if M == 2:
        return 1 - 2 * bits.astype(float)
    k = int(np.log2(M)) // 2                      # bits per I or Q
    b = bits.reshape(-1, 2 * k)
    w = 2 ** np.arange(k - 1, -1, -1)
    i_idx = b[:, :k] @ w
    q_idx = b[:, k:] @ w
    lv = gray_pam_levels(k)
    sym = lv[i_idx] + 1j * lv[q_idx]
    return sym / np.sqrt(2 * (M - 1) / 3)


def demodulate(rx, M):
    if M == 2:
        return (rx.real < 0).astype(int)
    k = int(np.log2(M)) // 2
    scale = np.sqrt(2 * (M - 1) / 3)
    lv = gray_pam_levels(k)
    r = rx * scale

    def slice_axis(x):
        # nearest PAM level -> its bit pattern
        idx = np.abs(x[:, None] - lv[None, :]).argmin(axis=1)
        return ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1)

    return np.hstack([slice_axis(r.real), slice_axis(r.imag)]).ravel()


def simulate(M, ebn0_db, n_bits):
    bps = int(np.log2(M))
    n_bits -= n_bits % bps
    bits = rng.integers(0, 2, n_bits)
    tx = modulate(bits, M)
    ebn0 = 10 ** (ebn0_db / 10)
    # Es = 1  ->  N0 = Es / (bps * Eb/N0)
    n0 = 1 / (bps * ebn0)
    if M == 2:
        noise = np.sqrt(n0 / 2) * rng.standard_normal(len(tx))
    else:
        noise = np.sqrt(n0 / 2) * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    rx = tx + noise
    return np.mean(demodulate(rx, M) != bits), tx, rx


def theory(M, ebn0_db):
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    if M in (2, 4):
        return Q(np.sqrt(2 * ebn0))
    k = np.log2(M)
    return (4 / k) * (1 - 1 / np.sqrt(M)) * Q(np.sqrt(3 * k * ebn0 / (M - 1)))


def main():
    schemes = {"BPSK": 2, "QPSK": 4, "16-QAM": 16, "64-QAM": 64}
    ebn0_range = np.arange(0, 17, 1)
    plt.figure(figsize=(9, 6))
    print(f"{'Eb/N0':>6} | " + " | ".join(f"{n:>9}" for n in schemes))
    table = {n: [] for n in schemes}
    for name, M in schemes.items():
        for e in ebn0_range:
            n_bits = 2_000_000 if e < 10 else 6_000_000
            ber, _, _ = simulate(M, e, n_bits)
            table[name].append(ber)
        plt.semilogy(ebn0_range, np.maximum(table[name], 1e-7), "o", label=f"{name} (sim)")
        th = theory(M, ebn0_range)
        plt.semilogy(ebn0_range, th, "-", alpha=0.7, label=f"{name} (theory)")
    for e_i, e in enumerate(ebn0_range):
        if e % 4 == 0:
            print(f"{e:>6} | " + " | ".join(f"{table[n][e_i]:9.2e}" for n in schemes))
    plt.ylim(1e-6, 1)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("BER vs Eb/N0 over AWGN channel")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "ber_curves.png"), dpi=130)
    plt.close()

    # Constellations at 15 dB (16/64-QAM) and 8 dB (BPSK/QPSK)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (name, M), snr in zip(axs, schemes.items(), [8, 8, 15, 20]):
        _, tx, rx = simulate(M, snr, 8000)
        rx_c, ideal = np.asarray(rx, dtype=complex), np.unique(np.asarray(tx, dtype=complex))
        ax.scatter(rx_c.real, rx_c.imag, s=3, alpha=0.4)
        ax.scatter(ideal.real, ideal.imag, c="r", s=14)
        ax.set_title(f"{name} @ {snr} dB Eb/N0")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "constellations.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
