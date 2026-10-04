"""DSP: discrete-time signal x[n], FFT, Butterworth IIR low-pass, noise metrics."""
import numpy as np
from scipy import signal


def spectrum(x, fs):
    """Single-sided amplitude spectrum of x[n] using the FFT.

    Steps:
      1. Remove the mean (DC) so the 0 Hz bin does not hide the other frequencies.
      2. Multiply by a Hann window to reduce spectral leakage.
      3. rfft -> one-sided spectrum, frequency axis f_k = k*fs/N  (0 ... fs/2).
      4. Scale by 2/sum(w) so a sine of amplitude A shows a peak of height A.
         (The window lowers the signal level; sum(w) is the "coherent gain" correction.
          Dividing by N instead would under-read the amplitude by about half for Hann.)
    """
    x = np.asarray(x, float)
    x = x - np.mean(x)
    N = len(x)
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = 2.0 * np.abs(X) / np.sum(w)
    mag[0] = mag[0] / 2.0                      # DC bin is not mirrored, so no doubling
    if N % 2 == 0:
        mag[-1] = mag[-1] / 2.0                # Nyquist bin is not mirrored either
    return f, mag


def dominant_frequency(f, mag):
    """Strongest non-DC peak. Parabolic interpolation refines it between FFT bins."""
    if len(mag) < 4:
        return 0.0, 0.0
    k = int(np.argmax(mag[1:])) + 1
    if 1 <= k < len(mag) - 1:
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(f[k] + shift * (f[1] - f[0])), float(b)
    return float(f[k]), float(mag[k])


def butter_lowpass(x, fs, cutoff, order):
    """Butterworth IIR low-pass. sosfiltfilt = forward+backward => zero phase (doubles effective order)."""
    nyq = fs / 2.0
    if not 0 < cutoff < nyq:
        raise ValueError(f"Cutoff must be between 0 and Nyquist ({nyq:g}).")
    sos = signal.butter(order, cutoff / nyq, btype="low", output="sos")
    y = signal.sosfiltfilt(sos, x)
    # frequency response of the filter itself (single pass) for plotting
    w, h = signal.sosfreqz(sos, worN=256, fs=fs)
    return y, sos, w, 20 * np.log10(np.maximum(np.abs(h), 1e-6))


def noise_metrics(raw, filt, f_raw, mag_raw, f_f, mag_f, cutoff):
    """Noise = high-frequency content above the cutoff. Compare raw vs filtered."""
    hf = lambda f, m: float(np.sum(m[f > cutoff] ** 2)) + 1e-12
    hf_raw, hf_filt = hf(f_raw, mag_raw), hf(f_f, mag_f)
    d_raw, d_f = np.diff(raw), np.diff(filt)    # first difference ~ sample-to-sample jitter
    return {
        "noise_std_raw": round(float(np.std(raw - filt)), 4),
        "hf_energy_reduction_db": round(10 * np.log10(hf_raw / hf_filt), 2),
        "jitter_reduction_pct": round(100 * (1 - np.std(d_f) / (np.std(d_raw) + 1e-12)), 1),
        "variance_raw": round(float(np.var(raw)), 4),
        "variance_filtered": round(float(np.var(filt)), 4),
    }


def run_dsp(x, fs, cutoff, order):
    x = np.asarray(x, float)
    if len(x) < 16:
        raise ValueError("Need at least 16 samples for DSP analysis.")
    y, sos, w, h_db = butter_lowpass(x, fs, cutoff, order)
    f1, m1 = spectrum(x, fs)
    f2, m2 = spectrum(y, fs)
    peak_f, peak_a = dominant_frequency(f1, m1)
    r = lambda a: np.round(a, 4).tolist()
    return {
        "n": list(range(len(x))), "raw": r(x), "filtered": r(y),
        "freq": r(f1), "mag_raw": r(m1), "mag_filtered": r(m2),
        "filter_response": {"freq": r(w), "gain_db": r(h_db)},
        "params": {"fs": fs, "cutoff": cutoff, "order": order, "nyquist": fs / 2, "N": len(x),
                   "freq_resolution": round(fs / len(x), 5), "filter_type": "Butterworth IIR low-pass (zero-phase, sosfiltfilt)"},
        "dominant": {"frequency": round(peak_f, 4), "amplitude": round(peak_a, 4),
                     "period_samples": round(1 / peak_f, 2) if peak_f > 0 else None},
        "metrics": noise_metrics(x, y, f1, m1, f2, m2, cutoff),
    }
