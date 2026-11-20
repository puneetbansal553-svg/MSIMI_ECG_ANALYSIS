import numpy as np
import scipy.signal

# 1. Test Lomb-Scargle on synthetic signal
fs_eff = 1.0  # 1 Hz average sampling
T = 300.0  # 300 seconds
t = np.sort(np.random.uniform(0, T, int(T * fs_eff)))
f0 = 0.1  # 0.1 Hz sinusoid
A = 100.0  # Amplitude = 100 ms
y = A * np.sin(2 * np.pi * f0 * t)

var_true = np.var(y)  # A^2 / 2 = 5000 ms^2
print(f"True variance: {var_true:.2f} ms^2")

# Scipy lombscargle
freqs = np.linspace(0.01, 0.5, 1000)
angular_freqs = 2 * np.pi * freqs
p_raw = scipy.signal.lombscargle(t, y, angular_freqs, normalize=False)

# What is the integral of p_raw?
print(f"p_raw max: {np.max(p_raw):.2f}, mean: {np.mean(p_raw):.2f}")
# If scaled by 4 / (N * T)?
N = len(t)
# Let's check Parseval: integral P(f) df == Var(y)
# In standard Press & Rybicki / Scargle:
# Normalized periodogram: p_norm = scipy.signal.lombscargle(t, y, angular_freqs, normalize=True)
p_norm = scipy.signal.lombscargle(t, y, angular_freqs, normalize=True)
print(f"p_norm max: {np.max(p_norm):.4f}")

# To convert to PSD in ms^2/Hz:
# If P_psd(f) = (2 * p_raw) / (fs * N)? Or 2 * p_raw / (T_span)?
# Notice p_raw peak value:
# At resonance f = f0, sum y * sin(wt) ~ N * A / 2.
# Numerator ~ (N * A / 2)^2 = N^2 * A^2 / 4.
# Denominator ~ N / 2.
# p_raw ~ 0.5 * (N^2 A^2 / 4) / (N / 2) = N * A^2 / 4.
# For A=100, A^2=10000, N=300: p_raw peak ~ 300 * 10000 / 4 = 750,000!
print(f"Theoretical p_raw peak: {N * (A**2) / 4:.2f}, Actual: {np.max(p_raw):.2f}")
# The width of the sinc-like peak in frequency is delta_f ~ 1/T = 1/300 Hz.
# The integral under the peak is Peak * (1/T) ~ (N * A^2 / 4) * (1/T) = (N/T) * (A^2 / 4).
# Since N/T = fs_eff: integral ~ fs_eff * A^2 / 4.
# To make the integral equal Var(y) = A^2 / 2, we must scale by 2 / (fs_eff * N)? No!
# Let's integrate p_raw * [2 / (N * fs_eff)] or p_raw * (2 * dt)?
print(f"Trapz of p_raw: {np.trapz(p_raw, freqs):.2f}")
scale_factor = var_true / np.trapz(p_raw, freqs)
print(f"Empirical scale factor to match variance: {scale_factor:.6e}")

