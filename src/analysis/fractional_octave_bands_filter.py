import pyfar as pf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz_sos


sample_rate = 48000
n_fractions = 1
f_min = 20
f_max = 20000

center_freqs = pf.dsp.filter.fractional_octave_frequencies(
    num_fractions=n_fractions, frequency_range=(f_min, f_max), return_cutoff=False
)[0]
impulse = np.zeros(32000)
impulse[0] = 1.0

print("Center Frequencies:")
print(center_freqs)

f_bank = pf.dsp.filter.fractional_octave_bands(
    None,
    num_fractions=n_fractions,
    frequency_range=(f_min, f_max),
    sampling_rate=sample_rate,
)

print(f_bank.coefficients[0].shape)

print(f_bank.coefficients[0])
print("\n\n")

n_bands = f_bank.coefficients.shape[0]
n_stage = f_bank.coefficients.shape[1]
n_coeffs = f_bank.coefficients.shape[2]

print(
    f"Number of bands: {n_bands}, Number of stages: {n_stage}, Number of coefficients per stage: {n_coeffs}"
)

print(
    f"constexpr std::array<std::array<float, {n_stage*n_coeffs}>, {n_bands}> kOctaveBandCoeffs = {{{{"
)

for i in range(f_bank.coefficients.shape[0]):
    first_band = f_bank.coefficients[i]
    w, h = freqz_sos(first_band, worN=4096)
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
    plt.plot(w / np.pi, db)

    print("\t{")
    for j in range(first_band.shape[0]):
        print(f"\t", end="")
        for k in range(first_band.shape[1]):
            print(f"{first_band[j, k]}, ", end="")
    print("},")

print("}};\n")
# plt.show()

# .freq.T  # shape: (filter_length, n_bands)
# f_bank = np.squeeze(f_bank)

# print(f_bank.shape)

# for i, freq in enumerate(center_freqs):
#     plt.plot(f_bank[:, i], label=f"{freq:.2f} Hz")

# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude")
# plt.title("Fractional Octave Bands")
# plt.legend()
# plt.show()
