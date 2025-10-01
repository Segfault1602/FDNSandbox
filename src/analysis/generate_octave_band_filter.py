import numpy as np
import pyfar as pf
import os

import matplotlib.pyplot as plt

if __name__ == "__main__":
    SAMPLING_RATE = 48000
    N_SAMPLES = 4096
    FRACTION = 1
    FREQUENCY_RANGE = (31.25, 16000)
    OVERLAP = 1
    SLOPE = 0

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # OUT_PATH = os.path.join(CURRENT_DIR, "octave_band_filters")

    filter, freqs = pf.dsp.filter.reconstructing_fractional_octave_bands(None, num_fractions=FRACTION, frequency_range=FREQUENCY_RANGE, overlap=OVERLAP, slope=SLOPE, n_samples=N_SAMPLES, sampling_rate=SAMPLING_RATE)

    OUT_PATH = os.path.join(CURRENT_DIR, f"octave_band_filters_fir.h")

    f = open(OUT_PATH, "w")
    f.write("// This file was generated with src/analysis/generate_octave_band_filter.py\n")
    f.write(f"// Sampling rate: {SAMPLING_RATE} Hz\n")
    f.write(f"// Number of samples: {N_SAMPLES}\n")
    f.write(f"// Fraction: {FRACTION}\n")
    f.write(f"// Frequency range: {FREQUENCY_RANGE[0]} Hz to {FREQUENCY_RANGE[1]} Hz\n")
    f.write(f"// Overlap: {OVERLAP}\n")
    f.write(f"// Slope: {SLOPE} dB/octave\n\n")
    f.write("#pragma once\n\n")
    f.write("#include <array>\n\n")

    f.write(f"constexpr std::array<std::array<float, {N_SAMPLES}>, {len(freqs)}> kOctaveBandFirFilters = {{{{\n")

    for idx, freq, in enumerate(freqs):
        print(f"Band {idx}: {freq} Hz")
        var_name = f"kOctaveBandFir_{int(freq)}Hz"
        c_array = "    {"
        c_array += ", ".join([f"{v:.8e}f" for v in filter.coefficients[idx]])
        c_array += "},\n"
        f.write(c_array)

    f.write("}};\n")

    f.close()

    # x = np.zeros((N_SAMPLES,))
    # x[0] = 1
    # x = pf.Signal(x, sampling_rate=SAMPLING_RATE)
    # y = filter.process(x)
    # y_sum = pf.Signal(np.sum(y.time, 0), y.sampling_rate)

    # plt.figure()

    # pf.plot.time(y_sum)
    # plt.figure()

    # pf.plot.time_freq(y)

    # plt.show()