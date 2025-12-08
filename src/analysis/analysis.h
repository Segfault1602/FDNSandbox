#pragma once

#include <span>
#include <vector>

namespace fdn_analysis
{
/**
 * @brief Compute the energy decay curve of a signal.
 *
 * @param signal The input signal.
 * @param to_db If true, convert the energy values to decibels.
 * @return std::vector<float> The short-time energy of the signal.
 */
std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db = false);

/**
 * @brief Compute the energy decay relief of a signal using an octave band filter bank.
 *
 * @param signal The input signal.
 * @param to_db If true, convert the energy values to decibels.
 * @return std::array<std::vector<float>, 10> The energy decay relief for each octave band.
 */
std::array<std::vector<float>, 10> EnergyDecayRelief(std::span<const float> signal, bool to_db = false);

std::array<float, 10> GetOctaveBandFrequencies();

struct EstimateT60Results
{
    float t60;
    float decay_start_time;
    float decay_end_time;
    float intercept;
    float slope;
};

/**
 * @brief Estimate the T60 time from an energy decay curve.
 *
 * @param decay_curve The energy decay curve.
 * @param decay_start_db The starting dB value for the decay.
 * @param decay_end_db The ending dB value for the decay.
 * @return float The estimated T60 time in seconds.
 */
EstimateT60Results EstimateT60(std::span<const float> decay_curve, std::span<const float> time, float decay_start_db,
                               float decay_end_db);

struct EchoDensityResults
{
    std::vector<float> echo_densities;
    std::vector<int> sparse_indices;
    float mixing_time;
};

/**
 * @brief Compute the echo density of a signal.
 * From:
 * Abel & Huang 2006, "A simple, robust measure of reverberation echo
 * density", In: Proc. of the 121st AES Convention, San Francisco
 *
 * Based on the MATLAB implementation:
 * https://github.com/SebastianJiroSchlecht/fdnToolbox/blob/master/External/echoDensity.m
 *
 * @param signal The input signal.
 * @param window_size The size of the analysis window.
 * @param sample_rate The sample rate of the signal.
 * @param hop_size The hop size for the analysis.
 * @return EchoDensityResults The results of the echo density computation.
 */
EchoDensityResults EchoDensity(std::span<const float> signal, uint32_t window_size, uint32_t sample_rate,
                               uint32_t hop_size);

} // namespace fdn_analysis