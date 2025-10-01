#pragma once

#include "app.h"

#include <sffdn/sffdn.h>

namespace presets
{

std::unique_ptr<sfFDN::FDN> CreateDefaultFDN();

const FDNConfig kDefaultFDNConfig = {
    .N = 8, // Default number of channels
    .transposed = false,
    .input_gains = std::vector<float>(8, 0.5f),                                        // Default input gains
    .output_gains = std::vector<float>(8, 0.5f),                                       // Default output gains
    .delays = sfFDN::GetDelayLengths(8, 500, 3000, sfFDN::DelayLengthType::Primes, 1), // Default delays in milliseconds
    .matrix_info = sfFDN::GenerateMatrix(8, sfFDN::ScalarMatrixType::Hadamard),        // Default feedback matrix
    .attenuation_t60s = {0.9999f}, // Default feedback gain for proportional feedback
    .tc_gains = {},                // Default tone correction gains
    .tc_frequencies = {},          // Default tone correction frequencies
    .use_extra_delays = false,
    .input_stage_delays = {},
    .schroeder_allpass_delays = {},
    .schroeder_allpass_gains = {}

};

} // namespace presets