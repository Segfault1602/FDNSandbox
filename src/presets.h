#pragma once

#include "app.h"

#include <sffdn/sffdn.h>

namespace presets
{

std::unique_ptr<sfFDN::FDN> CreateDefaultFDN();

const sfFDN::FDNConfig kDefaultFDNConfig = {
    .N = 8, // Default number of channels
    .transposed = false,
    .direct_gain = 0.0f,
    .input_gains = std::vector<float>(8, 0.5f),                               // Default input gains
    .output_gains = std::vector<float>(8, 0.5f),                              // Default output gains
    .delays = {809, 877, 937, 1049, 1151, 1249, 1373, 1499},                  // Default delays in milliseconds
    .matrix_info = sfFDN::GenerateMatrix(8, sfFDN::ScalarMatrixType::Random), // Default feedback matrix
    .attenuation_t60s = {0.9999f}, // Default feedback gain for proportional feedback
    .tc_gains = {},                // Default tone correction gains
    .tc_frequencies = {},          // Default tone correction frequencies
    .use_extra_delays = false,
    .input_stage_delays = {},
    .input_schroeder_allpass_config = std::nullopt,
    .input_diffuser = std::nullopt,
};

} // namespace presets