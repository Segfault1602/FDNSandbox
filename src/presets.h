#pragma once

#include "app.h"

#include <sffdn/sffdn.h>

namespace presets
{

const FDNConfig kDefaultFDNConfig = {
    .N = 8,                                                                            // Default number of channels
    .input_gains = std::vector<float>(8, 0.5f),                                        // Default input gains
    .output_gains = std::vector<float>(8, 0.5f),                                       // Default output gains
    .delays = sfFDN::GetDelayLengths(8, 500, 3000, sfFDN::DelayLengthType::Primes, 1), // Default delays in milliseconds
    .feedback_matrix = sfFDN::GenerateMatrix(8, sfFDN::ScalarMatrixType::Hadamard),    // Default feedback matrix
    .is_cascaded = false, // Default to non-cascaded feedback matrix
    .cascaded_feedback_matrix_info = sfFDN::CascadedFeedbackMatrixInfo(),
    .num_stages = 1,                                    // Default number of stages for cascaded feedback matrix
    .sparsity = 1.0f,                                   // Default sparsity level
    .cascade_gain = 1.0f,                               // Default gain per sample for cascaded feedback matrix
    .delay_filter_type = DelayFilterType::Proportional, // Default delay filter type
    .feedback_gain = 0.9999f,                           // Default feedback gain for proportional feedback
    .t60_dc = 2.f,                                      // Default T60 for DC filter design
    .t60_ny = 0.1f,                                     // Default T60 for Nyquist filter design
    .t60s = {2.f, 2.f, 1.9f, 1.9f, 1.8f, 1.8f, 1.7f, 1.7f, 1.6f, 1.6f},
    .tc_gains = std::vector<float>(10, 0.0f) // Default tone correction gains
};

} // namespace presets