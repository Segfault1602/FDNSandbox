#pragma once

#include <sffdn/sffdn.h>

#include <vector>

namespace fdn_info
{
bool GetInputAndOutputGains(const sfFDN::FDN* fdn, std::vector<float>& input_gains, std::vector<float>& output_gains);

bool GetDelays(const sfFDN::FDN* fdn, std::vector<uint32_t>& delays);

bool GetFeedbackMatrix(const sfFDN::FDN* fdn, std::vector<float>& feedback_matrix, uint32_t& N,
                       uint64_t sample_index = 0);

} // namespace fdn_info