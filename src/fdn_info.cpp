#include "fdn_info.h"
#include "sffdn/audio_processor.h"
#include "sffdn/feedback_matrix.h"

#include <sffdn/fdn.h>

#include <iostream>

namespace fdn_info
{

bool GetInputAndOutputGains(const sfFDN::FDN* fdn, std::vector<float>& input_gains, std::vector<float>& output_gains)
{
    sfFDN::AudioProcessor* input_gains_processor = fdn->GetInputGains();
    sfFDN::AudioProcessor* output_gains_processor = fdn->GetOutputGains();

    const uint32_t N = input_gains_processor->OutputChannelCount();

    if (input_gains.size() != N || output_gains.size() != N)
    {
        input_gains.resize(N, 0.0f);
        output_gains.resize(N, 0.0f);
    }

    auto* input_parallel_gains = dynamic_cast<sfFDN::ParallelGains*>(input_gains_processor);
    if (input_parallel_gains != nullptr)
    {
        input_parallel_gains->GetGains(input_gains);
    }
    else
    {
        std::cerr << "[fdn_info::GetInputAndOutputGains]: Input gains processor is not a ParallelGains instance.\n";
        return false;
    }

    auto* output_parallel_gains = dynamic_cast<sfFDN::ParallelGains*>(output_gains_processor);
    if (output_parallel_gains != nullptr)
    {
        output_parallel_gains->GetGains(output_gains);
    }
    else
    {
        std::cerr << "[fdn_info::GetInputAndOutputGains]: Output gains processor is not a ParallelGains instance.\n";
        return false;
    }

    return true;
}

bool GetDelays(const sfFDN::FDN* fdn, std::vector<uint32_t>& delays)
{
    const sfFDN::DelayBank& delay_bank = fdn->GetDelayBank();
    std::vector<uint32_t> delay_values = delay_bank.GetDelays();

    delays.resize(delay_values.size());
    std::ranges::copy(delay_values, delays.begin());

    return true;
}

bool GetFeedbackMatrix(const sfFDN::FDN* fdn, std::vector<float>& feedback_matrix, uint32_t& N)
{
    sfFDN::AudioProcessor* feedback_matrix_processor = fdn->GetFeedbackMatrix();
    if (feedback_matrix_processor == nullptr)
    {
        std::cerr << "[fdn_info::GetFeedbackMatrix]: Feedback matrix processor is null.\n";
        return false;
    }

    N = feedback_matrix_processor->OutputChannelCount();
    feedback_matrix.resize(N * N);

    if (auto* scalar_matrix = dynamic_cast<sfFDN::ScalarFeedbackMatrix*>(feedback_matrix_processor))
    {
        N = scalar_matrix->GetSize();
        feedback_matrix.resize(N * N);
        return scalar_matrix->GetMatrix(feedback_matrix);
    }

    if (auto* filter_matrix = dynamic_cast<sfFDN::FilterFeedbackMatrix*>(feedback_matrix_processor))
    {
        N = filter_matrix->InputChannelCount();
        feedback_matrix.resize(N * N);
        return filter_matrix->GetFirstMatrix(feedback_matrix);
    }

    std::cerr << "[fdn_info::GetFeedbackMatrix]: Feedback matrix processor is not a ScalarFeedbackMatrix or "
                 "FilterFeedbackMatrix.\n";

    return false;
}

} // namespace fdn_info