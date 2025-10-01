#include "fdn_info.h"

#include <sffdn/sffdn.h>

#include "settings.h"

#include <imgui.h>
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
    else if (auto* input_tv_gains = dynamic_cast<sfFDN::TimeVaryingParallelGains*>(input_gains_processor))
    {
        uint32_t samples_elapsed = ImGui::GetIO().DeltaTime * Settings::Instance().SampleRate();
        std::vector<float> input(samples_elapsed, 1.f);
        std::vector<float> output(samples_elapsed * N, 0.f);

        sfFDN::AudioBuffer input_buffer(samples_elapsed, 1, input);
        sfFDN::AudioBuffer output_buffer(samples_elapsed, N, output);

        input_tv_gains->Process(input_buffer, output_buffer);

        for (auto i = 0; i < N; ++i)
        {
            input_gains[i] = output_buffer.GetChannelSpan(i).back();
        }
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
    else if (auto* output_tv_gains = dynamic_cast<sfFDN::TimeVaryingParallelGains*>(output_gains_processor))
    {
        uint32_t samples_elapsed = ImGui::GetIO().DeltaTime * Settings::Instance().SampleRate();
        samples_elapsed = std::max(samples_elapsed, static_cast<uint32_t>(N));
        std::vector<float> input(samples_elapsed * N, 0.f);
        std::vector<float> output(samples_elapsed * N, 0.f);

        // Kinda hacky way to do this but if we make sure each channel are set to zeros except for one value, as long as
        // that one value does not overlap between channels we should be able to work out the output gains for each
        // channel
        sfFDN::AudioBuffer input_buffer(samples_elapsed, 1, input);
        for (uint32_t i = 0; i < N; ++i)
        {
            input_buffer.GetChannelSpan(i).last(N)[i] = 1.f;
        }

        sfFDN::AudioBuffer output_buffer(samples_elapsed * N, 1, output);

        output_tv_gains->Process(input_buffer, output_buffer);

        for (auto i = 0; i < N; ++i)
        {
            output_gains[i] = output_buffer.GetChannelSpan(0).last(N)[i];
        }
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