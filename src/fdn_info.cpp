#include "fdn_info.h"

#include <sffdn/sffdn.h>

#include "settings.h"

#include <algorithm>
#include <cstddef>
#include <imgui.h>
#include <iostream>

namespace
{
bool GetInputGains(sfFDN::AudioProcessor* proc, std::vector<float>& input_gains)
{
    auto* input_parallel_gains = dynamic_cast<sfFDN::ParallelGains*>(proc);
    if (input_parallel_gains != nullptr)
    {
        input_parallel_gains->GetGains(input_gains);
        return true;
    }
    else if (auto* input_tv_gains = dynamic_cast<sfFDN::TimeVaryingParallelGains*>(proc))
    {
        const float delta_seconds = ImGui::GetIO().DeltaTime;
        const auto sample_rate = Settings::Instance().SampleRateAs<float>();
        const float elapsed_samples = std::max(0.0f, delta_seconds * sample_rate);
        const size_t samples_elapsed = std::max<size_t>(1U, static_cast<size_t>(elapsed_samples));

        std::vector<float> input(samples_elapsed, 1.f);
        const size_t output_size = samples_elapsed * static_cast<size_t>(input_tv_gains->OutputChannelCount());
        std::vector<float> output(output_size, 0.f);

        const sfFDN::AudioBuffer input_buffer(static_cast<uint32_t>(samples_elapsed), 1U, input);
        sfFDN::AudioBuffer output_buffer(static_cast<uint32_t>(samples_elapsed), input_tv_gains->OutputChannelCount(),
                                         output);

        input_tv_gains->Process(input_buffer, output_buffer);

        for (uint32_t i = 0; i < input_tv_gains->OutputChannelCount(); ++i)
        {
            input_gains[i] = output_buffer.GetChannelSpan(i).back();
        }
        return true;
    }

    return false;
}

bool GetOutputGains(sfFDN::AudioProcessor* proc, std::vector<float>& output_gains)
{
    auto* output_parallel_gains = dynamic_cast<sfFDN::ParallelGains*>(proc);
    if (output_parallel_gains != nullptr)
    {
        output_parallel_gains->GetGains(output_gains);
        return true;
    }
    else if (auto* output_tv_gains = dynamic_cast<sfFDN::TimeVaryingParallelGains*>(proc))
    {
        const uint32_t N = output_tv_gains->InputChannelCount();
        const float delta_seconds = ImGui::GetIO().DeltaTime;
        const auto sample_rate = Settings::Instance().SampleRateAs<float>();
        const float elapsed_samples = std::max(0.0f, delta_seconds * sample_rate);
        const size_t samples_elapsed = std::max<size_t>(static_cast<size_t>(N), static_cast<size_t>(elapsed_samples));

        const size_t input_size = samples_elapsed * static_cast<size_t>(N);
        std::vector<float> input(input_size, 0.f);
        std::vector<float> output(samples_elapsed, 0.f);

        // Kinda hacky way to do this but if we make sure each channel are set to zeros except for one value, as long as
        // that one value does not overlap between channels we should be able to work out the output gains for each
        // channel
        sfFDN::AudioBuffer input_buffer(static_cast<uint32_t>(samples_elapsed), N, input);
        for (uint32_t i = 0; i < N; ++i)
        {
            input_buffer.GetChannelSpan(i).last(N)[i] = 1.f;
        }

        sfFDN::AudioBuffer output_buffer(static_cast<uint32_t>(samples_elapsed), 1U, output);

        output_tv_gains->Process(input_buffer, output_buffer);

        for (uint32_t i = 0; i < N; ++i)
        {
            output_gains[i] = output_buffer.GetChannelSpan(0).last(N)[i];
        }
        return true;
    }

    return false;
}

enum class GainProcessorRole
{
    Input,
    Output,
};

bool HasExpectedChannelCounts(const sfFDN::AudioProcessor& processor, uint32_t fdn_size, GainProcessorRole role)
{
    if (role == GainProcessorRole::Input)
    {
        return processor.InputChannelCount() == 1 && processor.OutputChannelCount() == fdn_size;
    }

    return processor.InputChannelCount() == fdn_size && processor.OutputChannelCount() == 1;
}

bool GetGains(sfFDN::AudioProcessor* processor, std::vector<float>& gains, GainProcessorRole role)
{
    return role == GainProcessorRole::Input ? GetInputGains(processor, gains) : GetOutputGains(processor, gains);
}

void LogEmptyProcessorChain(GainProcessorRole role)
{
    if (role == GainProcessorRole::Input)
    {
        std::cerr << "[fdn_info::GetInputAndOutputGains]: Input gains processor chain is empty.\n";
        return;
    }

    std::cerr << "[fdn_info::GetInputAndOutputGains]: Output gains processor chain is empty.\n";
}

void LogUnexpectedProcessor(GainProcessorRole role)
{
    if (role == GainProcessorRole::Input)
    {
        std::cerr << "[fdn_info::GetInputAndOutputGains]: Input gains processor is not a ParallelGains instance.\n";
        return;
    }

    std::cerr << "[fdn_info::GetInputAndOutputGains]: Outupt gains processor is not a ParallelGains instance.\n";
}

bool GetGainsFromProcessor(sfFDN::AudioProcessor* processor, std::vector<float>& gains, uint32_t fdn_size,
                           GainProcessorRole role)
{
    if (GetGains(processor, gains, role))
    {
        return true;
    }

    auto* processor_chain = dynamic_cast<sfFDN::AudioProcessorChain*>(processor);
    if (processor_chain == nullptr)
    {
        LogUnexpectedProcessor(role);
        return false;
    }

    const uint32_t processor_count = processor_chain->GetProcessorCount();
    if (processor_count == 0)
    {
        LogEmptyProcessorChain(role);
        return false;
    }

    for (uint32_t index = 0; index < processor_count; ++index)
    {
        auto* candidate = processor_chain->GetProcessor(index);
        assert(candidate != nullptr);
        if (HasExpectedChannelCounts(*candidate, fdn_size, role) && GetGains(candidate, gains, role))
        {
            break;
        }
    }

    return true;
}
} // namespace

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

    if (!GetGainsFromProcessor(input_gains_processor, input_gains, N, GainProcessorRole::Input))
    {
        return false;
    }

    return GetGainsFromProcessor(output_gains_processor, output_gains, N, GainProcessorRole::Output);
}

bool GetDelays(const sfFDN::FDN* fdn, std::vector<uint32_t>& delays)
{
    const sfFDN::DelayBank& delay_bank = fdn->GetDelayBank();
    auto delay_values = delay_bank.GetDelays();

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
    const auto resize_matrix = [&](uint32_t matrix_order) {
        N = matrix_order;
        const size_t matrix_size = static_cast<size_t>(matrix_order) * static_cast<size_t>(matrix_order);
        feedback_matrix.resize(matrix_size);
    };

    if (auto* scalar_matrix = dynamic_cast<sfFDN::ScalarFeedbackMatrix*>(feedback_matrix_processor))
    {
        resize_matrix(scalar_matrix->InputChannelCount());
        return scalar_matrix->GetMatrix(feedback_matrix);
    }

    if (auto* filter_matrix = dynamic_cast<sfFDN::FilterFeedbackMatrix*>(feedback_matrix_processor))
    {
        resize_matrix(filter_matrix->InputChannelCount());
        return filter_matrix->GetFirstMatrix(feedback_matrix);
    }

    std::cerr << "[fdn_info::GetFeedbackMatrix]: Feedback matrix processor is not a ScalarFeedbackMatrix or "
                 "FilterFeedbackMatrix.\n";

    return false;
}

} // namespace fdn_info