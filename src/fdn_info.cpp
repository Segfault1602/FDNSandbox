#include "fdn_info.h"

#include <sffdn/sffdn.h>

#include "settings.h"

#include <algorithm>
#include <cstddef>
#include <imgui.h>
#include <iostream>

namespace
{
size_t GetElapsedSamples()
{
    const auto sample_rate = Settings::Instance().SampleRateAs<float>();
    float delta_seconds = 1.0f / sample_rate;
    if (ImGui::GetCurrentContext() != nullptr)
    {
        delta_seconds = ImGui::GetIO().DeltaTime;
    }

    const float elapsed_samples = std::max(0.0f, delta_seconds * sample_rate);
    return std::max<size_t>(1U, static_cast<size_t>(elapsed_samples));
}

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
        const size_t samples_elapsed = GetElapsedSamples();

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
        const size_t samples_elapsed = GetElapsedSamples();

        if (samples_elapsed > 1)
        {
            const size_t advance_count = samples_elapsed - 1;
            std::vector<float> advance_input(advance_count * static_cast<size_t>(N), 0.0f);
            std::vector<float> advance_output(advance_count, 0.0f);
            const sfFDN::AudioBuffer advance_input_buffer(static_cast<uint32_t>(advance_count), N, advance_input);
            sfFDN::AudioBuffer advance_output_buffer(static_cast<uint32_t>(advance_count), 1U, advance_output);
            output_tv_gains->Process(advance_input_buffer, advance_output_buffer);
        }

        for (uint32_t index = 0; index < N; ++index)
        {
            auto processor = output_tv_gains->Clone();
            auto* clone = dynamic_cast<sfFDN::TimeVaryingParallelGains*>(processor.get());
            if (clone == nullptr)
            {
                return false;
            }

            std::vector<float> input(N, 0.0f);
            std::vector<float> output(1, 0.0f);
            input[index] = 1.0f;
            const sfFDN::AudioBuffer input_buffer(1U, N, input);
            sfFDN::AudioBuffer output_buffer(1U, 1U, output);
            clone->Process(input_buffer, output_buffer);
            output_gains[index] = output.front();
        }

        std::vector<float> advance_input(N, 0.0f);
        std::vector<float> advance_output(1, 0.0f);
        const sfFDN::AudioBuffer advance_input_buffer(1U, N, advance_input);
        sfFDN::AudioBuffer advance_output_buffer(1U, 1U, advance_output);
        output_tv_gains->Process(advance_input_buffer, advance_output_buffer);
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

    std::cerr << "[fdn_info::GetInputAndOutputGains]: Output gains processor is not a ParallelGains instance.\n";
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
            return true;
        }
    }

    LogUnexpectedProcessor(role);
    return false;
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