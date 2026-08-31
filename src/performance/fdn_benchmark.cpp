#include "performance/fdn_benchmark.h"

#include <sffdn/sffdn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fdn_sandbox::performance
{
namespace
{

using ProcessorArray = std::array<std::unique_ptr<sfFDN::AudioProcessor>, kBenchmarkStageCount>;

constexpr std::size_t StageIndex(BenchmarkStage stage) noexcept
{
    return static_cast<std::size_t>(stage);
}

std::vector<float> MakeInput(std::size_t sample_count)
{
    std::vector<float> input(sample_count);
    std::uint32_t state = 0x6d2b79f5U;
    for (float& sample : input)
    {
        state = state * 1664525U + 1013904223U;
        const float normalized = static_cast<float>(state >> 8U) / static_cast<float>(0x00ffffffU);
        sample = (normalized * 2.0f - 1.0f) * 0.25f;
    }
    return input;
}

std::expected<ProcessorArray, BenchmarkError> CreateProcessors(const sfFDN::FDNConfig& source_config, float sample_rate,
                                                               std::string_view slot_name)
{
    try
    {
        sfFDN::FDNConfig config = source_config;
        config.sample_rate = sample_rate;
        config.block_size = kBenchmarkBlockSize;
        config.delay_bank_config.block_size = kBenchmarkBlockSize;
        auto fdn = sfFDN::CreateFDNFromConfig(config);
        if (fdn == nullptr)
        {
            return std::unexpected(BenchmarkError{.message = std::string(slot_name) + " construction returned no FDN"});
        }

        // The committed preview/audio FDNs mute the direct path, so benchmark the same processing graph.
        fdn->SetDirectGain(0.0f);

        const sfFDN::AudioProcessor* input_block = fdn->GetInputGains();
        const sfFDN::AudioProcessor* feedback_matrix = fdn->GetFeedbackMatrix();
        const sfFDN::AudioProcessor* output_block = fdn->GetOutputGains();
        if (input_block == nullptr || feedback_matrix == nullptr || output_block == nullptr)
        {
            return std::unexpected(
                BenchmarkError{.message = std::string(slot_name) + " is missing a required processing stage"});
        }

        ProcessorArray processors;
        processors[StageIndex(BenchmarkStage::InputBlock)] = input_block->Clone();
        processors[StageIndex(BenchmarkStage::DelayBank)] = fdn->GetDelayBank().Clone();
        if (const sfFDN::AudioProcessor* loop_filters = fdn->GetLoopFilter())
        {
            processors[StageIndex(BenchmarkStage::LoopFilters)] = loop_filters->Clone();
        }
        processors[StageIndex(BenchmarkStage::FeedbackMatrix)] = feedback_matrix->Clone();
        processors[StageIndex(BenchmarkStage::OutputBlock)] = output_block->Clone();
        if (const sfFDN::AudioProcessor* tone_correction = fdn->GetTCFilter())
        {
            processors[StageIndex(BenchmarkStage::ToneCorrection)] = tone_correction->Clone();
        }
        processors[StageIndex(BenchmarkStage::CompleteFdn)] = std::move(fdn);

        for (const BenchmarkStage required_stage :
             {BenchmarkStage::CompleteFdn, BenchmarkStage::InputBlock, BenchmarkStage::DelayBank,
              BenchmarkStage::FeedbackMatrix, BenchmarkStage::OutputBlock})
        {
            if (processors[StageIndex(required_stage)] == nullptr)
            {
                return std::unexpected(BenchmarkError{.message = std::string(slot_name) + " could not clone " +
                                                                 BenchmarkStageName(required_stage)});
            }
        }
        return processors;
    }
    catch (const std::exception& error)
    {
        return std::unexpected(
            BenchmarkError{.message = std::string(slot_name) + " could not be prepared: " + error.what()});
    }
}

std::optional<BenchmarkTiming> MeasureProcessor(sfFDN::AudioProcessor& processor, std::uint32_t block_size,
                                                std::stop_token stop_token)
{
    if (stop_token.stop_requested())
    {
        return std::nullopt;
    }

    const std::size_t input_sample_count =
        static_cast<std::size_t>(block_size) * static_cast<std::size_t>(processor.InputChannelCount());
    const std::size_t output_sample_count =
        static_cast<std::size_t>(block_size) * static_cast<std::size_t>(processor.OutputChannelCount());
    std::vector<float> input_storage = MakeInput(input_sample_count);
    std::vector<float> output_storage(output_sample_count);
    const sfFDN::AudioBuffer input(block_size, processor.InputChannelCount(), input_storage);
    sfFDN::AudioBuffer output(block_size, processor.OutputChannelCount(), output_storage);

    constexpr std::size_t kWarmupIterations = 8;
    for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration)
    {
        processor.Process(input, output);
    }

    constexpr std::size_t kProbeIterations = 4;
    const auto probe_start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < kProbeIterations; ++iteration)
    {
        processor.Process(input, output);
    }
    const auto probe_end = std::chrono::steady_clock::now();
    const double probe_nanoseconds =
        std::chrono::duration<double, std::nano>(probe_end - probe_start).count() / kProbeIterations;

    constexpr double kTargetBatchNanoseconds = 1'000'000.0;
    constexpr std::size_t kMaximumBatchIterations = 1'000'000;
    const double safe_probe_nanoseconds = std::max(probe_nanoseconds, 1.0);
    const std::size_t batch_iterations =
        std::clamp(static_cast<std::size_t>(std::ceil(kTargetBatchNanoseconds / safe_probe_nanoseconds)),
                   std::size_t{1}, kMaximumBatchIterations);

    constexpr std::size_t kSampleCount = 21;
    std::array<double, kSampleCount> samples;
    for (double& sample : samples)
    {
        if (stop_token.stop_requested())
        {
            return std::nullopt;
        }

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t iteration = 0; iteration < batch_iterations; ++iteration)
        {
            processor.Process(input, output);
        }
        const auto end = std::chrono::steady_clock::now();
        sample = std::chrono::duration<double, std::nano>(end - start).count() / static_cast<double>(batch_iterations);
    }

    std::ranges::sort(samples);
    const double nanoseconds_per_block = samples[samples.size() / 2];
    const double lower_nanoseconds_per_block = samples[samples.size() / 4];
    const double upper_nanoseconds_per_block = samples[(samples.size() * 3) / 4];
    return BenchmarkTiming{
        .nanoseconds_per_block = nanoseconds_per_block,
        .nanoseconds_per_sample = nanoseconds_per_block / static_cast<double>(block_size),
        .lower_nanoseconds_per_sample = lower_nanoseconds_per_block / static_cast<double>(block_size),
        .upper_nanoseconds_per_sample = upper_nanoseconds_per_block / static_cast<double>(block_size),
    };
}

} // namespace

const char* BenchmarkStageName(BenchmarkStage stage) noexcept
{
    switch (stage)
    {
    case BenchmarkStage::CompleteFdn:
        return "Complete FDN";
    case BenchmarkStage::InputBlock:
        return "Input Block";
    case BenchmarkStage::DelayBank:
        return "Delay Bank";
    case BenchmarkStage::LoopFilters:
        return "Loop Filters";
    case BenchmarkStage::FeedbackMatrix:
        return "Feedback Matrix";
    case BenchmarkStage::OutputBlock:
        return "Output Block";
    case BenchmarkStage::ToneCorrection:
        return "Tone Correction";
    case BenchmarkStage::Count:
        break;
    }
    return "Unknown";
}

BenchmarkOutcome MeasureFdnPair(const BenchmarkRequest& request, std::stop_token stop_token)
{
    try
    {
        if (!std::isfinite(request.sample_rate) || request.sample_rate <= 0.0f)
        {
            return std::unexpected(BenchmarkError{.message = "Benchmark sample rate must be positive"});
        }
        auto processors_a = CreateProcessors(request.config_a, request.sample_rate, "FDN A");
        if (!processors_a)
        {
            return std::unexpected(std::move(processors_a.error()));
        }
        auto processors_b = CreateProcessors(request.config_b, request.sample_rate, "FDN B");
        if (!processors_b)
        {
            return std::unexpected(std::move(processors_b.error()));
        }

        BenchmarkResult result{
            .fdn_a = {.block_size = kBenchmarkBlockSize, .stages = {}},
            .fdn_b = {.block_size = kBenchmarkBlockSize, .stages = {}},
        };
        const auto measure_stage = [&](ProcessorArray& processors, FdnBenchmarkResult& fdn_result,
                                       std::size_t index) -> bool {
            if (processors[index] == nullptr)
            {
                return true;
            }
            fdn_result.stages[index] = MeasureProcessor(*processors[index], fdn_result.block_size, stop_token);
            return fdn_result.stages[index].has_value();
        };

        // Alternate which side runs first so thermal and scheduler drift do not systematically favor A.
        for (std::size_t index = 0; index < kBenchmarkStageCount; ++index)
        {
            const bool a_first = index % 2 == 0;
            const bool first_completed = a_first ? measure_stage(*processors_a, result.fdn_a, index)
                                                 : measure_stage(*processors_b, result.fdn_b, index);
            const bool second_completed = a_first ? measure_stage(*processors_b, result.fdn_b, index)
                                                  : measure_stage(*processors_a, result.fdn_a, index);
            if (!first_completed || !second_completed || stop_token.stop_requested())
            {
                return std::unexpected(BenchmarkError{.message = "Benchmark canceled"});
            }
        }
        return result;
    }
    catch (const std::exception& error)
    {
        return std::unexpected(
            BenchmarkError{.message = std::string("Performance measurement failed: ") + error.what()});
    }
}

FdnBenchmarkRunner::~FdnBenchmarkRunner()
{
    StopAndJoin();
}

std::expected<void, BenchmarkError> FdnBenchmarkRunner::Start(BenchmarkRequest request)
{
    {
        const std::scoped_lock lock(mutex_);
        if (running_)
        {
            return std::unexpected(BenchmarkError{.message = "A performance measurement is already running"});
        }
    }

    if (thread_.joinable())
    {
        thread_.join();
    }

    {
        const std::scoped_lock lock(mutex_);
        running_ = true;
        outcome_.reset();
    }
    try
    {
        thread_ = std::jthread([this, request = std::move(request)](std::stop_token stop_token) {
            BenchmarkOutcome outcome = [&]() -> BenchmarkOutcome {
                try
                {
                    return MeasureFdnPair(request, stop_token);
                }
                catch (const std::exception& error)
                {
                    return std::unexpected(
                        BenchmarkError{.message = std::string("Performance measurement failed: ") + error.what()});
                }
                catch (...)
                {
                    return std::unexpected(BenchmarkError{.message = "Performance measurement failed unexpectedly"});
                }
            }();

            const std::scoped_lock lock(mutex_);
            outcome_ = std::move(outcome);
            running_ = false;
        });
    }
    catch (const std::exception& error)
    {
        const std::scoped_lock lock(mutex_);
        running_ = false;
        return std::unexpected(
            BenchmarkError{.message = std::string("Could not start performance measurement: ") + error.what()});
    }
    return {};
}

bool FdnBenchmarkRunner::IsRunning() const
{
    const std::scoped_lock lock(mutex_);
    return running_;
}

std::optional<BenchmarkOutcome> FdnBenchmarkRunner::ConsumeOutcome()
{
    const std::scoped_lock lock(mutex_);
    std::optional<BenchmarkOutcome> outcome = std::move(outcome_);
    outcome_.reset();
    return outcome;
}

void FdnBenchmarkRunner::StopAndJoin()
{
    if (thread_.joinable())
    {
        thread_.request_stop();
        thread_.join();
    }

    const std::scoped_lock lock(mutex_);
    running_ = false;
    outcome_.reset();
}

} // namespace fdn_sandbox::performance
