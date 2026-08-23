#include "audio/audio_block_ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <Eigen/Core>

namespace fdn_sandbox::audio
{
namespace
{

using ConstArrayMap = Eigen::Map<const Eigen::ArrayXf, Eigen::Unaligned>;
using ArrayMap = Eigen::Map<Eigen::ArrayXf, Eigen::Unaligned>;

Eigen::Index EigenSize(std::size_t size) noexcept
{
    return static_cast<Eigen::Index>(size);
}

double Sum(std::span<const float> values) noexcept
{
    return ConstArrayMap(values.data(), EigenSize(values.size())).cast<double>().sum();
}

double SumSquares(std::span<const float> values) noexcept
{
    return ConstArrayMap(values.data(), EigenSize(values.size())).square().cast<double>().sum();
}

void CopySquares(std::span<float> destination, std::span<const float> source) noexcept
{
    assert(destination.size() == source.size());
    ArrayMap(destination.data(), EigenSize(destination.size())) =
        ConstArrayMap(source.data(), EigenSize(source.size())).square();
}

} // namespace

void Crossfade(std::span<const float> fade_in, std::span<const float> fade_out, std::span<float> output) noexcept
{
    assert(fade_in.size() == fade_out.size());
    assert(fade_in.size() == output.size());
    assert(output.size() > 1);

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(output.size() - 1);
        const float fadeout_gain = (t * t) * (3.0f - (2.0f * t));
        const float fadein_gain = 1.0f - fadeout_gain;
        output[i] = (fade_in[i] * fadein_gain) + (fade_out[i] * fadeout_gain);
    }
}

void Mix(std::span<float> wet_output, std::span<const float> dry_input, MixLevels levels) noexcept
{
    assert(wet_output.size() == dry_input.size());

    for (std::size_t i = 0; i < wet_output.size(); ++i)
    {
        wet_output[i] = levels.output_gain * ((levels.wet * wet_output[i]) + (levels.dry * dry_input[i]));
    }
}

void AddMonoToInterleaved(std::span<float> output, std::span<const float> mono, std::size_t channels) noexcept
{
    assert(channels > 0);
    assert(output.size() >= mono.size() * channels);

    for (std::size_t frame = 0; frame < mono.size(); ++frame)
    {
        const std::size_t offset = frame * channels;
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            output[offset + channel] += mono[frame];
        }
    }
}

OutputLevelMeter::OutputLevelMeter(std::size_t window_size)
    : squared_samples_(window_size)
{
    assert(window_size > 0);
}

MeterReading OutputLevelMeter::Measure(std::span<const float> samples) noexcept
{
    if (samples.empty())
    {
        return {
            .rms = std::sqrt(
                static_cast<float>(std::max(0.0, sum_squares_) / static_cast<double>(squared_samples_.size()))),
            .peak = 0.0f,
            .clipped = false,
        };
    }

    const ConstArrayMap sample_map(samples.data(), EigenSize(samples.size()));
    const float block_peak = sample_map.abs().maxCoeff();
    const std::size_t window_size = squared_samples_.size();

    // The ring stores squared samples so the rolling RMS can be updated without
    // rescanning the entire integration window on every audio block.
    if (samples.size() < window_size)
    {
        // Split at the ring boundary, subtract the values being replaced, then
        // use Eigen reductions and assignments for the incoming block.
        const std::size_t first_count = std::min(samples.size(), window_size - sample_index_);
        const std::size_t second_count = samples.size() - first_count;
        auto first_destination = std::span(squared_samples_).subspan(sample_index_, first_count);
        auto second_destination = std::span(squared_samples_).first(second_count);

        sum_squares_ -= Sum(first_destination);
        sum_squares_ -= Sum(second_destination);
        sum_squares_ += SumSquares(samples);

        CopySquares(first_destination, samples.first(first_count));
        CopySquares(second_destination, samples.subspan(first_count));
        sample_index_ = (sample_index_ + samples.size()) % window_size;
    }
    else [[unlikely]]
    {
        // Only the newest window can affect the result. Place it so sample_index_
        // still identifies the oldest value to overwrite on the next call.
        sample_index_ = (sample_index_ + samples.size()) % window_size;
        const auto retained_samples = samples.last(window_size);
        const std::size_t first_count = window_size - sample_index_;
        CopySquares(std::span(squared_samples_).subspan(sample_index_, first_count),
                    retained_samples.first(first_count));
        CopySquares(std::span(squared_samples_).first(sample_index_), retained_samples.subspan(first_count));
        sum_squares_ = SumSquares(retained_samples);
    }

    const double mean_square = std::max(0.0, sum_squares_) / static_cast<double>(squared_samples_.size());
    return {
        .rms = std::sqrt(static_cast<float>(mean_square)),
        .peak = block_peak,
        .clipped = block_peak >= 1.0f,
    };
}

CpuUsageReading CpuAverage::Push(std::int64_t duration_ns, std::size_t frame_size, float sample_rate) noexcept
{
    const float allowed_time = (1e9f / sample_rate) * static_cast<float>(frame_size);
    const float cpu_usage = static_cast<float>(duration_ns) / allowed_time;
    sum_ -= samples_[sample_index_];
    samples_[sample_index_] = cpu_usage;
    sum_ += cpu_usage;
    sample_index_ = (sample_index_ + 1) % kWindowSize;
    sample_count_ = std::min(sample_count_ + 1, kWindowSize);
    return {
        .current = cpu_usage,
        .average = sum_ / static_cast<float>(sample_count_),
    };
}

} // namespace fdn_sandbox::audio
