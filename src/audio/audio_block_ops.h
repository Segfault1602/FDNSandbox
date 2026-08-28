#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace fdn_sandbox::audio
{

struct MixLevels
{
    float output_gain;
    float wet;
    float dry;
};

void Crossfade(std::span<const float> fade_in, std::span<const float> fade_out, std::span<float> output) noexcept;
void Mix(std::span<float> wet_output, std::span<const float> dry_input, MixLevels levels) noexcept;
void AddMonoToInterleaved(std::span<float> output, std::span<const float> mono, std::size_t channels) noexcept;

struct MeterReading
{
    float rms;
    float peak;
    bool clipped;
};

class OutputLevelMeter final
{
  public:
    explicit OutputLevelMeter(std::size_t window_size);

    MeterReading Measure(std::span<const float> samples) noexcept;

  private:
    std::vector<float> squared_samples_;
    std::size_t sample_index_ = 0;
    double sum_squares_ = 0.0;
};

struct CpuUsageReading
{
    float current;
    float average;
};

/**
 * @brief Bridges an arbitrary device block size to a processor that requires a fixed block size.
 *
 * Partitioned convolution only works at a fixed, power-of-two block size, but audio devices are
 * free to deliver any block size. Samples are accumulated until a full block is available, the
 * block is handed to the callback, and results are drained back out. This costs one block of
 * latency and is only used when the device block size differs from the processing block size.
 */
class FixedBlockAdapter final
{
  public:
    explicit FixedBlockAdapter(std::size_t block_size);

    void Reset() noexcept;
    [[nodiscard]] std::size_t BlockSize() const noexcept;

    /// Processes `input` into `output`, invoking `process(block_in, block_out)` once per full block.
    template <typename ProcessBlock>
    void Process(std::span<const float> input, std::span<float> output, ProcessBlock&& process) noexcept
    {
        assert(input.size() == output.size());
        for (std::size_t index = 0; index < input.size(); ++index)
        {
            pending_input_[fill_] = input[index];
            output[index] = ready_output_[fill_];
            ++fill_;
            if (fill_ == block_size_)
            {
                fill_ = 0;
                process(std::span<const float>(pending_input_), std::span<float>(ready_output_));
            }
        }
    }

  private:
    std::size_t block_size_;
    std::size_t fill_ = 0;
    std::vector<float> pending_input_;
    std::vector<float> ready_output_;
};

class CpuAverage final
{
  public:
    CpuUsageReading Push(std::int64_t duration_ns, std::size_t frame_size, float sample_rate) noexcept;

  private:
    static constexpr std::size_t kWindowSize = 12;
    std::array<float, kWindowSize> samples_{};
    std::size_t sample_index_ = 0;
    std::size_t sample_count_ = 0;
    float sum_ = 0.0f;
};

} // namespace fdn_sandbox::audio
