#pragma once

#include <sffdn/sffdn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>

namespace fdn_sandbox::performance
{

enum class BenchmarkStage : std::uint8_t
{
    CompleteFdn,
    InputBlock,
    DelayBank,
    LoopFilters,
    FeedbackMatrix,
    OutputBlock,
    ToneCorrection,
    Count,
};

inline constexpr std::size_t kBenchmarkStageCount = static_cast<std::size_t>(BenchmarkStage::Count);
inline constexpr std::uint32_t kBenchmarkBlockSize = 256;

struct BenchmarkTiming
{
    double nanoseconds_per_block = 0.0;
    double nanoseconds_per_sample = 0.0;
    double lower_nanoseconds_per_sample = 0.0;
    double upper_nanoseconds_per_sample = 0.0;
};

struct FdnBenchmarkResult
{
    std::uint32_t block_size = 0;
    std::array<std::optional<BenchmarkTiming>, kBenchmarkStageCount> stages;
};

struct BenchmarkResult
{
    FdnBenchmarkResult fdn_a;
    FdnBenchmarkResult fdn_b;
};

struct BenchmarkRequest
{
    sfFDN::FDNConfig config_a;
    sfFDN::FDNConfig config_b;
    float sample_rate = 0.0f;
};

struct BenchmarkError
{
    std::string message;
};

using BenchmarkOutcome = std::expected<BenchmarkResult, BenchmarkError>;

[[nodiscard]] const char* BenchmarkStageName(BenchmarkStage stage) noexcept;
[[nodiscard]] BenchmarkOutcome MeasureFdnPair(const BenchmarkRequest& request, std::stop_token stop_token = {});

class FdnBenchmarkRunner final
{
  public:
    FdnBenchmarkRunner() = default;
    ~FdnBenchmarkRunner();

    FdnBenchmarkRunner(const FdnBenchmarkRunner&) = delete;
    FdnBenchmarkRunner& operator=(const FdnBenchmarkRunner&) = delete;
    FdnBenchmarkRunner(FdnBenchmarkRunner&&) = delete;
    FdnBenchmarkRunner& operator=(FdnBenchmarkRunner&&) = delete;

    [[nodiscard]] std::expected<void, BenchmarkError> Start(BenchmarkRequest request);
    [[nodiscard]] bool IsRunning() const;
    [[nodiscard]] std::optional<BenchmarkOutcome> ConsumeOutcome();
    void StopAndJoin();

  private:
    mutable std::mutex mutex_;
    std::jthread thread_;
    bool running_ = false;
    std::optional<BenchmarkOutcome> outcome_;
};

} // namespace fdn_sandbox::performance
