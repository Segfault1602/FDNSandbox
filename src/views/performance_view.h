#pragma once

#include "performance/fdn_benchmark.h"

#include <array>
#include <optional>
#include <string>

namespace fdn_sandbox::views
{

class PerformanceView final
{
  public:
    enum class Action
    {
        Measure,
    };

    [[nodiscard]] std::optional<Action> Draw();
    void SetRunning();
    void SetResult(performance::BenchmarkResult result);
    void SetError(std::string message);
    [[nodiscard]] bool IsRunning() const noexcept;

    struct DisplayTiming
    {
        std::string per_block;
        std::string per_sample;
    };

    struct DisplayRow
    {
        const char* label = "";
        std::optional<DisplayTiming> fdn_a;
        std::optional<DisplayTiming> fdn_b;
        std::string difference;
    };

  private:
    void RebuildDisplayRows();

    bool running_ = false;
    std::optional<performance::BenchmarkResult> result_;
    std::array<DisplayRow, performance::kBenchmarkStageCount> display_rows_;
    std::string fdn_a_header_ = "FDN A";
    std::string fdn_b_header_ = "FDN B";
    std::string error_;
};

} // namespace fdn_sandbox::views
