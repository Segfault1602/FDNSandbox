#include "views/performance_view.h"

#include "performance/fdn_benchmark.h"
#include "theme.h"
#include "views/window_ids.h"

#include <imgui.h>

#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <string>
#include <utility>

namespace fdn_sandbox::views
{
namespace
{

PerformanceView::DisplayTiming FormatTiming(const performance::BenchmarkTiming& timing, double total_nanoseconds)
{
    std::string duration;
    if (timing.nanoseconds_per_block >= 1'000'000.0)
    {
        duration = std::format("{:.3f} ms/block", timing.nanoseconds_per_block / 1'000'000.0);
    }
    else if (timing.nanoseconds_per_block >= 1'000.0)
    {
        duration = std::format("{:.3f} us/block", timing.nanoseconds_per_block / 1'000.0);
    }
    else
    {
        duration = std::format("{:.1f} ns/block", timing.nanoseconds_per_block);
    }

    return PerformanceView::DisplayTiming{
        .per_block =
            total_nanoseconds > 0.0
                ? std::format("{:.1f}% ({})", timing.nanoseconds_per_block / total_nanoseconds * 100.0, duration)
                : std::move(duration),
        .per_sample = std::format("{:.2f} ns/sample (IQR {:.2f}-{:.2f})", timing.nanoseconds_per_sample,
                                  timing.lower_nanoseconds_per_sample, timing.upper_nanoseconds_per_sample),
    };
}

void DrawTiming(const std::optional<PerformanceView::DisplayTiming>& timing)
{
    if (!timing)
    {
        ImGui::TextDisabled("--");
        return;
    }

    ImGui::TextUnformatted(timing->per_block.c_str());
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("%s", timing->per_sample.c_str());
    }
}

} // namespace

std::optional<PerformanceView::Action> PerformanceView::Draw()
{
    if (!ImGui::Begin(ids::kPerformance))
    {
        ImGui::End();
        return std::nullopt;
    }

    std::optional<Action> action;
    ImGui::BeginDisabled(running_);
    if (ImGui::Button("Measure"))
    {
        action = Action::Measure;
    }
    ImGui::EndDisabled();

    if (running_)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("Measuring FDN A and FDN B...");
    }
    if (!error_.empty())
    {
        ImGui::TextColored(theme::Color(theme::ColorRole::StatusError), "%s", error_.c_str());
    }

#ifndef NDEBUG
    ImGui::TextColored(theme::Color(theme::ColorRole::StatusWarning),
                       "Debug/hardened build: use Release for representative timings.");
#endif

    if (result_)
    {
        ImGui::Spacing();
        ImGui::TextDisabled(
            "Snapshot results at 256 samples. Stage percentages are isolated time / complete FDN time and do not sum.");
        ImGui::TextDisabled("B vs A compares median time; positive means B is slower.");
        if (ImGui::BeginTable("##PerformanceResults", 4,
                              ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
                                  ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Stage");
            ImGui::TableSetupColumn(fdn_a_header_.c_str());
            ImGui::TableSetupColumn(fdn_b_header_.c_str());
            ImGui::TableSetupColumn("B vs A");
            ImGui::TableHeadersRow();

            for (const DisplayRow& row : display_rows_)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(row.label);
                ImGui::TableSetColumnIndex(1);
                DrawTiming(row.fdn_a);
                ImGui::TableSetColumnIndex(2);
                DrawTiming(row.fdn_b);
                ImGui::TableSetColumnIndex(3);
                if (row.difference.empty())
                {
                    ImGui::TextDisabled("--");
                }
                else
                {
                    ImGui::TextUnformatted(row.difference.c_str());
                }
            }
            ImGui::EndTable();
        }
    }

    ImGui::End();
    return action;
}

void PerformanceView::SetRunning()
{
    running_ = true;
    error_.clear();
}

void PerformanceView::SetResult(performance::BenchmarkResult result)
{
    running_ = false;
    error_.clear();
    result_ = std::move(result);
    RebuildDisplayRows();
}

void PerformanceView::SetError(std::string message)
{
    running_ = false;
    error_ = std::move(message);
}

bool PerformanceView::IsRunning() const noexcept
{
    return running_;
}

void PerformanceView::RebuildDisplayRows()
{
    if (!result_)
    {
        return;
    }

    fdn_a_header_ = std::format("FDN A ({} samples)", result_->fdn_a.block_size);
    fdn_b_header_ = std::format("FDN B ({} samples)", result_->fdn_b.block_size);
    const auto& total_a = result_->fdn_a.stages[static_cast<std::size_t>(performance::BenchmarkStage::CompleteFdn)];
    const auto& total_b = result_->fdn_b.stages[static_cast<std::size_t>(performance::BenchmarkStage::CompleteFdn)];

    for (std::size_t index = 0; index < display_rows_.size(); ++index)
    {
        DisplayRow& row = display_rows_[index];
        row = {};
        row.label = performance::BenchmarkStageName(static_cast<performance::BenchmarkStage>(index));

        const auto& timing_a = result_->fdn_a.stages[index];
        const auto& timing_b = result_->fdn_b.stages[index];
        if (timing_a && total_a)
        {
            row.fdn_a = FormatTiming(*timing_a, total_a->nanoseconds_per_block);
        }
        if (timing_b && total_b)
        {
            row.fdn_b = FormatTiming(*timing_b, total_b->nanoseconds_per_block);
        }
        if (timing_a && timing_b && timing_a->nanoseconds_per_sample > 0.0)
        {
            const double difference =
                (timing_b->nanoseconds_per_sample / timing_a->nanoseconds_per_sample - 1.0) * 100.0;
            const bool ranges_overlap =
                timing_a->lower_nanoseconds_per_sample <= timing_b->upper_nanoseconds_per_sample &&
                timing_b->lower_nanoseconds_per_sample <= timing_a->upper_nanoseconds_per_sample;
            if (ranges_overlap)
            {
                row.difference = "No measurable change";
            }
            else if (std::isfinite(difference))
            {
                row.difference = std::format("{:+.1f}%", difference);
            }
        }
    }
}

} // namespace fdn_sandbox::views
