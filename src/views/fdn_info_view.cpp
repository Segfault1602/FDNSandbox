#include "views/fdn_info_view.h"

#include "icons.h"
#include "settings.h"
#include "utils.h"
#include "views/window_ids.h"

#include <imgui.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <string>

namespace fdn_sandbox::views
{

void FdnInfoView::Draw(const session::FdnSession& session, analysis::AnalysisWorkspace& workspace, float cpu_usage)
{
    if (!ImGui::Begin(ids::kFdnInfo))
    {
        ImGui::End();
        return;
    }

    const std::string matrix_name =
        std::visit([](const auto& matrix_options) { return utils::GetMatrixName(matrix_options.type); },
                   session.Draft().feedback_matrix_config);
    const auto& delays = session.Draft().delay_bank_config.delays;
    float min_delay = 0.0f;
    float max_delay = 0.0f;
    if (!delays.empty())
    {
        const auto [min_it, max_it] = std::ranges::minmax_element(delays);
        min_delay = *min_it;
        max_delay = *max_it;
    }

    const auto t60_data = workspace.Generated().GetT60Data(-5.0f, -50.0f);
    const auto echo_density_data = workspace.Generated().GetEchoDensityData(25, 10);
    const auto sample_rate = Settings::Instance().SampleRateAs<float>();

    if (ImGui::BeginTable("##FDNSummary", 2,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp))
    {
        const auto draw_row = [](const char* label, const std::string& value) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("%s", label);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(value.c_str());
        };

        draw_row("FDN Size", std::to_string(session.Draft().fdn_size));
        draw_row("Feedback Matrix", matrix_name);
        draw_row("Delay Count", std::to_string(delays.size()));
        draw_row("Delay Range (samples)", delays.empty() ? "--" : std::format("{:.0f} - {:.0f}", min_delay, max_delay));
        draw_row("Delay Range (ms)", delays.empty() ? "--"
                                                    : std::format("{:.2f} - {:.2f}", min_delay / sample_rate * 1000.0f,
                                                                  max_delay / sample_rate * 1000.0f));
        draw_row("Wideband T60", std::isfinite(t60_data.overall_t60.t60) && t60_data.overall_t60.t60 > 0.0f
                                     ? std::format("{:.2f} s", t60_data.overall_t60.t60)
                                     : "--");
        draw_row("Mixing Time", std::isfinite(echo_density_data.mixing_time) && echo_density_data.mixing_time > 0.0f
                                    ? std::format("{:.1f} ms", echo_density_data.mixing_time * 1000.0f)
                                    : "--");
        draw_row("Sample Rate", std::format("{} Hz", Settings::Instance().SampleRate()));
        draw_row("CPU", std::format("{:.1f}%", cpu_usage * 100.0f));
        ImGui::EndTable();
    }

    const nlohmann::json json_config = session.Draft();
    std::string json_str = json_config.dump(4);
    if (ImGui::CollapsingHeader("Raw Configuration (JSON)"))
    {
        if (ImGui::Button(fdn_sandbox::icons::CopyJson))
        {
            ImGui::SetClipboardText(json_str.c_str());
        }
        ImGui::InputTextMultiline("##FDNConfig", json_str.data(), json_str.size() + 1, ImVec2(-1, -1),
                                  ImGuiInputTextFlags_ReadOnly);
    }
    ImGui::End();
}

} // namespace fdn_sandbox::views
