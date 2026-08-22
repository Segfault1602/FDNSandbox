#include "app.h"

#include "icons.h"
#include "settings.h"
#include "theme.h"

#include "imgui_internal.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace
{
constexpr ImGuiWindowFlags kSideBarFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
} // namespace

void FDNToolboxApp::DrawToolbar()
{
#ifndef NDEBUG
    static bool fontaudio_verified = false;
    if (!fontaudio_verified)
    {
        IM_ASSERT(ImGui::GetFontBaked()->FindGlyphNoFallback(static_cast<ImWchar>(0xF169)) != nullptr);
        fontaudio_verified = true;
    }
#endif

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 side_bar_padding(style.WindowPadding.x, style.FramePadding.y);
    const float toolbar_height = ImGui::GetFrameHeight() + side_bar_padding.y * 2.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, side_bar_padding);
    const bool open =
        ImGui::BeginViewportSideBar("##FDNSandboxToolbar", viewport, ImGuiDir_Up, toolbar_height, kSideBarFlags);
    ImGui::PopStyleVar();
    if (!open)
    {
        ImGui::End();
        return;
    }

    DrawTransportControls(selected_audio_file_);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(170.0f);
    DrawAudioFileSelector(selected_audio_file_, "##ToolbarAudioFile");

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    DrawConfigurationSwitcher();

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    DrawQuickFileActions();

    if (optimization_gui_.IsActive())
    {
        const float pulse = 0.55f + 0.45f * std::sin(static_cast<float>(ImGui::GetTime()) * 4.0f);
        ImVec4 color = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotOptimization);
        color.w = pulse;
        ImGui::SameLine();
        ImGui::TextColored(color, "%s Optimizing", fdn_sandbox::icons::StatusOn);
    }

    ImGui::End();
}

void FDNToolboxApp::DrawQuickFileActions()
{
    if (ImGui::Button(fdn_sandbox::icons::LoadRir))
    {
        load_rir_browser.Open();
    }
    ImGui::SameLine();
    if (ImGui::Button(fdn_sandbox::icons::LoadConfig))
    {
        load_config_browser.Open();
    }
    ImGui::SameLine();
    if (ImGui::Button(fdn_sandbox::icons::SaveConfig))
    {
        save_config_browser.Open();
    }
    ImGui::SameLine();
    if (ImGui::Button(fdn_sandbox::icons::SaveIr))
    {
        save_ir_browser.Open();
    }
}

void FDNToolboxApp::DrawStatusBar()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 side_bar_padding(style.WindowPadding.x, style.FramePadding.y);
    const float status_bar_height = ImGui::GetFrameHeight() + side_bar_padding.y * 2.0f;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, side_bar_padding);
    const bool open =
        ImGui::BeginViewportSideBar("##FDNSandboxStatusBar", viewport, ImGuiDir_Down, status_bar_height, kSideBarFlags);
    ImGui::PopStyleVar();
    if (!open)
    {
        ImGui::End();
        return;
    }

    const bool stream_running = audio_manager_->is_audio_stream_running();
    ImGui::TextColored(fdn_sandbox::theme::Color(stream_running ? fdn_sandbox::theme::ColorRole::StatusOk
                                                                : fdn_sandbox::theme::ColorRole::StatusError),
                       "%s", fdn_sandbox::icons::StatusOn);
    ImGui::SameLine();
    ImGui::Text("%u Hz", Settings::Instance().SampleRate());
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (loaded_rir_filename_.empty())
    {
        ImGui::TextDisabled("No RIR");
    }
    else
    {
        const std::string rir_name = std::filesystem::path(loaded_rir_filename_).filename().string();
        ImGui::Text("RIR: %s", rir_name.c_str());
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    DrawCompactOutputMeter();

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("CPU %.1f%%", displayed_cpu_usage_ * 100.0f);

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    DrawFrameRateStatus();

    if (!stream_running)
    {
        ImGui::SameLine();
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError),
                           "Audio stream stopped");
    }

    ImGui::End();
}

void FDNToolboxApp::DrawCompactOutputMeter()
{
    ImGui::Text("RMS");
    ImGui::SameLine();
    DrawMeterBar(ImVec2(150.0f, 0.0f));
    if (output_meter_display_.clipping_warning_displayed)
    {
        ImGui::SameLine();
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError), "CLIP");
    }
}

void FDNToolboxApp::BuildDefaultDockLayout(ImGuiID dockspace_id)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodePos(dockspace_id, ImGui::GetMainViewport()->WorkPos);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->WorkSize);

    ImGuiID dock_main_id = dockspace_id;
    const ImGuiID dock_id_fdn = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
    const ImGuiID dock_id_ir = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.33f, nullptr, &dock_main_id);

    ImGui::DockBuilderDockWindow("FDN Configurator", dock_id_fdn);

    ImGui::DockBuilderDockWindow("Impulse Response", dock_id_ir);
    ImGui::DockBuilderDockWindow("Audio Player", dock_id_ir);
    ImGui::DockBuilderDockWindow("Settings", dock_id_ir);
    ImGui::DockBuilderDockWindow("FDN Info", dock_id_ir);

    ImGui::DockBuilderDockWindow("Spectrogram", dock_main_id);
    ImGui::DockBuilderDockWindow("Spectrum", dock_main_id);
    ImGui::DockBuilderDockWindow("Autocorrelation", dock_main_id);
    ImGui::DockBuilderDockWindow("Filter Response", dock_main_id);
    ImGui::DockBuilderDockWindow("Energy Decay Curve", dock_main_id);
    ImGui::DockBuilderDockWindow("Energy Decay Relief", dock_main_id);
    ImGui::DockBuilderDockWindow("RT60s", dock_main_id);
    ImGui::DockBuilderDockWindow("Cepstrum", dock_main_id);
    ImGui::DockBuilderDockWindow("Echo Density", dock_main_id);
    ImGui::DockBuilderDockWindow("Optimization", dock_main_id);

    ImGui::DockBuilderFinish(dockspace_id);
}
