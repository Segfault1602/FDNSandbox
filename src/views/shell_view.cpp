#include "views/shell_view.h"

#include "files/file_dialogs.h"
#include "icons.h"
#include "imgui.h"
#include "session/fdn_session.h"
#include "theme.h"
#include "views/audio_view.h"
#include "views/window_ids.h"

#include "imgui_internal.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace
{
constexpr ImGuiWindowFlags kSideBarFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

bool FancyButtons(const char* label1, const char* label2, std::uint32_t& selected)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    std::uint32_t new_selected = selected;
    if (selected == 0)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentActive));
    }
    if (ImGui::Button(label1, ImVec2(50, 0)))
    {
        new_selected = 0;
    }
    if (selected == 0)
    {
        ImGui::PopStyleColor(3);
    }

    ImGui::SameLine();
    if (selected == 1)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentActive));
    }
    if (ImGui::Button(label2, ImVec2(50, 0)))
    {
        new_selected = 1;
    }
    if (selected == 1)
    {
        ImGui::PopStyleColor(3);
    }
    ImGui::PopStyleVar();

    const bool changed = selected != new_selected;
    selected = new_selected;
    return changed;
}
} // namespace

namespace fdn_sandbox::views
{

ShellActions ShellView::DrawToolbar(AudioView& audio_view, audio::AudioEngine& engine, session::FdnSlot active_slot,
                                    bool optimization_active)
{
    ShellActions actions;
#ifndef NDEBUG
    if (!fontaudio_verified_)
    {
        IM_ASSERT(ImGui::GetFontBaked()->FindGlyphNoFallback(static_cast<ImWchar>(0xF169)) != nullptr);
        fontaudio_verified_ = true;
    }
#endif

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 side_bar_padding(style.WindowPadding.x, style.FramePadding.y);
    const float toolbar_height = ImGui::GetFrameHeight() + (side_bar_padding.y * 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, side_bar_padding);
    const bool open = ImGui::BeginViewportSideBar(ids::kToolbar, viewport, ImGuiDir_Up, toolbar_height, kSideBarFlags);
    ImGui::PopStyleVar();
    if (!open)
    {
        ImGui::End();
        return actions;
    }

    actions.play_audio_file = audio_view.DrawTransportControls(engine);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(170.0f);
    audio_view.DrawAudioFileSelector(engine, "##ToolbarAudioFile");

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    actions.switch_slot = DrawConfigurationSwitcher(active_slot);

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    actions.file_dialog = DrawQuickFileActions();

    if (optimization_active)
    {
        const float phase = static_cast<float>(ImGui::GetTime()) * 4.0f;
        const float pulse = 0.55f + (0.45f * std::sin(phase));
        ImVec4 color = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotOptimization);
        color.w = pulse;
        ImGui::SameLine();
        ImGui::TextColored(color, "%s Optimizing", fdn_sandbox::icons::StatusOn);
    }

    ImGui::End();
    return actions;
}

std::optional<files::FileDialogRequest> ShellView::DrawQuickFileActions()
{
    if (ImGui::Button(fdn_sandbox::icons::LoadRir))
    {
        return files::FileDialogRequest::LoadRir;
    }
    ImGui::SameLine();
    if (ImGui::Button(fdn_sandbox::icons::LoadConfig))
    {
        return files::FileDialogRequest::LoadConfiguration;
    }
    ImGui::SameLine();
    if (ImGui::Button(fdn_sandbox::icons::SaveConfig))
    {
        return files::FileDialogRequest::SaveConfiguration;
    }
    ImGui::SameLine();
    if (ImGui::Button(fdn_sandbox::icons::SaveIr))
    {
        return files::FileDialogRequest::SaveImpulseResponse;
    }
    return std::nullopt;
}

void ShellView::DrawStatusBar(const StatusBarData& data, const AudioView& audio_view)
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 side_bar_padding(style.WindowPadding.x, style.FramePadding.y);
    const float status_bar_height = ImGui::GetFrameHeight() + (side_bar_padding.y * 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, side_bar_padding);
    const bool open =
        ImGui::BeginViewportSideBar(ids::kStatusBar, viewport, ImGuiDir_Down, status_bar_height, kSideBarFlags);
    ImGui::PopStyleVar();
    if (!open)
    {
        ImGui::End();
        return;
    }

    ImGui::TextColored(fdn_sandbox::theme::Color(data.stream_running ? fdn_sandbox::theme::ColorRole::StatusOk
                                                                     : fdn_sandbox::theme::ColorRole::StatusError),
                       "%s", fdn_sandbox::icons::StatusOn);
    ImGui::SameLine();
    ImGui::Text("%u Hz", data.sample_rate);
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (data.reference_name.empty())
    {
        ImGui::TextDisabled("No RIR");
    }
    else
    {
        ImGui::Text("RIR: %.*s", static_cast<int>(data.reference_name.size()), data.reference_name.data());
    }

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    audio_view.DrawCompactOutputMeter();

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    ImGui::Text("CPU %.1f%%", audio_view.DisplayedCpuUsage() * 100.0f);

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    DrawFrameRateStatus();

    if (!data.stream_running)
    {
        ImGui::SameLine();
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError),
                           "Audio stream stopped");
    }

    ImGui::End();
}

void ShellView::BuildDefaultDockLayout(ImGuiID dockspace_id)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodePos(dockspace_id, ImGui::GetMainViewport()->WorkPos);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->WorkSize);

    ImGuiID dock_main_id = dockspace_id;
    const ImGuiID dock_id_fdn = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
    const ImGuiID dock_id_ir = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.33f, nullptr, &dock_main_id);

    ImGui::DockBuilderDockWindow(ids::kFdnConfigurator, dock_id_fdn);

    ImGui::DockBuilderDockWindow(ids::kImpulseResponse, dock_id_ir);
    ImGui::DockBuilderDockWindow(ids::kAudioPlayer, dock_id_ir);
    ImGui::DockBuilderDockWindow(ids::kSettings, dock_id_ir);
    ImGui::DockBuilderDockWindow(ids::kFdnInfo, dock_id_ir);
    ImGui::DockBuilderDockWindow(ids::kPerformance, dock_id_ir);

    ImGui::DockBuilderDockWindow(ids::kSpectrogram, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kSpectrum, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kAutocorrelation, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kFilterResponse, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kEnergyDecayCurve, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kEnergyDecayRelief, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kRt60s, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kCepstrum, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kEchoDensity, dock_main_id);
    ImGui::DockBuilderDockWindow(ids::kOptimization, dock_main_id);

    ImGui::DockBuilderFinish(dockspace_id);
}

ShellActions ShellView::DrawMainMenuBar(AudioView& audio_view, audio::AudioEngine& engine)
{
    ShellActions actions;
    if (ImGui::BeginMainMenuBar())
    {
        fdn_sandbox::theme::DrawWordmark();
        actions.file_dialog = DrawFileMenu();
        actions.save_slot = DrawOptionsMenu();
        actions.reset_layout = DrawViewMenu();
        ImGui::EndMainMenuBar();
    }

    DrawAudioConfigurationWindow(audio_view, engine);
    return actions;
}

std::optional<files::FileDialogRequest> ShellView::DrawFileMenu()
{
    if (!ImGui::BeginMenu("File"))
    {
        return std::nullopt;
    }

    std::optional<files::FileDialogRequest> request;
    if (ImGui::MenuItem("Save IR"))
    {
        request = files::FileDialogRequest::SaveImpulseResponse;
    }
    if (ImGui::MenuItem("Load Config"))
    {
        request = files::FileDialogRequest::LoadConfiguration;
    }
    if (ImGui::MenuItem("Save Config"))
    {
        request = files::FileDialogRequest::SaveConfiguration;
    }
    if (ImGui::MenuItem("Load RIR"))
    {
        request = files::FileDialogRequest::LoadRir;
    }
    ImGui::EndMenu();
    return request;
}

std::optional<session::FdnSlot> ShellView::DrawOptionsMenu()
{
    if (!ImGui::BeginMenu("Options"))
    {
        return std::nullopt;
    }

    ImGui::MenuItem(fdn_sandbox::icons::AudioSettings, nullptr, &show_audio_config_window_);
    ImGui::Separator();
    std::optional<session::FdnSlot> slot;
    if (ImGui::MenuItem("Save current to B"))
    {
        slot = session::FdnSlot::B;
    }
    if (ImGui::MenuItem("Save current to A"))
    {
        slot = session::FdnSlot::A;
    }
    ImGui::EndMenu();
    return slot;
}

bool ShellView::DrawViewMenu()
{
    if (!ImGui::BeginMenu("View"))
    {
        return false;
    }
    const bool reset_layout = ImGui::MenuItem("Reset Layout");
    ImGui::EndMenu();
    return reset_layout;
}

std::optional<session::FdnSlot> ShellView::DrawConfigurationSwitcher(session::FdnSlot active_slot)
{
    std::uint32_t selected = active_slot == session::FdnSlot::A ? 0U : 1U;
    if (!FancyButtons("A", "B", selected))
    {
        return std::nullopt;
    }
    return selected == 0 ? session::FdnSlot::A : session::FdnSlot::B;
}

void ShellView::DrawFrameRateStatus()
{
    const float fps = ImGui::GetIO().Framerate;
    if (fps >= 58.f)
    {
        ImGui::Text("FPS: %.1f", fps);
        return;
    }
    const auto color_role =
        fps >= 50.f ? fdn_sandbox::theme::ColorRole::StatusWarning : fdn_sandbox::theme::ColorRole::StatusError;
    ImGui::TextColored(fdn_sandbox::theme::Color(color_role), "FPS: %.1f", fps);
}

void ShellView::DrawAudioConfigurationWindow(AudioView& audio_view, audio::AudioEngine& engine)
{
    if (!show_audio_config_window_)
    {
        return;
    }
    if (ImGui::Begin(ids::kAudioConfiguration, &show_audio_config_window_))
    {
        audio_view.DrawAudioDeviceConfiguration(engine);
    }
    ImGui::End();
}

} // namespace fdn_sandbox::views
