#include "app.h"

#include "presets.h"
#include "settings.h"
#include "theme.h"
#include "views/window_ids.h"

#include "imgui_internal.h"
#include <imgui.h>
#include <quill/LogMacros.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

FDNToolboxApp::FDNToolboxApp(float ui_scale)
    : audio_engine_({.sample_rate = Settings::Instance().SampleRate()})
    , fdn_session_(presets::GetDefaultFDNConfig())
    , analysis_workspace_(Settings::Instance().SampleRate(), Settings::Instance().GetLogger())
    , optimization_view_(Settings::Instance().GetLogger())
{
    LOG_INFO(Settings::Instance().GetLogger(), "Starting FDN Toolbox");

    fdn_sandbox::theme::Apply(ui_scale);
    fdn_sandbox::theme::InitializeFonts();

    UpdateFDN();

    StartAudioStream();
}
FDNToolboxApp::~FDNToolboxApp()
{
    audio_engine_.Shutdown();
}

bool FDNToolboxApp::StartAudioStream()
{
    const bool started = audio_engine_.Start();
    if (!started)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to start audio stream");
        return false;
    }

    LOG_INFO(Settings::Instance().GetLogger(), "Audio stream started");
    return true;
}

void FDNToolboxApp::StopAudioStream()
{
    audio_engine_.Stop();
}

void FDNToolboxApp::loop()
{
    HandleShellActions(shell_view_.DrawMainMenuBar(audio_view_, audio_engine_));
    ProcessFileDialogSelections();
    UpdateUiTelemetry();
    ProcessNotifications();
    HandleShellActions(
        shell_view_.DrawToolbar(audio_view_, audio_engine_, fdn_session_.ActiveSlot(), optimization_view_.IsActive()));
    notification_center_.DrawCriticalBanner();
    std::string reference_name;
    if (analysis_workspace_.HasReference())
    {
        reference_name = std::filesystem::path(analysis_workspace_.ReferenceMetadata()->filename).filename().string();
    }
    shell_view_.DrawStatusBar(
        fdn_sandbox::views::StatusBarData{
            .stream_running = audio_engine_.IsRunning(),
            .sample_rate = Settings::Instance().SampleRate(),
            .reference_name = reference_name,
        },
        audio_view_);

    constexpr ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(viewport->WorkSize, ImGuiCond_Always);
    ImGui::SetNextWindowViewport(viewport->ID);

    constexpr ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                              ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                              ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                              ImGuiWindowFlags_NoNavFocus;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin(fdn_sandbox::views::ids::kMainWindow, nullptr, window_flags);
    ImGui::PopStyleVar(2);

    const ImGuiID dockspace_id = ImGui::GetID(fdn_sandbox::views::ids::kDockspace);
    const bool reset_layout = std::exchange(reset_layout_requested_, false);
    if (reset_layout)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);
    }
    if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr)
    {
        fdn_sandbox::views::ShellView::BuildDefaultDockLayout(dockspace_id);
    }
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

    if (fdn_configurator_view_.Draw(fdn_session_))
    {
        UpdateFDN();
    }

    analysis_views_.DrawImpulseResponse(analysis_workspace_);
    audio_view_.DrawAudioPlayer(audio_engine_, analysis_workspace_.HasReference());
    settings_view_.Draw(analysis_workspace_);
    if (optimization_view_.Draw(fdn_session_.Draft(), analysis_workspace_.Reference().GetImpulseResponse()))
    {
        UpdateFDN();
    }
    fdn_info_view_.Draw(fdn_session_, analysis_workspace_, audio_view_.DisplayedCpuUsage());
    analysis_views_.DrawAll(analysis_workspace_);

    ImGui::End(); // End the main window
    notification_center_.DrawToasts();

    if (first_frame_ || reset_layout)
    {
        ImGui::SetWindowFocus(fdn_sandbox::views::ids::kSpectrogram);
    }
    first_frame_ = false;

    audio_engine_.FlushPendingCommands();
    audio_engine_.CollectRetiredObjects();
}

void FDNToolboxApp::HandleShellActions(const fdn_sandbox::views::ShellActions& actions)
{
    if (actions.file_dialog)
    {
        file_dialogs_.Open(*actions.file_dialog);
    }
    if (actions.save_slot)
    {
        fdn_session_.SaveTo(*actions.save_slot);
    }
    if (actions.switch_slot)
    {
        auto commit = fdn_session_.SwitchTo(*actions.switch_slot, Settings::Instance().SampleRateAs<float>());
        if (!commit)
        {
            ReportFdnBuildError(commit.error());
        }
        else
        {
            PublishFdnCommit(std::move(*commit));
        }
    }
    if (actions.play_audio_file)
    {
        PlaySelectedAudioFile(*actions.play_audio_file);
    }
    reset_layout_requested_ |= actions.reset_layout;
}

void FDNToolboxApp::UpdateFDN()
{
    LOG_INFO(Settings::Instance().GetLogger(), "Configuration changed, updating FDN...");
    auto start = std::chrono::high_resolution_clock::now();
    auto commit = fdn_session_.Commit(Settings::Instance().SampleRateAs<float>());
    if (!commit)
    {
        ReportFdnBuildError(commit.error());
        return;
    }
    PublishFdnCommit(std::move(*commit));

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    LOG_INFO(Settings::Instance().GetLogger(), "FDN updated in {} milliseconds", duration.count());
}

void FDNToolboxApp::PublishFdnCommit(fdn_sandbox::session::FdnCommit commit)
{
    const std::uint64_t generation = commit.generation;
    if (!audio_engine_.TryInstallFdn(generation, std::move(commit.audio_fdn)))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to publish FDN generation {}", generation);
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "fdn-publication-failed",
                                  "Failed to update FDN", "The audio engine rejected the new FDN.", 6.0);
        return;
    }

    analysis_workspace_.InstallGeneratedFdn(std::move(commit.analysis_fdn));
    const bool marked_accepted = fdn_session_.MarkAccepted(generation);
    assert(marked_accepted);
    static_cast<void>(marked_accepted);
    notification_center_.ClearCritical("fdn-instability");
}

void FDNToolboxApp::ReportFdnBuildError(const fdn_sandbox::session::FdnBuildError& error)
{
    LOG_ERROR(Settings::Instance().GetLogger(), "Failed to build FDN: {}", error.message);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "fdn-build-failed", "Failed to build FDN",
                              error.message, 6.0);
}
