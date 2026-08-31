#include "app.h"

#include "notifications.h"
#include "performance/fdn_benchmark.h"
#include "presets.h"
#include "session/fdn_session.h"
#include "settings.h"
#include "theme.h"
#include "views/fdn_info_view.h"
#include "views/performance_view.h"
#include "views/shell_view.h"
#include "views/window_ids.h"

#include "imgui_internal.h"
#include <cstdint>
#include <imgui.h>
#include <quill/LogMacros.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <ratio>
#include <string>
#include <utility>

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
    performance_runner_.StopAndJoin();
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
    UpdatePerformanceBenchmark();
    UpdateUiTelemetry();
    ProcessNotifications();
    HandleShellActions(
        shell_view_.DrawToolbar(audio_view_, audio_engine_, fdn_session_.ActiveSlot(), optimization_view_.IsActive()));
    notification_center_.DrawCriticalBanner();
    std::string reference_name;
    const auto& metadata = analysis_workspace_.ReferenceMetadata();
    if (metadata.has_value())
    {
        reference_name = std::filesystem::path(metadata->filename).filename().string();
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
    fdn_sandbox::views::FdnInfoView::Draw(fdn_session_, analysis_workspace_, audio_view_.DisplayedCpuUsage());
    if (const ImGuiWindow* fdn_info_window = ImGui::FindWindowByName(fdn_sandbox::views::ids::kFdnInfo);
        fdn_info_window != nullptr && fdn_info_window->DockId != 0)
    {
        ImGui::SetNextWindowDockID(fdn_info_window->DockId, ImGuiCond_FirstUseEver);
    }
    if (performance_view_.Draw() == fdn_sandbox::views::PerformanceView::Action::Measure)
    {
        StartPerformanceBenchmark();
    }
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

void FDNToolboxApp::StartPerformanceBenchmark()
{
    if (performance_runner_.IsRunning())
    {
        return;
    }

    fdn_sandbox::performance::BenchmarkRequest request{
        .config_a = fdn_session_.ActiveSlot() == fdn_sandbox::session::FdnSlot::A
                        ? fdn_session_.Draft()
                        : fdn_session_.ConfigForSlot(fdn_sandbox::session::FdnSlot::A),
        .config_b = fdn_session_.ActiveSlot() == fdn_sandbox::session::FdnSlot::B
                        ? fdn_session_.Draft()
                        : fdn_session_.ConfigForSlot(fdn_sandbox::session::FdnSlot::B),
        .sample_rate = Settings::Instance().SampleRateAs<float>(),
    };

    performance_audio_was_running_ = audio_engine_.IsRunning();
    if (performance_audio_was_running_)
    {
        StopAudioStream();
    }

    auto start_result = performance_runner_.Start(std::move(request));
    if (!start_result)
    {
        const std::string message = start_result.error().message;
        performance_view_.SetError(message);
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "performance-start-failed",
                                  "Performance measurement failed", message, 6.0);
        if (performance_audio_was_running_ && !StartAudioStream())
        {
            notification_center_.Push(
                fdn_sandbox::NotificationSeverity::Error, "audio-restart-failed", "Audio restart failed",
                "The audio stream could not be restarted after the benchmark failed to start.", 6.0);
        }
        performance_audio_was_running_ = false;
        return;
    }

    performance_view_.SetRunning();
}

void FDNToolboxApp::UpdatePerformanceBenchmark()
{
    if (performance_runner_.IsRunning())
    {
        // The audio configuration window can request a restart while measuring; keep the benchmark uncontended.
        if (audio_engine_.IsRunning())
        {
            StopAudioStream();
        }
        return;
    }

    auto outcome = performance_runner_.ConsumeOutcome();
    if (!outcome)
    {
        return;
    }

    if (*outcome)
    {
        performance_view_.SetResult(std::move(**outcome));
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "performance-completed",
                                  "Performance measurement completed", "FDN A and FDN B results are ready.", 6.0);
    }
    else
    {
        const std::string message = (*outcome).error().message;
        LOG_ERROR(Settings::Instance().GetLogger(), "Performance measurement failed: {}", message);
        performance_view_.SetError(message);
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "performance-failed",
                                  "Performance measurement failed", message, 6.0);
    }

    if (performance_audio_was_running_ && !StartAudioStream())
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "audio-restart-failed",
                                  "Audio restart failed",
                                  "The audio stream could not be restarted after performance measurement.", 6.0);
    }
    performance_audio_was_running_ = false;
}
