#include "app.h"

#include "presets.h"
#include "settings.h"
#include "theme.h"

#include "imgui_internal.h"
#include <imgui.h>
#include <quill/LogMacros.h>

#include <chrono>
#include <cassert>
#include <stdexcept>

FDNToolboxApp::FDNToolboxApp(float ui_scale)
    : audio_engine_({.sample_rate = Settings::Instance().SampleRate()})
    , fdn_session_(presets::GetDefaultFDNConfig())
    , analysis_workspace_(Settings::Instance().SampleRate(), Settings::Instance().GetLogger())
    , optimization_gui_(Settings::Instance().GetLogger())
    , save_ir_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_config_browser(0)
    , save_config_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_rir_browser(0)
{
    LOG_INFO(Settings::Instance().GetLogger(), "Starting FDN Toolbox");

    fdn_sandbox::theme::Apply(ui_scale);
    fdn_sandbox::theme::InitializeFonts();

    save_ir_browser.SetTitle("Save Impulse Response");
    save_ir_browser.SetTypeFilters({".wav"});

    load_config_browser.SetTitle("Load FDN Configuration");
    load_config_browser.SetTypeFilters({".json"});

    save_config_browser.SetTitle("Save FDN Configuration");
    save_config_browser.SetTypeFilters({".json"});

    load_rir_browser.SetTitle("Load RIR File");
    load_rir_browser.SetTypeFilters({".wav"});

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
    DrawMainMenuBar();
    UpdateUiTelemetry();
    ProcessNotifications();
    DrawToolbar();
    notification_center_.DrawCriticalBanner();
    DrawStatusBar();

    static bool first_frame = true;
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
    ImGui::Begin("FDNToolbox", nullptr, window_flags);
    ImGui::PopStyleVar(2);

    const ImGuiID dockspace_id = ImGui::GetID("FDNSandboxDockSpace_v2");
    const bool reset_layout = std::exchange(reset_layout_requested_, false);
    if (reset_layout)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);
    }
    if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr)
    {
        BuildDefaultDockLayout(dockspace_id);
    }
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

    bool config_changed = false;
    config_changed = DrawFDNConfigurator();
    if (config_changed)
    {
        UpdateFDN();
    }

    DrawImpulseResponse();

    DrawAudioPlayer();

    DrawSettingsWindow();

    DrawOptimizationWindow();

    DrawFDNInfoWindow();

    DrawVisualization();

    ImGui::End(); // End the main window
    notification_center_.DrawToasts();

    if (first_frame || reset_layout)
    {
        ImGui::SetWindowFocus("Spectrogram");
    }
    first_frame = false;

    audio_engine_.FlushPendingCommands();
    audio_engine_.CollectRetiredObjects();
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
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "fdn-build-failed",
                              "Failed to build FDN", error.message, 6.0);
}
