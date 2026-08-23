#pragma once

#include <imgui.h>

#include "analysis/analysis_workspace.h"
#include "audio/audio_engine.h"
#include "files/file_dialogs.h"
#include "files/file_workflows.h"
#include "notifications.h"
#include "session/fdn_session.h"
#include "views/analysis_views.h"
#include "views/audio_view.h"
#include "views/fdn_configurator_view.h"
#include "views/fdn_info_view.h"
#include "views/optimization_view.h"
#include "views/settings_view.h"
#include "views/shell_view.h"

#include <cstdint>
#include <memory>
#include <utility>

class FDNToolboxApp
{
  public:
    explicit FDNToolboxApp(float ui_scale);
    ~FDNToolboxApp();

    FDNToolboxApp(const FDNToolboxApp&) = delete;
    FDNToolboxApp& operator=(const FDNToolboxApp&) = delete;
    FDNToolboxApp(FDNToolboxApp&&) = delete;
    FDNToolboxApp& operator=(FDNToolboxApp&&) = delete;

    void loop();

  private:
    void UpdateUiTelemetry();
    void ProcessNotifications();
    void HandleShellActions(const fdn_sandbox::views::ShellActions& actions);

    void UpdateFDN();
    void PublishFdnCommit(fdn_sandbox::session::FdnCommit commit);
    void ReportFdnBuildError(const fdn_sandbox::session::FdnBuildError& error);

    bool StartAudioStream();
    void StopAudioStream();

    void PlaySelectedAudioFile(int selected_audio_file);
    void ProcessFileDialogSelections();
    void HandleFileDialogSelection(const fdn_sandbox::files::SaveImpulseResponseSelection& selection);
    void HandleFileDialogSelection(const fdn_sandbox::files::LoadConfigurationSelection& selection);
    void HandleFileDialogSelection(const fdn_sandbox::files::SaveConfigurationSelection& selection);
    void HandleFileDialogSelection(const fdn_sandbox::files::LoadRirSelection& selection);

    fdn_sandbox::audio::AudioEngine audio_engine_;
    fdn_sandbox::session::FdnSession fdn_session_;

    bool first_frame_ = true;
    bool reset_layout_requested_ = false;
    bool clipping_notification_active_ = false;
    uint64_t reported_event_overflow_count_ = 0;
    fdn_sandbox::NotificationCenter notification_center_;

    fdn_sandbox::analysis::AnalysisWorkspace analysis_workspace_;
    fdn_sandbox::files::FileWorkflows file_workflows_;
    fdn_sandbox::files::FileDialogs file_dialogs_;
    fdn_sandbox::views::FdnConfiguratorView fdn_configurator_view_;
    fdn_sandbox::views::AudioView audio_view_;
    fdn_sandbox::views::SettingsView settings_view_;
    fdn_sandbox::views::FdnInfoView fdn_info_view_;
    fdn_sandbox::views::OptimizationView optimization_view_;
    fdn_sandbox::views::AnalysisViews analysis_views_;
    fdn_sandbox::views::ShellView shell_view_;
};
