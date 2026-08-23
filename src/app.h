#pragma once

#include <imgui.h>

#include "analysis/analysis_workspace.h"
#include "audio/audio_engine.h"
#include "files/file_dialogs.h"
#include "files/file_workflows.h"
#include "notifications.h"
#include "optimization_gui.h"
#include "session/fdn_session.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

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
    // Functions
    void DrawMainMenuBar();
    void DrawToolbar();
    void DrawStatusBar();
    void DrawQuickFileActions();
    void DrawCompactOutputMeter();
    void UpdateUiTelemetry();
    void ProcessNotifications();
    static void BuildDefaultDockLayout(ImGuiID dockspace_id);
    void DrawAudioDeviceGUI();
    bool DrawFDNConfigurator();
    void DrawImpulseResponse();
    void DrawAudioPlayer();
    void DrawSettingsWindow();
    void DrawOptimizationWindow();
    void DrawFDNInfoWindow();
    void DrawVisualization();
    void DrawSpectrogram();
    void DrawSpectrum();
    void DrawAutocorrelation();
    void DrawFilterResponse();
    void DrawEnergyDecayCurve();
    void DrawEnergyDecayRelief();
    void DrawCepstrum();
    void DrawEchoDensity();
    void DrawT60s();

    void UpdateFDN();
    void PublishFdnCommit(fdn_sandbox::session::FdnCommit commit);
    void ReportFdnBuildError(const fdn_sandbox::session::FdnBuildError& error);

    bool StartAudioStream();
    void StopAudioStream();

    struct OutputMeterDisplayState
    {
        float displayed_peak = 0.0f;
        float peak_hold_timer = 0.0f;
        float clipping_debounce_timer = 0.0f;
        float rms_db = -60.0f;
        float peak_db = -60.0f;
        float meter_fraction = 0.0f;
        bool clipping_warning_displayed = false;
    };

    void DrawTransportControls(int selected_audio_file);
    void PlaySelectedAudioFile(int selected_audio_file);
    void DrawAudioFileSelector(int& selected_audio_file, const char* label);
    void DrawAudioMixControls();
    void DrawConvolutionControls();
    void DrawOutputMeter();
    void DrawMeterBar(const ImVec2& size);
    void DrawAudioStreamControl();

    void DrawFileMenu();
    void DrawOptionsMenu(bool& show_audio_config_window);
    void DrawViewMenu();
    void DrawConfigurationSwitcher();
    static void DrawFrameRateStatus();
    void DrawAudioConfigurationWindow(bool& show_audio_config_window);
    void ProcessFileDialogSelections();
    void HandleFileDialogSelection(const fdn_sandbox::files::SaveImpulseResponseSelection& selection);
    void HandleFileDialogSelection(const fdn_sandbox::files::LoadConfigurationSelection& selection);
    void HandleFileDialogSelection(const fdn_sandbox::files::SaveConfigurationSelection& selection);
    void HandleFileDialogSelection(const fdn_sandbox::files::LoadRirSelection& selection);

    struct StructureSectionResult
    {
        bool config_changed = false;
        bool fdn_size_changed = false;
    };

    StructureSectionResult DrawStructureSection(uint32_t& random_seed);
    void DrawVisualOverviewSection(bool fdn_size_changed, int max_delay);
    bool DrawInputStageSection();
    bool DrawDelayNetworkSection();
    bool DrawFeedbackMatrixSection();
    bool DrawLoopFiltersSection();
    bool DrawOutputStageSection();
    bool DrawToneCorrectionSection();

    // Member variables
    fdn_sandbox::audio::AudioEngine audio_engine_;
    fdn_sandbox::session::FdnSession fdn_session_;

    float displayed_cpu_usage_ = 0.0f;
    float cpu_usage_display_timer_ = 0.0f;
    OutputMeterDisplayState output_meter_display_{};

    int selected_audio_file_ = 0;
    bool reset_layout_requested_ = false;
    bool clipping_notification_active_ = false;
    uint64_t reported_event_overflow_count_ = 0;
    fdn_sandbox::NotificationCenter notification_center_;

    fdn_sandbox::analysis::AnalysisWorkspace analysis_workspace_;
    OptimizationGUI optimization_gui_;
    fdn_sandbox::files::FileWorkflows file_workflows_;
    fdn_sandbox::files::FileDialogs file_dialogs_;
};
