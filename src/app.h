#pragma once

#include <imgui.h>

#include <imfilebrowser.h>

#include "fdn_analyzer.h"
#include "notifications.h"
#include "optimization_gui.h"
#include <audio_utils/audio_file_manager.h>
#include <audio_utils/audio_manager.h>
#include <readerwriterqueue.h>

#include <sffdn/sffdn.h>

#include <array>
#include <atomic>
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
    bool DrawFDNExtras(bool force_update);
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

    struct AudioCallbackArgs
    {
        std::span<float> output_buffer;
        size_t frame_size;
        size_t num_channels;
    };

    void AudioCallback(AudioCallbackArgs callback_args);
    void ApplyPendingFDNUpdates();
    void AdoptPendingConvolutionReverb(size_t frame_size);

    struct AudioProcessingTimes
    {
        int64_t convolution_ns = 0;
        int64_t fdn_ns = 0;
    };

    struct AudioEngineBuffers
    {
        std::span<float> input;
        std::span<float> fdn_output;
        std::span<float> convolution_output;
    };

    struct PendingFDNUpdate
    {
        uint64_t generation = 0;
        std::unique_ptr<sfFDN::FDN> fdn;
    };

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

    AudioProcessingTimes ProcessAudioEngines(const AudioEngineBuffers& buffers);
    void ClearUnstableFDN(std::span<const float> fdn_output);
    std::pair<float, float> PrepareMix(int reverb_type, std::span<float> input);
    void MixAndMeter(std::span<float> output, std::span<const float> input, float gain, float wet_mix, float dry_mix);
    static void WriteOutput(std::span<float> output_buffer, std::span<const float> output, size_t num_channels);
    struct CpuUsageUpdate
    {
        int64_t duration_ns;
        size_t frame_size;
    };
    void UpdateCpuUsage(CpuUsageUpdate update);

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
    void ProcessFileBrowserSelections();
    void SaveSelectedImpulseResponse();
    void LoadSelectedConfiguration();
    void SaveSelectedConfiguration();
    void LoadSelectedRIR();

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
    std::unique_ptr<audio_manager> audio_manager_;
    std::unique_ptr<audio_file_manager> audio_file_manager_;

    std::unique_ptr<sfFDN::FDN> gui_fdn_;
    std::unique_ptr<sfFDN::FDN> audio_fdn_;
    std::unique_ptr<sfFDN::FDN> other_fdn_;

    moodycamel::ReaderWriterQueue<PendingFDNUpdate> fdn_update_queue_;
    moodycamel::ReaderWriterQueue<std::unique_ptr<sfFDN::FDN>> fdn_cleanup_queue_;
    uint64_t submitted_fdn_generation_ = 0;
    uint64_t active_audio_fdn_generation_ = 0;

    sfFDN::FDNConfig fdn_config_;

    sfFDN::FDNConfig fdn_config_A_;
    sfFDN::FDNConfig fdn_config_B_;

    enum class AudioState
    {
        Idle,
        ImpulseRequested,

    } audio_state_ = AudioState::Idle;

    std::atomic<float> audio_gain_ = 1.0f;
    std::atomic<float> dry_wet_mix_ = 0.5f;
    std::atomic<float> fdn_cpu_usage_ = 0.0f;
    static constexpr size_t kCpuUsageWindowSize = 12;
    std::array<float, kCpuUsageWindowSize> cpu_usage_samples_{};
    size_t cpu_usage_sample_index_ = 0;
    size_t cpu_usage_sample_count_ = 0;
    float cpu_usage_sum_ = 0.0f;
    float displayed_cpu_usage_ = 0.0f;
    float cpu_usage_display_timer_ = 0.0f;
    OutputMeterDisplayState output_meter_display_{};

    std::atomic<float> meter_rms_ = 0.0f;
    std::atomic<float> meter_peak_ = 0.0f;
    std::atomic<bool> meter_clipped_ = false;
    std::vector<float> meter_squared_samples_;
    size_t meter_sample_index_ = 0;
    double meter_sum_squares_ = 0.0;

    constexpr static int kFDN_REVERB = 0;
    constexpr static int kCONV_REVERB = 1;
    std::atomic<int> reverb_engine_ = kFDN_REVERB;

    std::atomic<float> fdn_dry_level_ = 0.5f;
    std::atomic<float> fdn_wet_level_ = 0.5f;

    // For convolution reverb
    std::atomic<float> conv_wet_level_ = 1.0f;

    std::atomic<uint32_t> direct_delay_ms_ = 0;
    sfFDN::Delay direct_delay_;

    int selected_audio_file_ = 0;
    uint32_t selected_config_slot_ = 0;
    bool reset_layout_requested_ = false;
    bool clipping_notification_active_ = false;
    std::atomic<uint64_t> fdn_instability_generation_ = 0;
    fdn_sandbox::NotificationCenter notification_center_;

    fdn_analysis::FDNAnalyzer fdn_analyzer_;
    OptimizationGUI optimization_gui_;

    ImGui::FileBrowser save_ir_browser;
    ImGui::FileBrowser load_config_browser;
    ImGui::FileBrowser save_config_browser;
    ImGui::FileBrowser load_rir_browser;

    audio_utils::analysis::STFTOptions stft_options_;
    enum class SpectrogramType : uint8_t
    {
        STFT,
        Mel
    } spectrogram_type_ = SpectrogramType::STFT;

    std::string loaded_rir_filename_;
    fdn_analysis::IRAnalyzer rir_analyzer_;

    std::unique_ptr<sfFDN::PartitionedConvolver> convolution_reverb_;
    std::unique_ptr<sfFDN::PartitionedConvolver> active_convolution_reverb_;
    int last_reverb_type_ = kFDN_REVERB;
    size_t cpu_highwater_mark_ns_ = 0;
};
