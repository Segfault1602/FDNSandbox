#pragma once

#include <imgui.h>

#include <imfilebrowser.h>

#include "fdn_analyzer.h"
#include "optimization_gui.h"
#include <audio_utils/audio_file_manager.h>
#include <audio_utils/audio_manager.h>
#include <readerwriterqueue.h>

#include <sffdn/sffdn.h>

#include <array>
#include <atomic>
#include <span>
#include <vector>

class FDNToolboxApp
{
  public:
    FDNToolboxApp();
    ~FDNToolboxApp();

    FDNToolboxApp(const FDNToolboxApp&) = delete;
    FDNToolboxApp& operator=(const FDNToolboxApp&) = delete;

    void loop();

  private:
    // Functions
    void DrawMainMenuBar();
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
    void DrawOptimizationLoss();

    void UpdateFDN();

    void AudioCallback(std::span<float> output_buffer, size_t frame_size, size_t num_channels);

    // Member variables
    std::unique_ptr<audio_manager> audio_manager_;
    std::unique_ptr<audio_file_manager> audio_file_manager_;

    std::unique_ptr<sfFDN::FDN> gui_fdn_;
    std::unique_ptr<sfFDN::FDN> audio_fdn_;
    std::unique_ptr<sfFDN::FDN> other_fdn_;

    moodycamel::ReaderWriterQueue<std::unique_ptr<sfFDN::FDN>> fdn_update_queue_;
    moodycamel::ReaderWriterQueue<std::unique_ptr<sfFDN::FDN>> fdn_cleanup_queue_;

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

    std::string loaded_rir_filename_{};
    fdn_analysis::IRAnalyzer rir_analyzer_;

    std::unique_ptr<sfFDN::PartitionedConvolver> convolution_reverb_;
};