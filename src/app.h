#pragma once

#include <atomic>
#include <span>

#include <imgui.h>

#include <imfilebrowser.h>

#include <audio_utils/audio_file_manager.h>
#include <audio_utils/audio_manager.h>
#include <audio_utils/ring_buffer.h>

#include "analysis/fdn_analyzer.h"
#include "optimization_gui.h"

#include <sffdn/sffdn.h>

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

    void AudioCallback(std::span<float> output_buffer, size_t frame_size, size_t num_channels);

    // Member variables
    std::unique_ptr<audio_manager> audio_manager_;
    std::unique_ptr<audio_file_manager> audio_file_manager_;

    bool show_tc_filter_designer_;

    std::unique_ptr<sfFDN::FDN> gui_fdn_;
    std::unique_ptr<sfFDN::FDN> audio_fdn_;
    std::unique_ptr<sfFDN::FDN> other_fdn_;

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

    constexpr static int kFDN_REVERB = 0;
    constexpr static int kCONV_REVERB = 1;
    std::atomic<int> reverb_engine_ = kFDN_REVERB;

    std::atomic<float> fdn_dry_level_ = 0.5f;
    std::atomic<float> fdn_wet_level_ = 0.5f;

    // For convolution reverb
    std::atomic<float> conv_wet_level_ = 1.0f;

    std::atomic<uint32_t> pre_delay_ms_ = 0;
    sfFDN::Delay pre_delay_;

    fdn_analysis::FDNAnalyzer fdn_analyzer_;
    OptimizationGUI optimization_gui_;

    ImGui::FileBrowser save_ir_browser;
    ImGui::FileBrowser load_config_browser;
    ImGui::FileBrowser save_config_browser;
    ImGui::FileBrowser load_rir_browser;

    audio_utils::analysis::SpectrogramInfo spectrogram_info_;
    enum class SpectrogramType : uint8_t
    {
        STFT,
        Mel
    } spectrogram_type_ = SpectrogramType::STFT;

    ring_buffer<float> audio_output_buffer_;

    std::string loaded_rir_filename_{};
    fdn_analysis::IRAnalyzer rir_analyzer_;

    std::unique_ptr<sfFDN::PartitionedConvolver> convolution_reverb_;
};