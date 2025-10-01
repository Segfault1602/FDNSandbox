#pragma once

#include <atomic>
#include <span>

#include <imgui.h>

#include <imfilebrowser.h>

#include <audio_utils/audio_file_manager.h>
#include <audio_utils/audio_manager.h>

#include "analysis/fdn_analyzer.h"

#include <sffdn/sffdn.h>

#include "fdn_config.h"

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
    void DrawVisualization();
    void DrawSpectrogram();
    void DrawSpectrum();
    void DrawAutocorrelation();
    void DrawFilterResponse();
    void DrawEnergyDecayCurve();
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

    FDNConfig fdn_config_;

    FDNConfig fdn_config_A_;
    FDNConfig fdn_config_B_;

    enum class AudioState
    {
        Idle,
        ImpulseRequested,

    } audio_state_ = AudioState::Idle;

    std::atomic<float> audio_gain_ = 1.0f;
    std::atomic<float> dry_wet_mix_ = 0.5f;
    std::atomic<float> fdn_cpu_usage_ = 0.0f;

    fdn_analysis::FDNAnalyzer fdn_analyzer_;

    ImGui::FileBrowser save_ir_browser;
    ImGui::FileBrowser load_config_browser;
    ImGui::FileBrowser save_config_browser;

    audio_utils::analysis::SpectrogramInfo spectrogram_info_;
    enum class SpectrogramType : uint8_t
    {
        STFT,
        Mel
    } spectrogram_type_ = SpectrogramType::STFT;
};