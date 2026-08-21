#include "app.h"

#include <audio_utils/fft_utils.h>

#include "presets.h"
#include "settings.h"
#include "theme.h"

#include "imgui_internal.h"
#include <imgui.h>
#include <quill/LogMacros.h>

#include <chrono>
#include <memory>
#include <stdexcept>

namespace
{
constexpr size_t kSystemBlockSize = 1024; // System block size for audio processing
constexpr float kMeterIntegrationSeconds = 0.3f;

audio_utils::analysis::STFTOptions MakeDefaultStftOptions()
{
    return {
#ifndef NDEBUG
        .fft_size = 1024,
        .overlap = 300,
        .window_size = 1024,
#else
        .fft_size = 2048,
        .overlap = 400,
        .window_size = 512,
#endif
        .window_type = audio_utils::FFTWindowType::Hann,
        .samplerate = Settings::Instance().SampleRate(),
    };
}
} // namespace

FDNToolboxApp::FDNToolboxApp(float ui_scale)
    : fdn_update_queue_(16)
    , fdn_cleanup_queue_(16)
    , direct_delay_(0, Settings::Instance().SampleRate())
    , fdn_analyzer_(Settings::Instance().SampleRate(), Settings::Instance().GetLogger())
    , optimization_gui_(Settings::Instance().GetLogger())
    , save_ir_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_config_browser(0)
    , save_config_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_rir_browser(0)
    , stft_options_(MakeDefaultStftOptions())
    , rir_analyzer_(Settings::Instance().SampleRate(), Settings::Instance().GetLogger())
{
    LOG_INFO(Settings::Instance().GetLogger(), "Starting FDN Toolbox");

    fdn_sandbox::theme::Apply(ui_scale);
    fdn_sandbox::theme::InitializeFonts();

    audio_manager_ = audio_manager::create_audio_manager();
    if (!audio_manager_)
    {
        throw std::runtime_error("Failed to create audio manager");
    }

    audio_file_manager_ = audio_file_manager::create_audio_file_manager();
    if (!audio_file_manager_)
    {
        throw std::runtime_error("Failed to create audio file manager");
    }

    meter_squared_samples_.resize(
        static_cast<size_t>(static_cast<float>(Settings::Instance().SampleRate()) * kMeterIntegrationSeconds));

    save_ir_browser.SetTitle("Save Impulse Response");
    save_ir_browser.SetTypeFilters({".wav"});

    load_config_browser.SetTitle("Load FDN Configuration");
    load_config_browser.SetTypeFilters({".json"});

    save_config_browser.SetTitle("Save FDN Configuration");
    save_config_browser.SetTypeFilters({".json"});

    load_rir_browser.SetTitle("Load RIR File");
    load_rir_browser.SetTypeFilters({".wav"});

    fdn_config_ = presets::GetDefaultFDNConfig();
    fdn_config_A_ = presets::GetDefaultFDNConfig();
    fdn_config_B_ = presets::GetDefaultFDNConfig();

    gui_fdn_ = presets::CreateDefaultFDN();
    UpdateFDN();

    // Initialize audio manager and start the audio stream
    if (!audio_manager_->start_audio_stream(
            audio_stream_option::kOutput,
            [this](std::span<float> output_buffer, size_t frame_size, size_t num_channels) {
                AudioCallback(output_buffer, frame_size, num_channels);
            },
            kSystemBlockSize))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to start audio stream");
    }
    LOG_INFO(Settings::Instance().GetLogger(), "Audio stream started");
}
FDNToolboxApp::~FDNToolboxApp()
{
    if (audio_manager_ && audio_manager_->is_audio_stream_running())
    {
        audio_manager_->stop_audio_stream();
    }
}
void FDNToolboxApp::loop()
{
    static bool first_time = true;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
    // because it would be confusing to have two docking targets within each others.
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
    const bool opt_fullscreen = true; // Set to true to use the entire viewport, false to use a smaller window
    if (opt_fullscreen)
    {
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(viewport->WorkSize, ImGuiCond_Always);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                        ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
    }
    else
    {
        dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
    }

    // // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
    // // and handle the pass-thru hole, so we ask Begin() to not render a background.
    if ((dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0)
    {
        window_flags |= ImGuiWindowFlags_NoBackground;
    }

    if (opt_fullscreen)
    {
        ImGui::PopStyleVar(2);
    }

    ImGui::Begin("FDNToolbox", nullptr, window_flags);

    const ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

    if (first_time)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

        ImGuiID dock_main_id = dockspace_id;

        const ImGuiID dock_id_fdn =
            ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
        const ImGuiID dock_id_ir =
            ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.33f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Impulse Response", dock_id_ir);
        ImGui::DockBuilderDockWindow("Audio Player", dock_id_ir);
        ImGui::DockBuilderDockWindow("Settings", dock_id_ir);
        ImGui::DockBuilderDockWindow("Optimization", dock_id_ir);
        ImGui::DockBuilderDockWindow("FDN Info", dock_id_ir);
        ImGui::DockBuilderDockWindow("FDN Configurator", dock_id_fdn);
        ImGui::DockBuilderDockWindow("Extras", dock_id_fdn);
        ImGui::DockBuilderDockWindow("Visualization", dock_main_id);
        ImGui::DockBuilderDockWindow("Spectrogram", dock_main_id);
        ImGui::DockBuilderDockWindow("Spectrum", dock_main_id);
        ImGui::DockBuilderDockWindow("Autocorrelation", dock_main_id);
        ImGui::DockBuilderDockWindow("Filter Response", dock_main_id);
        ImGui::DockBuilderDockWindow("Energy Decay Curve", dock_main_id);
        ImGui::DockBuilderDockWindow("Energy Decay Relief", dock_main_id);
        ImGui::DockBuilderDockWindow("RT60s", dock_main_id);
        ImGui::DockBuilderDockWindow("Cepstrum", dock_main_id);
        ImGui::DockBuilderDockWindow("Echo Density", dock_main_id);
        ImGui::DockBuilderDockWindow("Optimization Loss", dock_main_id);
        ImGui::DockBuilderFinish(dockspace_id);
    }

    DrawMainMenuBar();

    bool config_changed = false;
    config_changed = DrawFDNConfigurator();
    // config_changed |= DrawFDNExtras(config_changed);

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

    DrawOptimizationLoss();

    ImGui::End(); // End the main window

    if (first_time)
    {
        ImGui::SetWindowFocus("Spectrogram");
    }
    first_time = false;

    // Clean up old FDN instances that are no longer in use by the audio thread
    while (fdn_cleanup_queue_.pop())
    {
    }
}

void FDNToolboxApp::UpdateFDN()
{
    LOG_INFO(Settings::Instance().GetLogger(), "Configuration changed, updating FDN...");
    auto start = std::chrono::high_resolution_clock::now();
    fdn_config_.sample_rate = Settings::Instance().SampleRate();

    gui_fdn_ = sfFDN::CreateFDNFromConfig(fdn_config_);
    gui_fdn_->SetDirectGain(0.f);

    // Need to use CloneFDN() here because CreateFDNFromConfig is not always deterministic (e.g. when using random
    // matrices)
    fdn_analyzer_.SetFDN(gui_fdn_->CloneFDN());
    fdn_update_queue_.emplace(gui_fdn_->CloneFDN());

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    LOG_INFO(Settings::Instance().GetLogger(), "FDN updated in {} milliseconds", duration.count());
}
