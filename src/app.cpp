#include "app.h"

#include "fdn_analyzer.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft_utils.h>

#include "fdn_widget.h"
#include "optimization_gui.h"
#include "plot_ui.h"
#include "presets.h"
#include "settings.h"
#include "theme.h"
#include "utils.h"
#include "widget.h"

#include <sffdn/sffdn.h>

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include "imgui_internal.h"
#include <boost/dll/runtime_symbol_info.hpp>
#include <nlohmann/json.hpp>
#include <quill/LogMacros.h>
#include <quill/std/Vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <type_traits>
#include <vector>

namespace
{
constexpr size_t kSystemBlockSize = 1024; // System block size for audio processing
constexpr float kMeterIntegrationSeconds = 0.3f;

struct MelFormatterContext
{
    std::span<const float> mel_frequencies;
};

int MelFormatter(double value, char* buff, int size, void* data)
{
    MelFormatterContext context = *reinterpret_cast<MelFormatterContext*>(data);
    auto mel_index = static_cast<size_t>(value);
    if (mel_index >= context.mel_frequencies.size())
    {
        mel_index = context.mel_frequencies.size() - 1; // Clamp to the last index if out of bounds
    }
    auto mel = static_cast<uint32_t>(context.mel_frequencies[mel_index]);
    std::span<char> out_string(buff, size);
    const std::format_to_n_result result = std::format_to_n(out_string.begin(), size, "{}", mel);
    *result.out = '\0';
    return std::strlen(buff);
}

bool FancyButtons(const std::string_view label1, const std::string_view label2, uint32_t& selected)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    uint32_t new_selected = selected;

    if (selected == 0)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentActive));
    }
    if (ImGui::Button(label1.data(), ImVec2(50, 0)))
    {
        new_selected = 0;
    }

    if (selected == 0)
    {
        ImGui::PopStyleColor(3);
    }
    ImGui::PopStyleVar();

    if (selected == 1)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentActive));
    }

    if (ImGui::Button(label2.data(), ImVec2(50, 0)))
    {
        new_selected = 1;
    }

    if (selected == 1)
    {
        ImGui::PopStyleColor(3);
    }

    bool changed = (selected != new_selected);
    selected = new_selected;
    return changed;
}

void Crossfade(std::span<const float> fade_in, std::span<const float> fade_out, std::span<float> output)
{
    assert(fade_in.size() == fade_out.size());
    assert(fade_in.size() == output.size());

    for (size_t i = 0; i < output.size(); ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(output.size() - 1);
        const float fadeout_gain =
            t * t * (3 - 2 * t); // from https://signalsmith-audio.co.uk/writing/2021/cheap-energy-crossfade/
        const float fadein_gain = 1.0f - fadeout_gain;
        output[i] = fade_in[i] * fadein_gain + fade_out[i] * fadeout_gain;
    }
}

void FillAttenuationFilterBank(sfFDN::AttenuationFilterBankOptions& filter_bank,
                               const sfFDN::attenuation_filter_variant_t& filter, uint32_t fdn_size)
{
    filter_bank.filter_configs.assign(fdn_size, filter);
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

    stft_options_ = {
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
        .samplerate = Settings::Instance().SampleRate()};

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

void FDNToolboxApp::AudioCallback(std::span<float> output_buffer, size_t frame_size, size_t num_channels)
{
    if (frame_size != kSystemBlockSize)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Frame size mismatch: expected {}, got {}", kSystemBlockSize,
                  frame_size);
        return;
    }

    const size_t approx_size_fdn = fdn_update_queue_.size_approx();
    if (approx_size_fdn > 0)
    {
        // The audio thread is faster than the UI thread so there should not be more than one FDN instance waiting in
        // the queue. This is just to be safe
        for (size_t i = 0; i < approx_size_fdn; ++i)
        {
            std::unique_ptr<sfFDN::FDN> next_fdn;
            if (fdn_update_queue_.try_dequeue(next_fdn))
            {
                fdn_cleanup_queue_.emplace(
                    std::move(audio_fdn_)); // Move the old FDN to the cleanup queue for deletion by the UI thread
                audio_fdn_ = std::move(next_fdn);
            }
        }
    }

    assert(audio_fdn_ != nullptr);

    audio_fdn_->SetDirectGain(0.f); // Direct gain is controlled by the dry/wet mix instead

    static std::unique_ptr<sfFDN::PartitionedConvolver> rir_reverb = nullptr;

    if (convolution_reverb_ != nullptr)
    {
        rir_reverb = std::move(convolution_reverb_);
        convolution_reverb_ = nullptr;
        if (rir_reverb->GetBlockSize() != frame_size)
        {
            LOG_ERROR(Settings::Instance().GetLogger(),
                      "RIR PartitionedConvolver block size {} does not match system block size {}",
                      rir_reverb->GetBlockSize(), frame_size);
            rir_reverb = nullptr;
        }
    }

    if (audio_fdn_ == nullptr)
    {
        // If no FDN is configured, fill the output buffer with silence
        std::ranges::fill(output_buffer, 0.0f);
        return;
    }

    const uint32_t new_delay_samples = direct_delay_ms_.load() * Settings::Instance().SampleRate() / 1000;
    if (direct_delay_.GetMaximumDelay() < new_delay_samples)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Pre-delay samples {} exceed maximum delay {}", new_delay_samples,
                  direct_delay_.GetMaximumDelay());
    }
    direct_delay_.SetDelay(new_delay_samples);

    std::array<float, kSystemBlockSize> input_data = {0.0f};
    std::array<float, kSystemBlockSize> fdn_output_data = {0.0f};
    std::array<float, kSystemBlockSize> conv_output_data = {0.0f};

    std::ranges::fill(input_data, 0.f);
    std::ranges::fill(fdn_output_data, 0.f);
    std::ranges::fill(conv_output_data, 0.f);

    sfFDN::AudioBuffer input_audio_buffer(kSystemBlockSize, 1, input_data);
    sfFDN::AudioBuffer fdn_output_audio_buffer(kSystemBlockSize, 1, fdn_output_data);
    sfFDN::AudioBuffer conv_output_audio_buffer(kSystemBlockSize, 1, conv_output_data);

    if (audio_state_ == AudioState::ImpulseRequested)
    {
        input_data[0] = 1.0f; // Set the first sample to 1.0 for impulse response
    }

    audio_file_manager_->process_block(input_data, kSystemBlockSize, 1);

    auto conv_start_time = std::chrono::steady_clock::now();
    if (rir_reverb != nullptr)
    {
        rir_reverb->Process(input_audio_buffer, conv_output_audio_buffer);
    }
    auto conv_end_time = std::chrono::steady_clock::now();
    auto conv_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(conv_end_time - conv_start_time).count();

    auto start_time = std::chrono::steady_clock::now();
    audio_fdn_->Process(input_audio_buffer, fdn_output_audio_buffer);
    auto end_time = std::chrono::steady_clock::now();
    auto fdn_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    // If the FDN output ever goes above |5.0|, it is likely unstable. Clear the fdn
    if (std::ranges::any_of(fdn_output_data, [](float sample) { return std::abs(sample) > 5.0f; }))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "FDN output exceeded |5.0|, likely unstable. Clearing FDN.");
        audio_fdn_->Clear();
    }

    static int last_reverb_type = kFDN_REVERB;
    const int reverb_type = reverb_engine_.load();

    if (last_reverb_type == kFDN_REVERB && reverb_type == kCONV_REVERB)
    {
        Crossfade(conv_output_data, fdn_output_data, conv_output_data);
    }
    else if (last_reverb_type == kCONV_REVERB && reverb_type == kFDN_REVERB)
    {
        Crossfade(fdn_output_data, conv_output_data, fdn_output_data);
    }

    last_reverb_type = reverb_type;

    float dry_mix = 0.f;
    float wet_mix = 0.5f;
    if (reverb_type == kFDN_REVERB)
    {
        // Apply pre-delay only if using FDN reverb
        direct_delay_.Process(input_audio_buffer, input_audio_buffer);
        wet_mix = fdn_wet_level_.load();
        dry_mix = fdn_dry_level_.load();
    }
    else
    {
        wet_mix = conv_wet_level_.load();
    }

    if (audio_state_ == AudioState::ImpulseRequested)
    {
        input_data[0] -= 1.0f; // Clear the impulse sample after processing
        audio_state_ = AudioState::Idle;
    }

    const float gain = audio_gain_.load();

    std::span<float> output_data = reverb_type == kFDN_REVERB ? fdn_output_data : conv_output_data;

    float block_peak = 0.0f;
    for (size_t i = 0; i < kSystemBlockSize; ++i)
    {
        output_data[i] = gain * ((wet_mix * output_data[i]) + (dry_mix * input_data[i]));

        const float squared_sample = output_data[i] * output_data[i];
        meter_sum_squares_ -= static_cast<double>(meter_squared_samples_[meter_sample_index_]);
        meter_squared_samples_[meter_sample_index_] = squared_sample;
        meter_sum_squares_ += static_cast<double>(squared_sample);
        meter_sample_index_ = (meter_sample_index_ + 1) % meter_squared_samples_.size();
        block_peak = std::max(block_peak, std::abs(output_data[i]));
    }

    const double mean_square = std::max(0.0, meter_sum_squares_) / static_cast<double>(meter_squared_samples_.size());
    const float rms = std::sqrt(static_cast<float>(mean_square));
    meter_rms_.store(rms, std::memory_order_relaxed);

    float previous_peak = meter_peak_.load(std::memory_order_relaxed);
    while (previous_peak < block_peak &&
           !meter_peak_.compare_exchange_weak(previous_peak, block_peak, std::memory_order_relaxed))
    {
    }
    if (block_peak >= 1.0f)
    {
        meter_clipped_.store(true, std::memory_order_relaxed);
    }

    for (size_t i = 0; i < kSystemBlockSize; ++i)
    {
        int idx_offset = i * num_channels;
        for (size_t j = 0; j < num_channels; ++j)
        {
            output_buffer[idx_offset + j] += output_data[i];
        }
    }

    const float allowed_time = (1e9 / Settings::Instance().SampleRate()) * frame_size; // in nanoseconds
    const float duration =
        reverb_type == kFDN_REVERB ? static_cast<float>(fdn_duration) : static_cast<float>(conv_duration);
    const float cpu_usage = duration / allowed_time;
    cpu_usage_sum_ -= cpu_usage_samples_[cpu_usage_sample_index_];
    cpu_usage_samples_[cpu_usage_sample_index_] = cpu_usage;
    cpu_usage_sum_ += cpu_usage;
    cpu_usage_sample_index_ = (cpu_usage_sample_index_ + 1) % kCpuUsageWindowSize;
    cpu_usage_sample_count_ = std::min(cpu_usage_sample_count_ + 1, kCpuUsageWindowSize);
    fdn_cpu_usage_.store(cpu_usage_sum_ / static_cast<float>(cpu_usage_sample_count_), std::memory_order_relaxed);

    static size_t highwater_mark = 0;
    if (duration > highwater_mark)
    {
        highwater_mark = duration;
        LOG_WARNING(Settings::Instance().GetLogger(), "New CPU highwater mark: {:.2f}%", cpu_usage * 100.f);
    }
}

void FDNToolboxApp::loop()
{
    static bool first_time = true;
    static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
    // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
    // because it would be confusing to have two docking targets within each others.
    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
    bool opt_fullscreen = true; // Set to true to use the entire viewport, false to use a smaller window
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

    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

    if (first_time)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

        ImGuiID dock_main_id = dockspace_id;

        ImGuiID dock_id_fdn = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
        ImGuiID dock_id_ir = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.33f, nullptr, &dock_main_id);

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
        ;
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

void FDNToolboxApp::DrawMainMenuBar()
{
    static bool show_audio_config_window = false;
    if (ImGui::BeginMainMenuBar())
    {
        fdn_sandbox::theme::DrawWordmark();

        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Save IR"))
            {
                save_ir_browser.Open();
            }

            if (ImGui::MenuItem("Load Config"))
            {
                load_config_browser.Open();
            }

            if (ImGui::MenuItem("Save Config"))
            {
                save_config_browser.Open();
            }

            if (ImGui::MenuItem("Load RIR"))
            {
                load_rir_browser.Open();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Options"))
        {
            ImGui::MenuItem("Audio Menu", nullptr, &show_audio_config_window);

            ImGui::Separator();

            if (ImGui::MenuItem("Save current to B"))
            {
                fdn_config_B_ = fdn_config_;
            }

            if (ImGui::MenuItem("Save current to A"))
            {
                fdn_config_A_ = fdn_config_;
            }

            ImGui::EndMenu();
        }

        static uint32_t selected_config = 0;
        if (FancyButtons("A", "B", selected_config))
        {
            if (selected_config == 0)
            {
                fdn_config_B_ = fdn_config_;
                fdn_config_ = fdn_config_A_;
            }
            else
            {
                fdn_config_A_ = fdn_config_;
                fdn_config_ = fdn_config_B_;
            }

            UpdateFDN();
        }

        float fps = ImGui::GetIO().Framerate;
        if (fps >= 58.f)
        {
            ImGui::Text("FPS: %.1f", fps);
        }
        else if (fps >= 50.f)
        {
            ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusWarning), "FPS: %.1f",
                               fps);
        }
        else
        {
            ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError), "FPS: %.1f", fps);
        }

        ImGui::Separator();
        if (!loaded_rir_filename_.empty())
        {
            ImGui::SameLine();
            ImGui::TextWrapped("Loaded RIR: %s", loaded_rir_filename_.c_str());
        }

        ImGui::EndMainMenuBar();
    }

    if (show_audio_config_window)
    {
        if (ImGui::Begin("Audio", &show_audio_config_window))
        {
            DrawAudioDeviceGUI();
            ImGui::End();
        }
    }

    save_ir_browser.Display();
    load_config_browser.Display();
    save_config_browser.Display();
    load_rir_browser.Display();

    if (save_ir_browser.HasSelected())
    {
        std::string filename = save_ir_browser.GetSelected().string();
        if (!filename.ends_with(".wav"))
        {
            filename += ".wav"; // Ensure the file has a .wav extension
        }
        utils::WriteAudioFile(filename, fdn_analyzer_.GetImpulseResponse(), Settings::Instance().SampleRate());
        save_ir_browser.ClearSelected();
    }

    if (load_config_browser.HasSelected())
    {
        std::string filename = load_config_browser.GetSelected().string();
        try
        {
            // sfFDN::FDNConfig::LoadFromFile(filename, fdn_config_);
            UpdateFDN();
        }
        catch (const std::exception& e)
        {
            LOG_ERROR(Settings::Instance().GetLogger(), "Error loading configuration: {}", e.what());
        }
        load_config_browser.ClearSelected();
    }

    if (save_config_browser.HasSelected())
    {
        std::string filename = save_config_browser.GetSelected().string();
        if (!filename.ends_with(".json"))
        {
            filename += ".json"; // Ensure the file has a .json extension
        }

        // try
        // {
        //     sfFDN::FDNConfig::SaveToFile(filename, fdn_config_);
        // }
        // catch (const std::exception& e)
        // {
        //     LOG_ERROR(Settings::Instance().GetLogger(), "Error saving configuration: {}", e.what());
        // }
        save_config_browser.ClearSelected();
    }

    if (load_rir_browser.HasSelected())
    {
        std::string filename = load_rir_browser.GetSelected().string();
        std::vector<float> buffer;
        int file_sample_rate = Settings::Instance().SampleRate();
        int file_num_channels = 0;
        if (audio_utils::audio_file::ReadWavFile(filename, buffer, file_sample_rate, file_num_channels))
        {
            if (file_num_channels != 1)
            {
                LOG_ERROR(Settings::Instance().GetLogger(), "RIR file must be mono. Loaded file has {} channels.",
                          file_num_channels);
            }
            else
            {
                LOG_INFO(Settings::Instance().GetLogger(), "Loaded RIR file: {} ({} Hz, {} channels, {} samples)",
                         filename, file_sample_rate, file_num_channels, buffer.size());
                loaded_rir_filename_ = filename;

                auto conv_reverb = std::make_unique<sfFDN::PartitionedConvolver>(kSystemBlockSize, buffer);

                LOG_INFO(Settings::Instance().GetLogger(), "Created PartitionedConvolver with {}",
                         conv_reverb->GetShortInfo());

                rir_analyzer_.SetImpulseResponse(std::move(buffer));
                convolution_reverb_ = std::move(conv_reverb);
            }
        }
        else
        {
            LOG_ERROR(Settings::Instance().GetLogger(), "Failed to load RIR file: {}", filename);
        }

        load_rir_browser.ClearSelected();
    }
}

bool FDNToolboxApp::DrawFDNConfigurator()
{
    static uint32_t random_seed = 0;
    static int max_delay = 6000;

    constexpr uint32_t kNMin = 4;
    constexpr uint32_t kNMax = 32;

    if (!ImGui::Begin("FDN Configurator"))
    {
        ImGui::End();
        return false; // If the window is not open, return early
    }

    bool config_changed = false;
    uint32_t N = fdn_config_.fdn_size;
    bool fdn_size_changed = false;

    if (fdn_sandbox::theme::BeginSection("Structure", true))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Configure the network size, deterministic seed, and matrix orientation.");

        static bool random_seed_checkbox = false;
        ImGui::Checkbox("Random Seed", &random_seed_checkbox);
        if (random_seed_checkbox)
        {
            ImGui::SameLine();
            if (ImGui::InputScalar("Seed", ImGuiDataType_U32, &random_seed, nullptr, nullptr, "%u"))
            {
                config_changed = true;
            }
        }
        else
        {
            random_seed = 0;
        }

        fdn_size_changed =
            ImGui::SliderScalar("N", ImGuiDataType_U32, (&N), &kNMin, &kNMax, nullptr, ImGuiSliderFlags_AlwaysClamp);
        if (fdn_size_changed)
        {
            utils::ResizeFDNConfig(fdn_config_, N);
        }
        config_changed |= fdn_size_changed;
        fdn_config_.fdn_size = N;

        static bool transpose = false;
        if (ImGui::Checkbox("Transpose", &transpose))
        {
            fdn_config_.transposed = transpose;
            config_changed = true;
        }

        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Visual Overview", true))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Inspect gain distribution, delay lengths, and the active feedback matrix.");
        if (fdn_size_changed)
        {
            fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Metadata,
                                     fdn_sandbox::theme::ColorRole::StatusWarning,
                                     "Overview will refresh after the resized FDN is rebuilt.");
        }
        else
        {
            DrawInputOutputGainsPlot(fdn_config_, gui_fdn_.get());
            DrawDelaysPlot(fdn_config_, max_delay);
            DrawFeedbackMatrixPlot(fdn_config_, gui_fdn_.get());
        }
        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Input Stage", true))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Set input gains and processors applied before the feedback loop.");
        config_changed |= DrawFDNOptions(fdn_config_.input_block_config.parallel_gains_config, fdn_config_);
        config_changed |=
            DrawSingleChannelProcessorList(fdn_config_.input_block_config.single_channel_processors, fdn_config_);
        config_changed |=
            DrawMultiChannelProcessorList(fdn_config_.input_block_config.multichannel_processors, fdn_config_);
        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Delay Network", true))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Configure delay lengths and interpolation for each feedback path.");
        config_changed |= DrawFDNOptions(fdn_config_.delay_bank_config, fdn_config_);
        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Feedback Matrix"))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Choose the matrix topology and whether it is implemented as cascaded stages.");

        bool is_cascaded =
            std::holds_alternative<sfFDN::CascadedFeedbackMatrixOptions>(fdn_config_.feedback_matrix_config);
        if (ImGui::Checkbox("Cascaded", &is_cascaded))
        {
            config_changed = true;
            if (is_cascaded)
            {
                sfFDN::CascadedFeedbackMatrixOptions cascaded_options;
                std::visit(
                    [&cascaded_options](auto&& options) {
                        cascaded_options.matrix_size = options.matrix_size;
                        cascaded_options.type = options.type;
                    },
                    fdn_config_.feedback_matrix_config);
                fdn_config_.feedback_matrix_config = cascaded_options;
            }
            else
            {
                sfFDN::ScalarFeedbackMatrixOptions scalar_options;
                std::visit(
                    [&scalar_options](auto&& options) {
                        scalar_options.matrix_size = options.matrix_size;
                        scalar_options.type = options.type;
                    },
                    fdn_config_.feedback_matrix_config);
                fdn_config_.feedback_matrix_config = scalar_options;
            }
        }
        config_changed |= DrawFDNOptions(fdn_config_.feedback_matrix_config, fdn_config_);
        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Loop Filters"))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Shape decay independently or apply a shared attenuation profile.");

        assert(fdn_config_.attenuation_filter_bank_config.has_value());
        bool filter_bank_changed = utils::NormalizeAttenuationFilterBank(fdn_config_);
        auto att_filterbank_options = fdn_config_.attenuation_filter_bank_config.value();
        const std::array<const char*, 4> filter_types = {"Homogenous", "2 Bands", "3 Bands", "10 Bands"};

        static bool independent_t60s = false;
        const bool independent_t60s_changed = ImGui::Checkbox("Independent T60s", &independent_t60s);

        bool loop_filters_changed = false;
        int selected_filter_type = 0;
        auto att_filter_options = att_filterbank_options.filter_configs.front();
        std::visit(
            [&selected_filter_type](auto&& options) {
                using T = std::decay_t<decltype(options)>;
                if constexpr (std::is_same_v<T, sfFDN::HomogenousFilterOptions>)
                {
                    selected_filter_type = 0;
                }
                else if constexpr (std::is_same_v<T, sfFDN::TwoBandFilterOptions>)
                {
                    selected_filter_type = 1;
                }
                else if constexpr (std::is_same_v<T, sfFDN::ThreeBandFilterOptions>)
                {
                    selected_filter_type = 2;
                }
                else if constexpr (std::is_same_v<T, sfFDN::TenBandFilterOptions>)
                {
                    selected_filter_type = 3;
                }
            },
            att_filter_options);

        const int previously_selected_filter_type = selected_filter_type;
        if (ImGui::Combo("Filter Type", &selected_filter_type, filter_types.data(), filter_types.size()) &&
            previously_selected_filter_type != selected_filter_type)
        {
            loop_filters_changed = true;
            switch (selected_filter_type)
            {
            case 0:
                att_filter_options = sfFDN::HomogenousFilterOptions{};
                break;
            case 1:
                att_filter_options = sfFDN::TwoBandFilterOptions{};
                break;
            case 2:
                att_filter_options = sfFDN::ThreeBandFilterOptions{};
                break;
            case 3:
                att_filter_options = sfFDN::TenBandFilterOptions{};
                break;
            }
        }

        if (!independent_t60s)
        {
            const bool shared_filter_changed = DrawFDNOptions(att_filter_options, fdn_config_);
            if (shared_filter_changed || loop_filters_changed || independent_t60s_changed)
            {
                FillAttenuationFilterBank(att_filterbank_options, att_filter_options, fdn_config_.fdn_size);
                filter_bank_changed = true;
            }
        }
        else
        {
            if (loop_filters_changed)
            {
                FillAttenuationFilterBank(att_filterbank_options, att_filter_options, fdn_config_.fdn_size);
                filter_bank_changed = true;
            }

            for (uint32_t i = 0; i < fdn_config_.fdn_size; ++i)
            {
                ImGui::PushID(i);
                filter_bank_changed |= DrawFDNOptions(att_filterbank_options.filter_configs[i], fdn_config_);
                ImGui::PopID();
            }
        }

        if (filter_bank_changed)
        {
            fdn_config_.attenuation_filter_bank_config = att_filterbank_options;
            config_changed = true;
        }

        config_changed |= DrawMultiChannelProcessorList(fdn_config_.loop_filter_configs, fdn_config_);
        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Output Stage"))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Set output gains and processors applied after the feedback loop.");
        config_changed |= DrawFDNOptions(fdn_config_.output_block_config.parallel_gains_config, fdn_config_);
        config_changed |=
            DrawSingleChannelProcessorList(fdn_config_.output_block_config.single_channel_processors, fdn_config_);
        config_changed |=
            DrawMultiChannelProcessorList(fdn_config_.output_block_config.multichannel_processors, fdn_config_);
        fdn_sandbox::theme::EndSection();
    }

    if (fdn_sandbox::theme::BeginSection("Tone Correction"))
    {
        fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                        fdn_sandbox::theme::ColorRole::TextSecondary,
                                        "Apply final single-channel tone-correction processing.");
        config_changed |= DrawSingleChannelProcessorList(fdn_config_.tone_correction_filters, fdn_config_);
        fdn_sandbox::theme::EndSection();
    }

    ImGui::End();
    return config_changed;
}

void FDNToolboxApp::DrawImpulseResponse()
{
    if (!ImGui::Begin("Impulse Response"))
    {
        ImGui::End();
        return;
    }

    constexpr float kMinDuration = 1.f;
    constexpr float kMaxDuration = 10.f;
    static float ir_duration = 1.f;
    if (ImGui::SliderScalar("IR Duration", ImGuiDataType_Float, &ir_duration, &kMinDuration, &kMaxDuration, "%.2f"))
    {
        Settings::Instance().SetIRDuration(ir_duration);
        fdn_analyzer_.SetImpulseResponseSize(ir_duration * Settings::Instance().SampleRate());
    }

    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        if (ImGui::Button("Match RIR Length"))
        {
            float rir_duration = static_cast<float>(rir_analyzer_.GetImpulseResponse().size()) /
                                 static_cast<float>(Settings::Instance().SampleRate());
            ir_duration = rir_duration;
            Settings::Instance().SetIRDuration(ir_duration);
            fdn_analyzer_.SetImpulseResponseSize(ir_duration * Settings::Instance().SampleRate());
        }
    }

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    const auto imp_response = fdn_analyzer_.GetImpulseResponse();
    const auto time_data = fdn_analyzer_.GetTimeData();
    const auto rir_response = rir_analyzer_.GetImpulseResponse();
    const auto rir_time_data = rir_analyzer_.GetTimeData();
    const bool show_rir_overlay = show_rir && !rir_response.empty();
    const ImPlotFlags plot_flags = show_rir_overlay ? ImPlotFlags_None : ImPlotFlags_NoLegend;

    if (ImPlot::BeginPlot("Impulse Response", ImVec2(-1, -1), plot_flags))
    {
        ImPlot::SetupAxes("Time (s)", "Amplitude", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.25f, 1.25f);

        if (imp_response.empty() || time_data.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No impulse response available");
        }
        else
        {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, time_data.back(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, time_data.back());
            fdn_sandbox::plot_ui::DrawHorizontalGuide(
                "##Zero Amplitude", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary),
                0.75f);

            ImPlotSpec fdn_spec = fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn, 1.7f);
            if (fdn_analyzer_.IsClipping())
            {
                fdn_spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError);
            }
            const size_t fdn_count = std::min(imp_response.size(), time_data.size());
            ImPlot::PlotLine("FDN Response", time_data.data(), imp_response.data(), fdn_count, fdn_spec);

            if (show_rir_overlay && !rir_time_data.empty())
            {
                const size_t rir_count = std::min(rir_response.size(), rir_time_data.size());
                ImPlot::PlotLine("Reference RIR", rir_time_data.data(), rir_response.data(), rir_count,
                                 fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.3f));
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::End(); // End the Impulse Response window
}

void FDNToolboxApp::DrawAudioPlayer()
{
    ImGui::Begin("Audio Player");

    if (ImGui::Button("Impulse"))
    {
        audio_state_ = AudioState::ImpulseRequested;
        LOG_INFO(Settings::Instance().GetLogger(), "Playing impulse response...");
    }

    constexpr const char* kAudioFiles[] = {"drumloop.wav", "guitar.wav", "bleepsandbloops.wav", "saxophone.wav"};
    static int selected_audio_file = 0;

    ImGui::SameLine();
    if (audio_file_manager_->get_state() == audio_file_manager::AudioPlayerState::kPlaying)
    {
        if (ImGui::Button("Stop"))
        {
            audio_file_manager_->stop(true);
        }
    }
    else
    {
        if (ImGui::Button("Play"))
        {
            auto exe_path = boost::dll::program_location();
            auto parent_path = exe_path.parent_path();
            auto audio_path = parent_path / "audio" / kAudioFiles[selected_audio_file];
            LOG_INFO(Settings::Instance().GetLogger(), "Playing audio file: {}", audio_path.string());
            if (audio_file_manager_->open_audio_file(audio_path.string()))
            {
                audio_file_manager_->play(true);
            }
            else
            {
                LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file.");
            }
        }
    }

    if (ImGui::BeginCombo("Audio File", kAudioFiles[selected_audio_file]))
    {
        for (int i = 0; i < IM_ARRAYSIZE(kAudioFiles); i++)
        {
            bool is_selected = (selected_audio_file == i);
            if (ImGui::Selectable(kAudioFiles[i], is_selected))
            {
                selected_audio_file = i;
                audio_file_manager_->stop(true);
            }
        }
        ImGui::EndCombo();
    }

    ImGui::PushItemWidth(200);
    static float gain = 1.0f;
    if (ImGui::SliderFloat("Gain", &gain, 0.0f, 10.0f, "%.2f"))
    {
        audio_gain_ = gain; // Update the audio gain
    }

    static float fdn_dry_level = 0.5f;
    if (ImGui::SliderFloat("FDN Direct Level", &fdn_dry_level, 0.0f, 1.0f, "%.2f"))
    {
        fdn_dry_level_ = fdn_dry_level;
    }

    static float fdn_wet_level = 0.5f;
    if (ImGui::SliderFloat("FDN Reverb Level", &fdn_wet_level, 0.0f, 1.0f, "%.2f"))
    {
        fdn_wet_level_ = fdn_wet_level;
    }

    uint32_t direct_delay_ms = direct_delay_ms_.load();
    constexpr uint32_t kMinDirectDelay = 0;
    constexpr uint32_t kMaxDirectDelay = 100;
    if (ImGui::SliderScalar("Direct Delay (ms)", ImGuiDataType_U32, &direct_delay_ms, &kMinDirectDelay,
                            &kMaxDirectDelay))
    {
        direct_delay_ms = std::clamp(direct_delay_ms, kMinDirectDelay, kMaxDirectDelay);
        direct_delay_ms_ = direct_delay_ms;
    }

    ImGui::PopItemWidth();

    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        int selected_reverb_engine = reverb_engine_.load();
        ImGui::RadioButton("FDN Reverb", &selected_reverb_engine, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Convolution Reverb", &selected_reverb_engine, 1);
        reverb_engine_.store(selected_reverb_engine);

        static float wet_level = 1.0f;
        ImGui::PushItemWidth(200);

        if (ImGui::SliderFloat("Convolution Level", &wet_level, 0.0f, 1.0f, "%.2f"))
        {
            conv_wet_level_ = wet_level;
        }
        ImGui::PopItemWidth();
    }

    constexpr float kMinMeterDb = -60.0f;
    constexpr float kPeakHoldSeconds = 1.0f;
    constexpr float kPeakReleaseDbPerSecond = 20.0f;
    const float delta_time = ImGui::GetIO().DeltaTime;
    const float rms_value = meter_rms_.load(std::memory_order_relaxed);
    const float latest_peak = meter_peak_.exchange(0.0f, std::memory_order_relaxed);

    static float displayed_peak = 0.0f;
    static float peak_hold_timer = 0.0f;
    static bool clipping_warning_displayed = false;
    static float clipping_debounce_timer = 0.0f;

    if (latest_peak >= displayed_peak)
    {
        displayed_peak = latest_peak;
        peak_hold_timer = kPeakHoldSeconds;
    }
    else if (peak_hold_timer > 0.0f)
    {
        peak_hold_timer = std::max(0.0f, peak_hold_timer - delta_time);
    }
    else
    {
        const float release_gain = std::pow(10.0f, -kPeakReleaseDbPerSecond * delta_time / 20.0f);
        displayed_peak = std::max(latest_peak, displayed_peak * release_gain);
    }

    if (meter_clipped_.exchange(false, std::memory_order_relaxed))
    {
        clipping_warning_displayed = true;
        clipping_debounce_timer = 0.0f;
    }
    else
    {
        clipping_debounce_timer += delta_time;
    }

    if (clipping_warning_displayed && clipping_debounce_timer >= 1.0f)
    {
        clipping_warning_displayed = false;
    }

    const float rms_db = rms_value > 0.0f ? 20.0f * std::log10(rms_value) : kMinMeterDb;
    const float peak_db = displayed_peak > 0.0f ? 20.0f * std::log10(displayed_peak) : kMinMeterDb;
    const float meter_fraction = std::clamp((rms_db - kMinMeterDb) / -kMinMeterDb, 0.0f, 1.0f);
    const std::string meter_overlay =
        rms_db <= kMinMeterDb ? std::format("< {:.0f} dBFS", kMinMeterDb) : std::format("{:.1f} dBFS", rms_db);

    ImGui::Text("RMS level:");
    ImGui::SameLine();
    if (clipping_warning_displayed)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError));
    }
    else if (rms_db < -12.0f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::MeterSafe));
    }
    else if (rms_db < -3.0f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::MeterWarning));
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::MeterHot));
    }
    ImGui::ProgressBar(meter_fraction, ImVec2(200.0f, 0.0f), meter_overlay.c_str());

    ImGui::PopStyleColor();
    ImGui::Text("Sample peak hold: %.1f dBFS", peak_db);
    if (clipping_warning_displayed)
    {
        ImGui::SameLine();
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError), "CLIP");
    }

    constexpr float kCpuUsageDisplayInterval = 0.25f;
    cpu_usage_display_timer_ += ImGui::GetIO().DeltaTime;
    if (cpu_usage_display_timer_ >= kCpuUsageDisplayInterval)
    {
        displayed_cpu_usage_ = fdn_cpu_usage_.load(std::memory_order_relaxed);
        cpu_usage_display_timer_ = 0.0f;
    }
    ImGui::Text("CPU Usage: %.1f%%", displayed_cpu_usage_ * 100.0f);

    ImGui::End();
}

void FDNToolboxApp::DrawSettingsWindow()
{
    if (!ImGui::Begin("Settings"))
    {
        ImGui::End();
        return;
    }

    ImGui::Text("Sample Rate: %u Hz", Settings::Instance().SampleRate());
    ImGui::Separator();

    ImGui::Text("Spectrogram Settings");
    ImGui::Separator();

    constexpr int kColOffset = 100;

    constexpr std::array kSpectrogramTypes = {"STFT", "Mel"};
    static int selected_spectrogram_type = static_cast<int>(spectrogram_type_);
    ImGui::Text("Type:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##Type", &selected_spectrogram_type, kSpectrogramTypes.data(),
                     static_cast<int>(kSpectrogramTypes.size())))
    {
        spectrogram_type_ = static_cast<SpectrogramType>(selected_spectrogram_type);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
        rir_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }
    constexpr std::array kFFTSizeOptions = {"512", "1024", "2048", "4096", "8192"};
    constexpr std::array kWindowSizeOptions = {"256", "512", "1024", "2048", "4096", "8192"};
    static int selected_fft_size_index = 1;    // Default to 1024
    static int selected_window_size_index = 2; // Default to 512
    static float overlap = 0.5f;

    ImGui::Text("FFT Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##FFTSize", &selected_fft_size_index, kFFTSizeOptions.data(),
                     static_cast<int>(kFFTSizeOptions.size())))
    {
        stft_options_.fft_size = 512 * (1 << selected_fft_size_index);
        if (stft_options_.fft_size < stft_options_.window_size)
        {
            stft_options_.window_size = stft_options_.fft_size;
            stft_options_.overlap = static_cast<uint32_t>(overlap * stft_options_.window_size);
            selected_window_size_index = selected_fft_size_index + 1;
        }

        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Window Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowSize", &selected_window_size_index, kWindowSizeOptions.data(),
                     static_cast<int>(kWindowSizeOptions.size())))
    {
        stft_options_.window_size = 256 * (1 << selected_window_size_index);
        stft_options_.overlap = static_cast<uint32_t>(overlap * stft_options_.window_size);
        if (stft_options_.window_size > stft_options_.fft_size)
        {
            stft_options_.fft_size = stft_options_.window_size;
            selected_fft_size_index = selected_window_size_index - 1;
        }
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Overlap:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderFloat("##Overlap", &overlap, 0.01f, 0.95f, "%.2f"))
    {
        stft_options_.overlap = static_cast<uint32_t>(overlap * stft_options_.window_size);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Window Type:");
    ImGui::SameLine(kColOffset);
    static int selected_window_type = 2; // Default to Hann
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowType", &selected_window_type, "Rectangular\0Hamming\0Hann\0Blackman\0"))
    {
        stft_options_.window_type = static_cast<audio_utils::FFTWindowType>(selected_window_type);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::SeparatorText("Style Settings");

    // ImPlotStyle& style = ImPlot::GetStyle();

    // float line_weight = style.LineWeight;
    // ImGui::SetNextItemWidth(200);
    // if (ImGui::SliderFloat("Line Weight", &line_weight, 1.0f, 5.0f))
    // {
    //     style.LineWeight = line_weight;
    // }

    ImGui::SetNextItemWidth(200);
    ImGui::ShowFontSelector("ImGui Font");

    ImGui::SetNextItemWidth(200);
    ImGui::ShowStyleSelector("ImGui Style");

    ImGui::SetNextItemWidth(200);
    ImPlot::ShowStyleSelector("ImPlot Style");

    ImGui::SetNextItemWidth(200);
    ImPlot::ShowColormapSelector("ImPlot Colormap");

    ImGui::End(); // End the Settings window
}

void FDNToolboxApp::DrawOptimizationWindow()
{
    if (!ImGui::Begin("Optimization"))
    {
        ImGui::End();
        return;
    }

    if (optimization_gui_.Draw(fdn_config_, rir_analyzer_.GetImpulseResponse()))
    {
        UpdateFDN();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawFDNInfoWindow()
{
    if (!ImGui::Begin("FDN Info"))
    {
        ImGui::End();
        return;
    }

    ImGui::Text("FDN Size: %u", fdn_config_.fdn_size);

    nlohmann::json json_config = fdn_config_;
    std::string json_str = json_config.dump(4); // Pretty print with 4
    ImGui::InputTextMultiline("##FDNConfig", &json_str[0], json_str.size(), ImVec2(-1, -1),
                              ImGuiInputTextFlags_ReadOnly);

    ImGui::End();
}

void FDNToolboxApp::DrawVisualization()
{
    DrawSpectrogram();
    DrawSpectrum();
    DrawCepstrum();
    DrawAutocorrelation();
    DrawFilterResponse();
    DrawEnergyDecayCurve();
    DrawEnergyDecayRelief();
    DrawT60s();
    DrawEchoDensity();
}

void FDNToolboxApp::DrawOptimizationLoss()
{
    if (!ImGui::Begin("Optimization Loss"))
    {
        ImGui::End();
        return;
    }

    optimization_gui_.PlotLossHistory();

    ImGui::End();
}

void FDNToolboxApp::DrawSpectrogram()
{
    if (!ImGui::Begin("Spectrogram"))
    {
        ImGui::End();
        return;
    }

    static float min_dB = -70.0f;
    static float max_dB = 0.0f;
    static std::vector<float> mels{};
    static MelFormatterContext ctx{.mel_frequencies = mels};

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    ImPlot::PushColormap(ImPlotColormap_Plasma);

    constexpr float kColorBarWidth = 100.0f;
    fdn_analysis::SpectrogramData spectrogram_data{};
    double tmax = 1.0;
    if (show_rir)
    {
        spectrogram_data = rir_analyzer_.GetSpectrogram(stft_options_, spectrogram_type_ == SpectrogramType::Mel);
        tmax = rir_analyzer_.GetImpulseResponseSize() / static_cast<double>(Settings::Instance().SampleRate());
    }
    else
    {
        spectrogram_data = fdn_analyzer_.GetSpectrogram(stft_options_, spectrogram_type_ == SpectrogramType::Mel);
        tmax = fdn_analyzer_.GetImpulseResponseSize() / static_cast<double>(Settings::Instance().SampleRate());
    }

    static bool previous_show_rir = show_rir;
    static SpectrogramType previous_spectrogram_type = spectrogram_type_;
    static uint32_t previous_bin_count = 0;
    static uint32_t previous_frame_count = 0;
    const bool dimensions_changed = show_rir != previous_show_rir || spectrogram_type_ != previous_spectrogram_type ||
                                    spectrogram_data.bin_count != previous_bin_count ||
                                    spectrogram_data.frame_count != previous_frame_count;

    if (ImPlot::BeginPlot("##Spectrogram", ImVec2(ImGui::GetCurrentWindow()->Size[0] - kColorBarWidth, -1)))
    {
        const double tmin = 0.0;
        double bin_count = Settings::Instance().SampleRate() / 2.0;
        const char* frequency_axis_label = "Frequency (Hz)";

        if (spectrogram_type_ == SpectrogramType::Mel && !spectrogram_data.data.empty())
        {
            mels = audio_utils::GetMelFrequencies(spectrogram_data.bin_count, 0.f,
                                                  Settings::Instance().SampleRate() / 2.f);
            ctx.mel_frequencies = mels;
            ImPlot::SetupAxisFormat(ImAxis_Y1, MelFormatter, &ctx);
            bin_count = spectrogram_data.bin_count;
            frequency_axis_label = "Mel Band Frequency (Hz)";
        }

        ImPlot::SetupAxes("Time (s)", frequency_axis_label, ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        const ImPlotCond view_condition = dimensions_changed ? ImPlotCond_Always : ImPlotCond_Once;
        ImPlot::SetupAxisLimits(ImAxis_X1, tmin, std::max(tmax, 0.001), view_condition);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(bin_count, 1.0), view_condition);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, tmin, std::max(tmax, 0.001));
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0.0, std::max(bin_count, 1.0));

        if (spectrogram_data.data.empty() || spectrogram_data.bin_count == 0 || spectrogram_data.frame_count == 0)
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No spectrogram data available");
        }
        else
        {
            ImPlotSpec spec{};
            spec.Flags = ImPlotHeatmapFlags_ColMajor;
            ImPlot::PlotHeatmap("##Spectrogram", spectrogram_data.data.data(), spectrogram_data.bin_count,
                                spectrogram_data.frame_count, min_dB, max_dB, nullptr, {tmin, 0}, {tmax, bin_count},
                                spec);
        }

        ImPlot::EndPlot();
    }

    ImGui::SameLine();
    ImGui::BeginGroup();
    fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Metadata, fdn_sandbox::theme::ColorRole::TextSecondary,
                             "dB");
    ImPlot::ColormapScale("##HeatScale", min_dB, max_dB, ImVec2(-1, -1));

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    {
        ImGui::OpenPopup("Range");
    }
    if (ImGui::BeginPopup("Range"))
    {
        ImGui::SliderFloat("Max", &max_dB, min_dB, 100);
        ImGui::SliderFloat("Min", &min_dB, -100, max_dB);
        ImGui::EndPopup();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Right-click to adjust the displayed dB range");
    }
    ImGui::EndGroup();
    ImPlot::PopColormap();

    previous_show_rir = show_rir;
    previous_spectrogram_type = spectrogram_type_;
    previous_bin_count = spectrogram_data.bin_count;
    previous_frame_count = spectrogram_data.frame_count;

    ImGui::End(); // End the Spectrogram window
}

void FDNToolboxApp::DrawSpectrum()
{
    if (!ImGui::Begin("Spectrum"))
    {
        ImGui::End();
        return;
    }

    static double early_rir_duration = 0.5; // 500 ms

    bool plot_type_changed = false;
    static int peak_radio = 0;
    plot_type_changed |= ImGui::RadioButton("Only Spectrum", &peak_radio, 0);
    ImGui::SameLine();
    plot_type_changed |= ImGui::RadioButton("Only Peaks", &peak_radio, 1);
    ImGui::SameLine();
    plot_type_changed |= ImGui::RadioButton("Both", &peak_radio, 2);
    ImGui::SameLine();
    plot_type_changed |= ImGui::RadioButton("Histogram", &peak_radio, 3);

    (void)plot_type_changed; // Suppress unused variable warning
    static bool logarithmic_frequency = true;
    if (peak_radio < 3)
    {
        ImGui::SameLine();
        ImGui::Checkbox("Log Frequency", &logarithmic_frequency);
    }

    // static float frequency_range_min = 0.f;
    // static float frequency_range_max = Settings::Instance().SampleRate() / 2.f;
    // ImGui::DragFloatRange2("Frequency Range", &frequency_range_min, &frequency_range_max);

    // static bool lock_freq_range = false;
    // ImGui::SameLine();
    // ImGui::Checkbox("Lock Range", &lock_freq_range);

    static bool show_rir = false;
    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        ImGui::Checkbox("Show RIR", &show_rir);
    }
    else
    {
        show_rir = false;
    }

    static float kRowRatios[] = {0.20f, 0.80f};
    if (ImPlot::BeginSubplots("Spectrum Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, kRowRatios))
    {
        if (ImPlot::BeginPlot("Impulse Response", ImVec2(), ImPlotFlags_NoLegend))
        {
            if (!show_rir)
            {
                DrawEarlyRIRPicker(fdn_analyzer_.GetImpulseResponse(), fdn_analyzer_.GetTimeData(), early_rir_duration);
            }
            else
            {
                DrawEarlyRIRPicker(rir_analyzer_.GetImpulseResponse(), rir_analyzer_.GetTimeData(), early_rir_duration);
            }
            ImPlot::EndPlot();
        }

        fdn_analysis::SpectrumData spectrum_data{};
        if (show_rir)
        {
            spectrum_data = rir_analyzer_.GetSpectrum(early_rir_duration);
        }
        else
        {
            spectrum_data = fdn_analyzer_.GetSpectrum(early_rir_duration);
        }

        static int previous_peak_radio = peak_radio;
        const bool plot_mode_changed = previous_peak_radio != peak_radio;
        if (plot_mode_changed && peak_radio == 3)
        {
            ImPlot::SetNextAxisToFit(ImAxis_Y1);
        }

        std::string plot_title = std::format("Spectrum ({} peaks)###Spectrum Plot", spectrum_data.peaks.size());
        if (ImPlot::BeginPlot(plot_title.c_str(), ImVec2(), ImPlotFlags_NoLegend))
        {
            if (spectrum_data.spectrum.empty() || spectrum_data.frequency_bins.empty())
            {
                fdn_sandbox::plot_ui::DrawEmptyState("No spectrum data available");
            }
            else if (peak_radio < 3)
            {
                const auto series_role =
                    show_rir ? fdn_sandbox::plot_ui::SeriesRole::Reference : fdn_sandbox::plot_ui::SeriesRole::Fdn;
                static bool previous_logarithmic_frequency = logarithmic_frequency;
                const bool frequency_scale_changed = previous_logarithmic_frequency != logarithmic_frequency;
                fdn_sandbox::plot_ui::SetupFrequencyAxis(
                    ImAxis_X1, logarithmic_frequency, spectrum_data.frequency_bins.back(), ImPlotAxisFlags_None,
                    plot_mode_changed || frequency_scale_changed ? ImPlotCond_Always : ImPlotCond_Once);
                previous_logarithmic_frequency = logarithmic_frequency;
                ImPlot::SetupAxis(ImAxis_Y1, "Magnitude (dB)");
                ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
                ImPlot::SetupAxisLimits(ImAxis_Y1, -60.0, 20.0,
                                        plot_mode_changed ? ImPlotCond_Always : ImPlotCond_Once);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 40.0);
                fdn_sandbox::plot_ui::DrawHorizontalGuide(
                    "##0 dB", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
                fdn_sandbox::plot_ui::DrawHorizontalGuide(
                    "##Noise Floor", -60.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator), 0.75f);

                if (peak_radio == 0 || peak_radio == 2)
                {
                    const size_t spectrum_count =
                        std::min(spectrum_data.frequency_bins.size(), spectrum_data.spectrum.size());
                    ImPlot::PlotLine("Spectrum", spectrum_data.frequency_bins.data(), spectrum_data.spectrum.data(),
                                     spectrum_count, fdn_sandbox::plot_ui::LineSpec(series_role, 1.7f));
                }
                if ((peak_radio == 1 || peak_radio == 2) && !spectrum_data.peaks.empty())
                {
                    const size_t peak_count = std::min(spectrum_data.peaks_freqs.size(), spectrum_data.peaks.size());
                    const float peak_marker_size = peak_radio == 1 ? 2.5f : 3.0f;
                    ImPlot::PlotScatter(
                        "Peaks", spectrum_data.peaks_freqs.data(), spectrum_data.peaks.data(), peak_count,
                        fdn_sandbox::plot_ui::MarkerSpec(series_role, ImPlotMarker_Circle, peak_marker_size, 1.0f));
                }
            }
            else
            {
                ImPlot::SetupAxes("Magnitude (dB)", "Count");
                ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Linear);
                ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
                ImPlot::SetupAxisLimits(ImAxis_X1, -60.0, 20.0,
                                        plot_mode_changed ? ImPlotCond_Always : ImPlotCond_Once);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -120.0, 40.0);
                if (spectrum_data.peaks.empty())
                {
                    fdn_sandbox::plot_ui::DrawEmptyState("No spectral peaks available");
                }
                else
                {
                    ImPlotSpec histogram_spec = fdn_sandbox::plot_ui::LineSpec(
                        show_rir ? fdn_sandbox::plot_ui::SeriesRole::Reference : fdn_sandbox::plot_ui::SeriesRole::Fdn);
                    histogram_spec.FillColor = histogram_spec.LineColor;
                    histogram_spec.FillAlpha = 0.65f;
                    ImPlot::PlotHistogram("Histogram", spectrum_data.peaks.data(), spectrum_data.peaks.size(),
                                          ImPlotBin_Sqrt, 0.95f, ImPlotRange(-60, 20), histogram_spec);
                }
            }

            previous_peak_radio = peak_radio;
            ImPlot::EndPlot();
        }
        ImPlot::EndSubplots();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawAutocorrelation()
{
    if (!ImGui::Begin("Autocorrelation"))
    {
        ImGui::End();
        return;
    }

    static double xcorr_duration = 0.25;

    static int selected_autocorr_type = 0;
    if (ImGui::RadioButton("Time", selected_autocorr_type == 0))
    {
        selected_autocorr_type = 0;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Spectral", selected_autocorr_type == 1))
    {
        selected_autocorr_type = 1;
    }

    static bool show_rir = false;

    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        ImGui::Checkbox("Show RIR", &show_rir);
    }
    else
    {
        show_rir = false;
    }

    static float kRowRatios[] = {0.2f, 0.8f};
    if (ImPlot::BeginSubplots("Autocorrelation Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, kRowRatios))
    {
        if (ImPlot::BeginPlot("Impulse Response", ImVec2(), ImPlotFlags_NoLegend))
        {
            if (show_rir)
            {
                DrawEarlyRIRPicker(rir_analyzer_.GetImpulseResponse(), rir_analyzer_.GetTimeData(), xcorr_duration);
            }
            else
            {
                DrawEarlyRIRPicker(fdn_analyzer_.GetImpulseResponse(), fdn_analyzer_.GetTimeData(), xcorr_duration);
            }
            xcorr_duration = std::clamp(xcorr_duration, 0.1, 0.9);
            ImPlot::EndPlot();
        }

        auto xcorr_data = fdn_analyzer_.GetAutocorrelation(xcorr_duration);

        fdn_analysis::AutocorrelationData rir_xcorr_data;
        if (show_rir)
        {
            rir_xcorr_data = rir_analyzer_.GetAutocorrelation(xcorr_duration);
        }

        std::span<const float> xcorr_span =
            (selected_autocorr_type == 0) ? xcorr_data.autocorrelation : xcorr_data.spectral_autocorrelation;

        std::span<const float> rir_xcorr_span =
            (selected_autocorr_type == 0) ? rir_xcorr_data.autocorrelation : rir_xcorr_data.spectral_autocorrelation;

        const ImPlotFlags plot_flags = show_rir ? ImPlotFlags_None : ImPlotFlags_NoLegend;
        if (ImPlot::BeginPlot("Autocorrelation", ImVec2(-1, -1), plot_flags))
        {
            static std::vector<float> lag_axis;
            static std::vector<float> rir_lag_axis;
            const bool time_autocorrelation = selected_autocorr_type == 0;
            auto update_lag_axis = [time_autocorrelation](std::vector<float>& axis, size_t size) {
                if (axis.size() != size)
                {
                    axis.resize(size);
                }
                const float scale = time_autocorrelation ? 1000.0f / Settings::Instance().SampleRate() : 1.0f;
                for (size_t i = 0; i < axis.size(); ++i)
                {
                    axis[i] = static_cast<float>(i) * scale;
                }
            };
            update_lag_axis(lag_axis, xcorr_span.size());
            update_lag_axis(rir_lag_axis, rir_xcorr_span.size());

            ImPlot::SetupAxes(selected_autocorr_type == 0 ? "Lag (ms)" : "Frequency-bin Lag", "Normalized Correlation");
            ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.5, 1.5);

            if (xcorr_span.empty() || lag_axis.empty())
            {
                fdn_sandbox::plot_ui::DrawEmptyState("No autocorrelation data available");
            }
            else
            {
                static int previous_autocorr_type = -1;
                static size_t previous_lag_size = 0;
                const bool autocorr_range_changed =
                    previous_autocorr_type != selected_autocorr_type || previous_lag_size != lag_axis.size();
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, lag_axis.back(),
                                        autocorr_range_changed ? ImPlotCond_Always : ImPlotCond_Once);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, lag_axis.back());
                previous_autocorr_type = selected_autocorr_type;
                previous_lag_size = lag_axis.size();
                fdn_sandbox::plot_ui::DrawVerticalGuide(
                    "##Zero Lag", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
                ImPlot::PlotLine("FDN Autocorrelation", lag_axis.data(), xcorr_span.data(), xcorr_span.size(),
                                 fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));

                if (show_rir && !rir_xcorr_span.empty())
                {
                    ImPlot::PlotLine("RIR Autocorrelation", rir_lag_axis.data(), rir_xcorr_span.data(),
                                     rir_xcorr_span.size(),
                                     fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.3f));
                }
            }

            ImPlot::EndPlot();
        }
        ImPlot::EndSubplots();
    }
    ImGui::End();
}

void FDNToolboxApp::DrawFilterResponse()
{
    if (!ImGui::Begin("Filter Response"))
    {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("Filter Responses"))
    {
        auto filter_data = fdn_analyzer_.GetFilterData();
        float min_mag = 0.f;
        for (const auto& mag_response : filter_data.mag_responses)
        {
            if (!mag_response.empty())
            {
                auto min_response = *std::ranges::min_element(mag_response);
                min_mag = std::min(min_response, min_mag);
            }
        }

        if (ImGui::BeginTabItem("Attenuation filters"))
        {
            if (ImPlot::BeginSubplots("Attenuation Filters Mag/Phase", 2, 1, ImVec2(-1, -1),
                                      ImPlotSubplotFlags_LinkAllX))
            {
                if (ImPlot::BeginPlot("Magnitude", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f),
                                      ImPlotFlags_None))
                {
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
                    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
                    ImPlot::SetupAxis(ImAxis_Y1, "Magnitude (dB)");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, std::min(-60.0f, min_mag * 1.2f), 5.0, ImPlotCond_Once);
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 20.0);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##0 dB", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);

                    if (filter_data.mag_responses.empty() || filter_data.frequency_bins.empty())
                    {
                        fdn_sandbox::plot_ui::DrawEmptyState("No attenuation-filter response available");
                    }
                    for (size_t i = 0; i < filter_data.mag_responses.size(); ++i)
                    {
                        const auto mag_response = filter_data.mag_responses[i];
                        std::string line_name = "Delay filter " + std::to_string(i + 1);
                        ImPlotSpec spec{};
                        spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(i);
                        spec.LineWeight = 1.4f;
                        const size_t response_count = std::min(filter_data.frequency_bins.size(), mag_response.size());
                        ImPlot::PlotLine(line_name.c_str(), filter_data.frequency_bins.data(), mag_response.data(),
                                         response_count, spec);
                    }
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_None))
                {
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
                    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
                    ImPlot::SetupAxis(ImAxis_Y1, "Phase (rad)");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -std::numbers::pi, std::numbers::pi, ImPlotCond_Once);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##0 rad", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##+pi", std::numbers::pi, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator),
                        0.6f);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##-pi", -std::numbers::pi, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator),
                        0.6f);

                    if (filter_data.phase_responses.empty() || filter_data.frequency_bins.empty())
                    {
                        fdn_sandbox::plot_ui::DrawEmptyState("No attenuation-filter phase available");
                    }
                    for (size_t i = 0; i < filter_data.phase_responses.size(); ++i)
                    {
                        const auto phase_response = filter_data.phase_responses[i];
                        std::string line_name = "Delay filter " + std::to_string(i + 1);
                        ImPlotSpec spec{};
                        spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(i);
                        spec.LineWeight = 1.4f;
                        const size_t response_count =
                            std::min(filter_data.frequency_bins.size(), phase_response.size());
                        ImPlot::PlotLine(line_name.c_str(), filter_data.frequency_bins.data(), phase_response.data(),
                                         response_count, spec);
                    }

                    ImPlot::EndPlot();
                }
                ImPlot::EndSubplots();
            }
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Tone Correction Filter"))
        {
            if (ImPlot::BeginSubplots("Tone Correction Filters Mag/Phase", 2, 1, ImVec2(-1, -1),
                                      ImPlotSubplotFlags_LinkAllX))
            {
                if (ImPlot::BeginPlot("Magnitude", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f),
                                      ImPlotFlags_NoLegend))
                {
                    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
                    ImPlot::SetupAxis(ImAxis_Y1, "Magnitude (dB)");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -60.0, 10.0, ImPlotCond_Once);
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 30.0);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##0 dB", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
                    if (filter_data.tc_mag_response.empty() || filter_data.frequency_bins.empty())
                    {
                        fdn_sandbox::plot_ui::DrawEmptyState("No tone-correction response available");
                    }
                    else
                    {
                        const size_t response_count =
                            std::min(filter_data.frequency_bins.size(), filter_data.tc_mag_response.size());
                        ImPlot::PlotLine("TC Filter", filter_data.frequency_bins.data(),
                                         filter_data.tc_mag_response.data(), response_count,
                                         fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
                    }
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_NoLegend))
                {
                    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
                    ImPlot::SetupAxis(ImAxis_Y1, "Phase (rad)");
                    ImPlot::SetupAxisLimits(ImAxis_Y1, -std::numbers::pi, std::numbers::pi, ImPlotCond_Once);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##0 rad", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##+pi", std::numbers::pi, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator),
                        0.6f);
                    fdn_sandbox::plot_ui::DrawHorizontalGuide(
                        "##-pi", -std::numbers::pi, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator),
                        0.6f);
                    if (filter_data.tc_phase_response.empty() || filter_data.frequency_bins.empty())
                    {
                        fdn_sandbox::plot_ui::DrawEmptyState("No tone-correction phase available");
                    }
                    else
                    {
                        const size_t response_count =
                            std::min(filter_data.frequency_bins.size(), filter_data.tc_phase_response.size());
                        ImPlot::PlotLine("TC Filter", filter_data.frequency_bins.data(),
                                         filter_data.tc_phase_response.data(), response_count,
                                         fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
                    }

                    ImPlot::EndPlot();
                }
                ImPlot::EndSubplots();
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawEnergyDecayCurve()
{
    static float decay_db_start = 5;
    static float decay_db_end = 25;

    if (!ImGui::Begin("Energy Decay Curve"))
    {
        ImGui::End();
        return;
    }

    ImGui::DragFloatRange2("T60 decay range", &decay_db_start, &decay_db_end, 1.0, 0, 60);
    decay_db_start = std::clamp(decay_db_start, 0.0f, decay_db_end - 1.0f);
    decay_db_end = std::clamp(decay_db_end, decay_db_start + 1.0f, 60.0f);

    static bool show_rir = false;
    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        ImGui::Checkbox("Show RIR", &show_rir);
    }
    else
    {
        show_rir = false;
    }

    auto edc_data = fdn_analyzer_.GetEnergyDecayCurveData();

    fdn_analysis::EnergyDecayCurveData rir_edc_data{};
    if (show_rir)
    {
        rir_edc_data = rir_analyzer_.GetEnergyDecayCurveData();
    }

    auto t60_data = fdn_analyzer_.GetT60Data(-decay_db_start, -decay_db_end);

    constexpr std::array<const char*, 9> octave_band_names = {"63 Hz", "125 Hz", "250 Hz", "500 Hz", "1 kHz",
                                                              "2 kHz", "4 kHz",  "8 kHz",  "16 kHz"};
    static std::array<bool, 9> octave_band_visibility{};
    for (auto i = 0; i < octave_band_visibility.size(); ++i)
    {
        ImGui::Checkbox(octave_band_names[i], &octave_band_visibility[i]);
        if (i < octave_band_visibility.size() - 1)
        {
            ImGui::SameLine();
        }
    }
    bool show_octaves_bands = std::ranges::any_of(octave_band_visibility, [](bool visible) { return visible; });

    std::string edc_title = std::format("Energy Decay Curve (T60: {:.2f} s)", t60_data.overall_t60.t60);
    if (ImPlot::BeginPlot(edc_title.c_str(), ImVec2(-1, -1), ImPlotFlags_None))
    {
        ImPlot::SetupLegend(ImPlotLocation_NorthEast,
                            show_octaves_bands ? ImPlotLegendFlags_Outside : ImPlotLegendFlags_None);
        ImPlot::SetupAxes("Time (s)", "Level (dB)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -90.0f, 5.0f, ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 10.0);

        const auto time_data = fdn_analyzer_.GetTimeData();
        if (edc_data.energy_decay_curve.empty() || time_data.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No energy-decay data available");
        }
        else
        {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, time_data.back(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, time_data.back());
            fdn_sandbox::plot_ui::DrawHorizontalGuide(
                "##-60 dB", -60.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator), 0.75f);
            if (std::isfinite(t60_data.overall_t60.t60) && t60_data.overall_t60.t60 > 0.0f)
            {
                ImPlot::Annotation(t60_data.overall_t60.t60, -60.0,
                                   fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent),
                                   ImVec2(8.0f, -8.0f), true, "T60 %.2f s", t60_data.overall_t60.t60);
            }

            if (!show_octaves_bands)
            {
                const size_t fdn_count = std::min(time_data.size(), edc_data.energy_decay_curve.size());
                ImPlot::PlotLine("FDN EDC", time_data.data(), edc_data.energy_decay_curve.data(), fdn_count,
                                 fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
                if (show_rir && !rir_edc_data.energy_decay_curve.empty())
                {
                    const auto rir_time_data = rir_analyzer_.GetTimeData();
                    const size_t rir_count = std::min(rir_time_data.size(), rir_edc_data.energy_decay_curve.size());
                    ImPlot::PlotLine("Reference RIR EDC", rir_time_data.data(), rir_edc_data.energy_decay_curve.data(),
                                     rir_count,
                                     fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.3f));
                }
            }
            else
            {
                for (size_t i = 0; i < octave_band_names.size(); ++i)
                {
                    if (!octave_band_visibility[i] || edc_data.edc_octaves[i].empty())
                    {
                        continue;
                    }
                    const size_t point_count = std::min(time_data.size(), edc_data.edc_octaves[i].size());
                    const size_t stride = std::max<size_t>(1, point_count / 2048);
                    ImPlotSpec spec{};
                    spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(i);
                    spec.LineWeight = 1.5f;
                    spec.Stride = static_cast<int>(stride * sizeof(float));
                    ImPlot::PlotLine(octave_band_names[i], time_data.data(), edc_data.edc_octaves[i].data(),
                                     static_cast<int>((point_count - 1) / stride + 1), spec);

                    if (show_rir && !rir_edc_data.edc_octaves[i].empty())
                    {
                        const auto rir_time_data = rir_analyzer_.GetTimeData();
                        const size_t rir_point_count =
                            std::min(rir_time_data.size(), rir_edc_data.edc_octaves[i].size());
                        if (rir_point_count == 0)
                        {
                            continue;
                        }
                        const size_t rir_stride = std::max<size_t>(1, rir_point_count / 2048);
                        ImPlotSpec rir_spec = spec;
                        rir_spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(i, 0.55f);
                        rir_spec.LineWeight = 1.0f;
                        rir_spec.Stride = static_cast<int>(rir_stride * sizeof(float));
                        const std::string label = "RIR " + std::string(octave_band_names[i]);
                        ImPlot::PlotLine(label.c_str(), rir_time_data.data(), rir_edc_data.edc_octaves[i].data(),
                                         static_cast<int>((rir_point_count - 1) / rir_stride + 1), rir_spec);
                    }
                }
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawEnergyDecayRelief()
{

    if (!ImGui::Begin("Energy Decay Relief"))
    {
        ImGui::End();
        return;
    }

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    fdn_analysis::EnergyDecayReliefData edr{};
    if (show_rir)
    {
        edr = rir_analyzer_.GetEnergyDecayReliefData();
    }
    else
    {
        edr = fdn_analyzer_.GetEnergyDecayReliefData();
    }

    if (edr.energy_decay_relief.empty() || edr.bin_count == 0 || edr.frame_count == 0)
    {
        fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Description,
                                 fdn_sandbox::theme::ColorRole::TextSecondary,
                                 "No energy-decay-relief data available.");
        ImGui::End();
        return;
    }

    fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Metadata, fdn_sandbox::theme::ColorRole::TextSecondary,
                             "Drag to rotate | Scroll to zoom");

    static std::vector<float> x_data;
    static std::vector<float> y_data;

    // constexpr uint32_t kDownsampleFactor = 1024;
    const uint32_t y_size = edr.bin_count;
    const uint32_t x_size = edr.frame_count;

    const size_t grid_size = x_size * y_size;
    if (edr.energy_decay_relief.size() < grid_size)
    {
        fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Description, fdn_sandbox::theme::ColorRole::StatusError,
                                 "Energy-decay-relief data dimensions are invalid.");
        ImGui::End();
        return;
    }

    if (x_data.size() != grid_size || y_data.size() != grid_size)
    {
        x_data.resize(grid_size);
        y_data.resize(grid_size);

        auto x_mdspan = utils::Span2D(x_data, y_size, x_size);
        auto y_mdspan = utils::Span2D(y_data, y_size, x_size);
        for (size_t i = 0; i < y_size; ++i)
        {
            for (size_t j = 0; j < x_size; ++j)
            {
                x_mdspan(i, j) = static_cast<float>(j * edr.hop_size) /
                                 static_cast<float>(Settings::Instance().SampleRate()); // Time in seconds
                y_mdspan(i, j) = static_cast<float>(i);                                 // Mel band index
            }
        }
    }

    static std::vector<float> z_data;
    z_data.resize(grid_size);
    auto z_mdspan = utils::Span2D(z_data, y_size, x_size);
    for (size_t i = 0; i < y_size; ++i)
    {
        for (size_t j = 0; j < x_size; ++j)
        {
            z_mdspan(i, j) = edr.energy_decay_relief[i + j * y_size];
            if (z_mdspan(i, j) < -80.0f)
            {
                z_mdspan(i, j) = -80.0f; // Clamp to -80 dB for better visualization
            }
        }
    }

    if (ImPlot3D::BeginPlot("Energy Decay Relief", ImVec2(-1, -1), ImPlot3DFlags_None))
    {
        ImPlot3D::PushColormap(ImPlotColormap_Viridis);
        ImPlot3D::PushStyleVar(ImPlot3DStyleVar_FillAlpha, 0.8f);
        ImPlot3D::SetupBoxScale(2.0, 1.0, 0.8);
        ImPlot3DSpec spec{};
        spec.Flags = ImPlot3DSurfaceFlags_None;
        spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary);
        ImPlot3D::SetupAxes("Time (s)", "Mel Band", "Level (dB)", ImPlot3DAxisFlags_AutoFit, ImPlot3DAxisFlags_AutoFit,
                            ImPlot3DAxisFlags_AutoFit);

        ImPlot3D::PlotSurface("EDR Surface", x_data.data(), y_data.data(), z_data.data(), x_size, y_size, 0.0, 0.0,
                              spec);

        ImPlot3D::PopStyleVar();
        ImPlot3D::PopColormap();
        ImPlot3D::EndPlot();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawCepstrum()
{
    if (!ImGui::Begin("Cepstrum"))
    {
        ImGui::End();
        return;
    }
    static double early_rir_duration = 0.5; // 500 ms

    static float kRowRatios[] = {0.2f, 0.8f};
    if (ImPlot::BeginSubplots("Cepstrum Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, kRowRatios))
    {
        if (ImPlot::BeginPlot("Impulse Response Cepstrum#", ImVec2(), ImPlotFlags_NoLegend))
        {
            DrawEarlyRIRPicker(fdn_analyzer_.GetImpulseResponse(), fdn_analyzer_.GetTimeData(), early_rir_duration);
            early_rir_duration = std::max(early_rir_duration, 0.1);
            ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("Cepstrum", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            auto cepstrum_data = fdn_analyzer_.GetCepstrum(early_rir_duration);
            static std::vector<float> quefrency_ms;
            if (quefrency_ms.size() != cepstrum_data.cepstrum.size())
            {
                quefrency_ms.resize(cepstrum_data.cepstrum.size());
                for (size_t i = 0; i < quefrency_ms.size(); ++i)
                {
                    quefrency_ms[i] =
                        static_cast<float>(i) / static_cast<float>(Settings::Instance().SampleRate()) * 1000.0f;
                }
            }

            ImPlot::SetupAxes("Quefrency (ms)", "Cepstral Amplitude", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            if (cepstrum_data.cepstrum.empty() || quefrency_ms.empty())
            {
                fdn_sandbox::plot_ui::DrawEmptyState("No cepstrum data available");
            }
            else
            {
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, quefrency_ms.back(), ImPlotCond_Once);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, quefrency_ms.back());
                ImPlot::PlotLine("Cepstrum", quefrency_ms.data(), cepstrum_data.cepstrum.data(),
                                 cepstrum_data.cepstrum.size(),
                                 fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
            }

            ImPlot::EndPlot();
        }
        ImPlot::EndSubplots();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawEchoDensity()
{
    if (!ImGui::Begin("Echo Density"))
    {
        ImGui::End();
        return;
    }

    static int window_size_ms = 25;
    ImGui::SetNextItemWidth(200);
    ImGui::InputInt("Window Size (ms)", &window_size_ms, 1, 10);
    window_size_ms = std::clamp(window_size_ms, 5, 250); // Clamp to a reasonable range

    static int hop_size_ms = 10;
    ImGui::SetNextItemWidth(200);
    ImGui::InputInt("Hop Size (ms)", &hop_size_ms, 1, 10);
    hop_size_ms = std::clamp(hop_size_ms, 1, window_size_ms); // Clamp to a reasonable range

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    auto echo_density_data = fdn_analyzer_.GetEchoDensityData(window_size_ms, hop_size_ms);
    auto ir = fdn_analyzer_.GetImpulseResponse();
    auto time_data = fdn_analyzer_.GetTimeData();
    const ImPlotFlags plot_flags = show_rir ? ImPlotFlags_None : ImPlotFlags_NoLegend;

    if (ImPlot::BeginPlot("Echo Density", ImVec2(-1, -1), plot_flags))
    {
        ImPlot::SetupAxes("Time (s)", "Normalized Value", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.5f, ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 2.0f);

        if (ir.empty() || time_data.empty() || echo_density_data.echo_density.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No echo-density data available");
        }
        else
        {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, time_data.back(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, time_data.back());
            ImPlotSpec ir_spec{};
            ir_spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary);
            ir_spec.LineWeight = 0.8f;
            ir_spec.FillAlpha = 0.45f;
            ImPlot::PlotLine("FDN Impulse Response", time_data.data(), ir.data(), std::min(time_data.size(), ir.size()),
                             ir_spec);
            const size_t echo_density_count =
                std::min(echo_density_data.sparse_indices.size(), echo_density_data.echo_density.size());
            ImPlot::PlotLine("FDN Echo Density", echo_density_data.sparse_indices.data(),
                             echo_density_data.echo_density.data(), echo_density_count,
                             fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn, 1.8f));

            if (echo_density_data.mixing_time < time_data.back())
            {
                fdn_sandbox::plot_ui::DrawVerticalGuide(
                    "FDN Mixing Time", echo_density_data.mixing_time,
                    fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotFdn), 1.5f);
                ImPlot::Annotation(echo_density_data.mixing_time, 1.25,
                                   fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotFdn),
                                   ImVec2(6.0f, 0.0f), true, "FDN %.0f ms", echo_density_data.mixing_time * 1000.0f);
            }

            if (show_rir)
            {
                auto rir_echo_density_data = rir_analyzer_.GetEchoDensityData(window_size_ms, hop_size_ms);
                if (!rir_echo_density_data.echo_density.empty())
                {
                    const size_t rir_echo_density_count = std::min(rir_echo_density_data.sparse_indices.size(),
                                                                   rir_echo_density_data.echo_density.size());
                    ImPlot::PlotLine("RIR Echo Density", rir_echo_density_data.sparse_indices.data(),
                                     rir_echo_density_data.echo_density.data(), rir_echo_density_count,
                                     fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.5f));

                    if (rir_echo_density_data.mixing_time < time_data.back())
                    {
                        fdn_sandbox::plot_ui::DrawVerticalGuide(
                            "RIR Mixing Time", rir_echo_density_data.mixing_time,
                            fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotReference), 1.3f);
                        ImPlot::Annotation(rir_echo_density_data.mixing_time, 1.05,
                                           fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotReference),
                                           ImVec2(6.0f, 0.0f), true, "RIR %.0f ms",
                                           rir_echo_density_data.mixing_time * 1000.0f);
                    }
                }
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawT60s()
{
    if (!ImGui::Begin("RT60s"))
    {
        ImGui::End();
        return;
    }

    const float db_start = -5;
    const float db_end = -50;

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    auto t60_data = fdn_analyzer_.GetT60Data(db_start, db_end);
    ImGui::Text("Wideband T60: %.2f s", t60_data.overall_t60.t60);
    const ImPlotFlags plot_flags = show_rir ? ImPlotFlags_None : ImPlotFlags_NoLegend;
    if (ImPlot::BeginPlot("RT60s", ImVec2(-1, -1), plot_flags))
    {
        fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
        ImPlot::SetupAxis(ImAxis_Y1, "RT60 (s)");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 3.0f, ImPlotCond_Once);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0.0f, 50.f);
        ImPlot::SetupAxisZoomConstraints(ImAxis_Y1, 0.0f, 50.f);

        if (t60_data.t60_octaves.empty() || t60_data.octave_band_frequencies.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No RT60 data available");
        }
        else
        {
            const size_t fdn_t60_count = std::min(t60_data.octave_band_frequencies.size(), t60_data.t60_octaves.size());
            ImPlot::PlotLine(
                "FDN RT60", t60_data.octave_band_frequencies.data(), t60_data.t60_octaves.data(), fdn_t60_count,
                fdn_sandbox::plot_ui::MarkerSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn, ImPlotMarker_Circle, 5.0f));

            if (show_rir)
            {
                auto rir_t60_data = rir_analyzer_.GetT60Data(db_start, db_end);
                if (!rir_t60_data.t60_octaves.empty())
                {
                    const size_t rir_t60_count =
                        std::min(rir_t60_data.octave_band_frequencies.size(), rir_t60_data.t60_octaves.size());
                    ImPlot::PlotLine("RIR RT60", rir_t60_data.octave_band_frequencies.data(),
                                     rir_t60_data.t60_octaves.data(), rir_t60_count,
                                     fdn_sandbox::plot_ui::MarkerSpec(fdn_sandbox::plot_ui::SeriesRole::Reference,
                                                                      ImPlotMarker_Square, 5.0f));
                }
            }
        }

        ImPlot::EndPlot();
    }
    ImGui::End();
}

void FDNToolboxApp::DrawAudioDeviceGUI()
{
    static std::vector<std::string> supported_audio_drivers = audio_manager_->get_supported_audio_drivers();
    static std::vector<std::string> output_devices = audio_manager_->get_output_devices_name();

    // Audio Drivers Combo
    ImGui::Text("Audio Drivers ");
    ImGui::SameLine();
    static int selected_audio_driver = 0;
    if (ImGui::BeginCombo("##Audio Drivers", audio_manager_->get_current_audio_driver().c_str()))
    {
        for (int i = 0; i < supported_audio_drivers.size(); i++)
        {
            bool is_selected = (selected_audio_driver == i);
            if (ImGui::Selectable(supported_audio_drivers[i].c_str(), is_selected))
            {
                selected_audio_driver = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching audio driver to {}", supported_audio_drivers[i]);
                audio_manager_->set_audio_driver(supported_audio_drivers[i]);
                // Refresh audio devices
                output_devices = audio_manager_->get_output_devices_name();
            }

            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    // Output Devices Combo
    ImGui::AlignTextToFramePadding();
    ImGui::Text("Output Devices");
    ImGui::SameLine();
    static int selected_output_device = 0;
    if (ImGui::BeginCombo("##Output Devices", output_devices[selected_output_device].c_str()))
    {
        for (int i = 0; i < output_devices.size(); i++)
        {
            bool is_selected = (selected_output_device == i);
            if (ImGui::Selectable(output_devices[i].c_str(), is_selected))
            {
                selected_output_device = i;
                LOG_INFO(Settings::Instance().GetLogger(), "Switching output device to {}", output_devices[i]);
                audio_manager_->set_output_device(output_devices[i]);
            }

            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    bool stream_running = audio_manager_->is_audio_stream_running();
    if (ImGui::Checkbox("Start/Stop Audio Stream", &stream_running))
    {
        if (stream_running)
        {
            if (!audio_manager_->start_audio_stream(
                    audio_stream_option::kOutput,
                    [this](std::span<float> output_buffer, size_t frame_size, size_t num_channels) {
                        AudioCallback(output_buffer, frame_size, num_channels);
                    },
                    kSystemBlockSize))
            {
                LOG_ERROR(Settings::Instance().GetLogger(), "Failed to start audio stream");
                stream_running = false;
            }
        }
        else
        {
            audio_manager_->stop_audio_stream();
        }
    }
    ImGui::Text("Stream Status: ");
    ImGui::SameLine();
    if (audio_manager_->is_audio_stream_running())
    {
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusOk), "Running");
    }
    else
    {
        ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError), "Stopped");
    }

    auto audio_stream_info = audio_manager_->get_audio_stream_info();
    ImGui::Text("Sample Rate: %d", audio_stream_info.sample_rate);
    ImGui::Text("Buffer Size: %d", audio_stream_info.buffer_size);
    ImGui::Text("Num Output Channels: %d", audio_stream_info.num_output_channels);

    static bool play_test_tone = false;
    if (ImGui::Checkbox("Play Test Tone", &play_test_tone))
    {
        audio_manager_->play_test_tone(play_test_tone);
    }
}