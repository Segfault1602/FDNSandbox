#include "app.h"

#include "fdn_analyzer.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "imgui_internal.h"

#include <analysis/analysis.h>
#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft_utils.h>

#include "presets.h"
#include "settings.h"
#include "utils.h"
#include "widget.h"

#include <sffdn/sffdn.h>

namespace
{
constexpr size_t kSystemBlockSize = 1024; // System block size for audio processing
constexpr size_t kBlockSize = 64;         // Define a constant block size for audio processing

std::unique_ptr<sfFDN::AudioProcessor> CreateFilterBank(const FDNConfig& config)
{
    if (config.delay_filter_type == DelayFilterType::Proportional)
    {
        std::vector<float> proportional_fb_gains(config.N, 0.f);
        for (size_t i = 0; i < config.N; ++i)
        {
            proportional_fb_gains[i] = std::pow(config.feedback_gain, config.delays[i]);
        }

        auto feedback_gains =
            std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Parallel, proportional_fb_gains);
        return feedback_gains;
    }
    else if (config.delay_filter_type == DelayFilterType::OnePole)
    {
        auto filter_bank = std::make_unique<sfFDN::FilterBank>();
        for (size_t i = 0; i < config.N; ++i)
        {
            auto filter = std::make_unique<sfFDN::OnePoleFilter>();
            float b = 0.f;
            float a = 0.f;
            sfFDN::GetOnePoleAbsorption(config.t60_dc, config.t60_ny, Settings::Instance().SampleRate(),
                                        config.delays[i], b, a);
            filter->SetCoefficients(b, a);
            filter_bank->AddFilter(std::move(filter));
        }
        return filter_bank;
    }

    assert(config.delay_filter_type == DelayFilterType::TwoFilter);
    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (size_t i = 0; i < config.N; ++i)
    {
        std::vector<float> sos = sfFDN::GetTwoFilter(config.t60s, config.delays[i], Settings::Instance().SampleRate());
        const size_t num_stages = sos.size() / 6;
        auto filter = std::make_unique<sfFDN::CascadedBiquads>();
        filter->SetCoefficients(num_stages, sos);
        filter_bank->AddFilter(std::move(filter));
    }

    return filter_bank;
}

std::unique_ptr<sfFDN::FDN> CreateFDN(const FDNConfig& config)
{
    auto fdn = std::make_unique<sfFDN::FDN>(config.N, kBlockSize);

    fdn->SetInputGains(config.input_gains);
    fdn->SetOutputGains(config.output_gains);
    fdn->SetDelays(config.delays);

    if (!config.is_cascaded)
    {
        auto scalar_feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(config.N);
        scalar_feedback_matrix->SetMatrix(config.feedback_matrix);
        fdn->SetFeedbackMatrix(std::move(scalar_feedback_matrix));
    }
    else
    {
        auto ffm = sfFDN::MakeFilterFeedbackMatrix(config.cascaded_feedback_matrix_info);
        fdn->SetFeedbackMatrix(std::move(ffm));
    }

    fdn->SetFilterBank(CreateFilterBank(config));

    constexpr size_t kNBands = 10;
    std::vector<float> freqs(kNBands, 0.f);
    constexpr float kUpperLimit = 16000.0f;
    for (size_t i = 0; i < kNBands; ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
    }
    std::vector<float> tc_sos = sfFDN::DesignGraphicEQ(config.tc_gains, freqs, Settings::Instance().SampleRate());

    std::unique_ptr<sfFDN::CascadedBiquads> tc_filter = std::make_unique<sfFDN::CascadedBiquads>();
    const size_t num_stages = tc_sos.size() / 6;
    tc_filter->SetCoefficients(num_stages, tc_sos);
    fdn->SetTCFilter(std::move(tc_filter));

    return fdn;
}

int MelFormatter(double value, char* buff, int size, void*)
{
    std::vector<float> mels = audio_utils::GetMelFrequencies(512, 0.f, Settings::Instance().SampleRate() / 2.f);
    auto mel_index = static_cast<size_t>(value);
    if (mel_index >= mels.size())
    {
        mel_index = mels.size() - 1; // Clamp to the last index if out of bounds
    }
    auto mel = static_cast<uint32_t>(mels[mel_index]);
    std::span<char> out_string(buff, size);
    std::format_to(out_string.begin(), "{}", mel);
    return std::strlen(buff);
}

} // namespace

FDNToolboxApp::FDNToolboxApp()
    : fdn_analyzer_(Settings::Instance().SampleRate())
{
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

    // Initialize audio manager and start the audio stream
    if (!audio_manager_->start_audio_stream(
            audio_stream_option::kOutput,
            [this](std::span<float> output_buffer, size_t frame_size, size_t num_channels) {
                AudioCallback(output_buffer, frame_size, num_channels);
            },
            kSystemBlockSize))
    {
        throw std::runtime_error("Failed to start audio stream");
    }

    show_tc_filter_designer_ = false;

    fdn_config_ = presets::kDefaultFDNConfig;

    UpdateFDN();
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
    auto start_time = std::chrono::high_resolution_clock::now();

    if (frame_size != kSystemBlockSize)
    {
        std::cerr << "Frame size mismatch: expected " << kSystemBlockSize << ", got " << frame_size << "\n";
        return;
    }

    if (other_fdn_ != nullptr)
    {

        audio_fdn_ = std::move(other_fdn_);
        audio_fdn_->SetDirectGain(0.f); // Direct gain is controlled by the dry/wet mix instead
        other_fdn_ = nullptr;
    }

    if (audio_fdn_ == nullptr)
    {
        // If no FDN is configured, fill the output buffer with silence
        std::ranges::fill(output_buffer, 0.0f);
        return;
    }

    float gain = audio_gain_.load();
    float dry_wet_mix = dry_wet_mix_.load();

    std::array<float, kSystemBlockSize> input_data = {0.0f};
    std::array<float, kSystemBlockSize> output_data = {0.0f};

    sfFDN::AudioBuffer input_audio_buffer(kSystemBlockSize, 1, input_data);
    sfFDN::AudioBuffer output_audio_buffer(kSystemBlockSize, 1, output_data);
    std::ranges::fill(input_data, 0.f);  // Reset input data for each block
    std::ranges::fill(output_data, 0.f); // Reset output data for each block

    if (audio_state_ == AudioState::ImpulseRequested)
    {
        input_data[0] = 1.0f;            // Set the first sample to 1.0 for impulse response
        audio_state_ = AudioState::Idle; // Reset state after processing impulse response
    }

    audio_file_manager_->process_block(input_data.data(), kSystemBlockSize, 1);

    // Process the FDN for this block
    audio_fdn_->Process(input_audio_buffer, output_audio_buffer);

    // Copy to the interleaved output buffer
    const float wet_mix = dry_wet_mix * gain;
    const float dry_mix = (1.f - dry_wet_mix) * gain;

    for (size_t i = 0; i < kSystemBlockSize; ++i)
    {
        int idx_offset = i * num_channels;
        float sample = (wet_mix * output_data[i]) + (dry_mix * input_data[i]);
        for (size_t j = 0; j < num_channels; ++j)
        {
            output_buffer[idx_offset + j] = sample;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    const float allowed_time = (1e9 / Settings::Instance().SampleRate()) * frame_size; // in nanoseconds
    float cpu_usage = static_cast<float>(duration) / allowed_time;
    fdn_cpu_usage_.store(cpu_usage);
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
    static bool show_audio_config_window = false;

    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);

    if (first_time)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

        ImGuiID dock_main_id = dockspace_id;
        ImGuiID dock_id_fdn = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Left, 0.25f, nullptr, &dock_main_id);
        ImGuiID dock_id_ir = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.25f, nullptr, &dock_main_id);
        // ImGuiID dock_id_viz = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_None, 1.f, nullptr,
        // &dock_main_id);

        ImGui::DockBuilderDockWindow("Impulse Response", dock_id_ir);
        ImGui::DockBuilderDockWindow("Audio Player", dock_id_ir);
        ImGui::DockBuilderDockWindow("FDN Configurator", dock_id_fdn);
        ImGui::DockBuilderDockWindow("Visualization", dock_main_id);
        ImGui::DockBuilderDockWindow("Spectrogram", dock_main_id);
        ImGui::DockBuilderDockWindow("Spectrum", dock_main_id);
        ImGui::DockBuilderDockWindow("Autocorrelation", dock_main_id);
        ImGui::DockBuilderDockWindow("Filter Response", dock_main_id);
        ImGui::DockBuilderDockWindow("Energy Decay Curve", dock_main_id);
        ImGui::DockBuilderDockWindow("RT60s", dock_main_id);
        ImGui::DockBuilderDockWindow("Cepstrum", dock_main_id);
        ImGui::DockBuilderDockWindow("Echo Density", dock_main_id);
        ImGui::DockBuilderFinish(dockspace_id);
    }

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            ImGui::MenuItem("Save IR");
            constexpr const char* default_filename = "impulse_response.wav";
            utils::WriteAudioFile(default_filename, fdn_analyzer_.GetImpulseResponse(),
                                  Settings::Instance().SampleRate());
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Options"))
        {
            ImGui::MenuItem("Audio Menu", nullptr, &show_audio_config_window);

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::ShowMetricsWindow();

    if (show_audio_config_window)
    {
        if (ImGui::Begin("Audio", &show_audio_config_window))
        {
            DrawAudioDeviceGUI();
            ImGui::End();
        }
    }

    bool config_changed = false;
    config_changed = DrawFDNConfigurator(fdn_config_);

    if (config_changed)
    {
        UpdateFDN();
    }

    DrawImpulseResponse();

    DrawAudioPlayer();

    DrawVisualization();

    ImGui::End(); // End the main window

    if (first_time)
    {
        ImGui::SetWindowFocus("Spectrogram");
    }
    first_time = false;
}

void FDNToolboxApp::UpdateFDN()
{
    std::cout << "Configuration changed, updating FDN...\n";

    gui_fdn_ = CreateFDN(fdn_config_);
    gui_fdn_->SetDirectGain(0.f);
    fdn_analyzer_.SetFDN(CreateFDN(fdn_config_));

    other_fdn_ = CreateFDN(fdn_config_);
}

bool FDNToolboxApp::DrawFDNConfigurator(FDNConfig& fdn_config)
{
    static uint32_t random_seed = 0;

    static int min_delay = kBlockSize * 2;
    static int max_delay = *std::max_element(fdn_config.delays.begin(), fdn_config.delays.end());

    // Limit FDN size between 4 and 32 channels
    constexpr uint32_t kNMin = 4;
    constexpr uint32_t kNMax = 32;

    if (!ImGui::Begin("FDN Configurator"))
    {
        ImGui::End();
        return false; // If the window is not open, return early
    }

    bool config_changed = false;

    static bool random_seed_checkbox = false;
    ImGui::Checkbox("Random Seed", &random_seed_checkbox);

    if (random_seed_checkbox)
    {
        ImGui::SameLine();
        if (ImGui::InputScalar("Seed", ImGuiDataType_U32, &random_seed, nullptr, nullptr, "%u"))
        {
            config_changed = true; // Mark as changed if the seed is modified
        }
    }
    else
    {
        random_seed = 0; // Reset to zero when unchecked
    }

    DrawInputOutputGainsPlot(gui_fdn_.get());
    DrawDelaysPlot(gui_fdn_.get(), max_delay);
    DrawFeedbackMatrixPlot(gui_fdn_.get());

    bool fdn_size_changed = false;
    fdn_size_changed = ImGui::SliderScalar("N", ImGuiDataType_U32, (&fdn_config.N), &kNMin, &kNMax, nullptr,
                                           ImGuiSliderFlags_AlwaysClamp);

    config_changed |= fdn_size_changed;
    if (fdn_size_changed)
    {
        gui_fdn_->SetN(fdn_config.N);
        if (fdn_config.N != fdn_config.input_gains.size())
        {
            fdn_config.input_gains.resize(fdn_config.N, 0.5f);
        }
        if (fdn_config.N != fdn_config.output_gains.size())
        {
            fdn_config.output_gains.resize(fdn_config.N, 0.5f);
        }
        if (fdn_config.N != fdn_config.delays.size())
        {
            fdn_config.delays.resize(fdn_config.N, 500);
        }
        if (fdn_config.N * fdn_config.N != fdn_config.feedback_matrix.size())
        {
            fdn_config.feedback_matrix.resize(fdn_config.N * fdn_config.N, 0.0f);
        }
    }

    if (ImGui::TreeNode("Edit input gains"))
    {
        config_changed |= DrawGainsWidget(fdn_config.input_gains);
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Edit output gains"))
    {
        config_changed |= DrawGainsWidget(fdn_config.output_gains);
        ImGui::TreePop();
    }

    config_changed |=
        DrawDelayLengthsWidget(fdn_config.N, fdn_config.delays, min_delay, max_delay, random_seed, fdn_size_changed);

    config_changed |= DrawScalarMatrixWidget(fdn_config, random_seed, fdn_size_changed);

    if (ImGui::TreeNode("Delay Filters"))
    {
        config_changed |= DrawDelayFilterWidget(fdn_config);

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Tone Correction Filters"))
    {
        if (ImGui::Button("Edit"))
        {
            std::cout << "Opening filter designer...\n";
            show_tc_filter_designer_ = true;
            config_changed = true;
        }

        if (show_tc_filter_designer_)
        {
            config_changed |= DrawToneCorrectionFilterDesigner(fdn_config);
        }
        ImGui::TreePop();
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

    if (ImPlot::BeginPlot("Impulse Response", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
        auto imp_response = fdn_analyzer_.GetImpulseResponse();

        if (fdn_analyzer_.IsClipping())
        {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 1.0f)); // Red for clipping
        }
        else
        {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));
        }

        ImPlot::SetupAxes("Sample", "Amplitude", ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Once);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, imp_response.size() - 1);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 1.0f);

        ImPlot::PlotLine("Impulse Response", imp_response.data(), imp_response.size());
        ImPlot::EndPlot();

        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();
    }

    ImGui::End(); // End the Impulse Response window
}

void FDNToolboxApp::DrawAudioPlayer()
{
    ImGui::Begin("Audio Player");

    if (ImGui::Button("Impulse"))
    {
        audio_state_ = AudioState::ImpulseRequested;
        std::cout << "Playing impulse response...\n";
    }

    constexpr const char* kAudioFiles[] = {"drumloop.wav", "guitar_2003.wav", "bleepsandbloops.wav"};
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
            if (audio_file_manager_->open_audio_file(kAudioFiles[selected_audio_file]))
            {
                audio_file_manager_->play(true);
            }
            else
            {
                std::cerr << "Failed to open audio file.\n";
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

    static float gain = 1.0f;
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("Gain", &gain, 0.0f, 2.0f, "%.2f"))
    {
        audio_gain_ = gain; // Update the audio gain
    }

    static float mix = 0.5f;
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("Dry/Wet", &mix, 0.0f, 1.0f, "%.2f"))
    {
        dry_wet_mix_ = mix; // Update the dry/wet mix
    }

    static sfFDN::OnePoleFilter cpu_usage_filter;
    cpu_usage_filter.SetPole(0.99f); // Set the pole for the low-pass filter
    float cpu_usage = fdn_cpu_usage_.load();
    cpu_usage = cpu_usage_filter.Tick(cpu_usage); // Apply low-pass filter to CPU usage
    ImGui::Text("CPU Usage: %.2f%%", cpu_usage * 100.0f);

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
    DrawT60s();
    DrawEchoDensity();
}

void FDNToolboxApp::DrawSpectrogram()
{
    if (!ImGui::Begin("Spectrogram"))
    {
        ImGui::End();
        return;
    }

    static float min_dB = -50.0;
    static float max_dB = 10.0;

    ImPlot::PushColormap(ImPlotColormap_Plasma);

    if (ImPlot::BeginPlot("##Spectrogram", ImVec2(-1, -1), ImPlotFlags_NoMouseText))
    {
        fdn_analysis::SpectrogramData spectrogram_data = fdn_analyzer_.GetSpectrogram();

        const double tmin = 0.f;
        const double tmax =
            fdn_analyzer_.GetImpulseResponseSize() / static_cast<double>(Settings::Instance().SampleRate());
        ImPlot::SetupAxisLimits(ImAxis_X1, tmin, tmax, ImGuiCond_Always);
        ImPlot::SetupAxisFormat(ImAxis_Y1, MelFormatter, nullptr);

        const std::array frequencies_ticks = {5.0, 10.0, 50.0};
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr, false);

        double bin_count = static_cast<double>(spectrogram_data.bin_count);

        ImPlot::PlotHeatmap("##Spectrogram", spectrogram_data.data.data(), spectrogram_data.bin_count,
                            spectrogram_data.frame_count, min_dB, max_dB, nullptr, {tmin, 0}, {tmax, bin_count});

        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();

    ImGui::SameLine();

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

    static int peak_radio = 0;
    ImGui::RadioButton("Only Spectrum", &peak_radio, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Only Peaks", &peak_radio, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Both", &peak_radio, 2);

    static float frequency_range_min = 0.f;
    static float frequency_range_max = Settings::Instance().SampleRate() / 2.f;
    ImGui::DragFloatRange2("Frequency Range", &frequency_range_min, &frequency_range_max);

    static bool lock_freq_range = false;
    ImGui::SameLine();
    ImGui::Checkbox("Lock Range", &lock_freq_range);

    static float kRowRatios[] = {0.15f, 0.85f};
    if (ImPlot::BeginSubplots("Spectrum Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, kRowRatios))
    {
        if (ImPlot::BeginPlot("Impulse Response", ImVec2(), ImPlotFlags_NoLegend))
        {
            DrawEarlyRIRPicker(fdn_analyzer_.GetImpulseResponse(), fdn_analyzer_.GetTimeData(), early_rir_duration);
            ImPlot::EndPlot();
        }

        fdn_analysis::SpectrumData spectrum_data = fdn_analyzer_.GetSpectrum(early_rir_duration);
        std::string plot_title = std::format("Spectrum ({} peaks)", spectrum_data.peaks.size());
        if (ImPlot::BeginPlot(plot_title.c_str(), ImVec2(), ImPlotFlags_NoLegend))
        {

            ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)");
            ImPlot::SetupAxesLimits(frequency_range_min, frequency_range_max, -60.0, 0.0,
                                    (lock_freq_range) ? ImPlotCond_Always : ImPlotCond_Once);

            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, spectrum_data.frequency_bins.back());

            if (peak_radio != 1)
            {
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));
                ImPlot::PlotLine("Spectrum", spectrum_data.frequency_bins.data(), spectrum_data.spectrum.data(),
                                 spectrum_data.spectrum.size());
                ImPlot::PopStyleColor();
            }
            if (peak_radio != 0)
            {
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.98f, 0.45f, 0.04f, 1.0f));
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Asterisk, 2.0f);
                ImPlot::PlotScatter("Peaks", spectrum_data.peaks_freqs.data(), spectrum_data.peaks.data(),
                                    spectrum_data.peaks.size(), ImPlotFlags_NoLegend);
                ImPlot::PopStyleColor();
            }

            if (!lock_freq_range)
            {
                auto plot_limit = ImPlot::GetPlotLimits();
                frequency_range_min = static_cast<float>(plot_limit.X.Min);
                frequency_range_max = static_cast<float>(plot_limit.X.Max);
            }

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

    static float kRowRatios[] = {0.15f, 0.85f};
    if (ImPlot::BeginSubplots("Autocorrelation Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, kRowRatios))
    {
        if (ImPlot::BeginPlot("Impulse Response", ImVec2(), ImPlotFlags_NoLegend))
        {
            DrawEarlyRIRPicker(fdn_analyzer_.GetImpulseResponse(), fdn_analyzer_.GetTimeData(), xcorr_duration);
            xcorr_duration = std::clamp(xcorr_duration, 0.1, 0.9);
            ImPlot::EndPlot();
        }

        auto xcorr_data = fdn_analyzer_.GetAutocorrelation(xcorr_duration);

        std::span<const float> xcorr_span =
            (selected_autocorr_type == 0) ? xcorr_data.autocorrelation : xcorr_data.spectral_autocorrelation;

        if (ImPlot::BeginPlot("Autocorrelation", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));

            ImPlot::SetupAxisLimits(ImAxis_X1, -1000.0f, xcorr_span.size(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -1000, xcorr_span.size() + 100);

            ImPlot::PlotLine("Autocorrelation", xcorr_span.data(), xcorr_span.size());
            ImPlot::EndPlot();

            ImPlot::PopStyleVar();
            ImPlot::PopStyleColor();
        }
        ImPlot::EndSubplots();
    }
    ImGui::End();
}

void FDNToolboxApp::DrawFilterResponse()
{
    static std::vector<double> frequencies_ticks(0);

    constexpr uint32_t kNBands = 10; // Number of frequency bands for the GEQ

    if (frequencies_ticks.size() == 0)
    {
        frequencies_ticks.resize(kNBands);
        constexpr float kUpperLimit = 16000.0f;
        for (size_t i = 0; i < kNBands; ++i)
        {
            frequencies_ticks[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
        }
    }

    if (!ImGui::Begin("Filter Response"))
    {
        ImGui::End();
        return;
    }

    auto filter_data = fdn_analyzer_.GetFilterData();
    assert(fdn_config_.N == filter_data.mag_responses.size());

    if (ImGui::BeginTabBar("Filter Responses"))
    {
        if (ImGui::BeginTabItem("Attenuation filters"))
        {
            if (ImPlot::BeginSubplots("Attenuation Filters Mag/Phase", 2, 1, ImVec2(-1, -1),
                                      ImPlotSubplotFlags_LinkAllX))
            {
                if (ImPlot::BeginPlot("Magnitude", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f),
                                      ImPlotFlags_None))
                {
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
                    ImPlot::SetupAxes("Frequency", "Magnitude (dB)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                    for (uint32_t i = 0; i < filter_data.mag_responses.size(); ++i)
                    {
                        auto mag_response = filter_data.mag_responses[i];
                        std::string line_name = "Delay filter " + std::to_string(i + 1);
                        ImPlot::PlotLine(line_name.c_str(), filter_data.frequency_bins.data(), mag_response.data(),
                                         mag_response.size());
                    }
                    ImPlot::PopStyleVar();
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_None))
                {
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
                    ImPlot::SetupAxes("Frequency", "Phase", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);

                    for (uint32_t i = 0; i < filter_data.mag_responses.size(); ++i)
                    {
                        auto phase_response = filter_data.phase_responses[i];
                        std::string line_name = "Delay filter " + std::to_string(i + 1);
                        ImPlot::PlotLine(line_name.c_str(), filter_data.frequency_bins.data(), phase_response.data(),
                                         phase_response.size());
                    }

                    ImPlot::PopStyleVar();
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
                    ImPlot::SetupAxes("Frequency", "Magnitude (dB)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                    ImPlot::PlotLine("TC Filter", filter_data.frequency_bins.data(), filter_data.tc_mag_response.data(),
                                     filter_data.tc_mag_response.size());
                    ImPlot::PopStyleVar();
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_NoLegend))
                {
                    ImPlot::SetupAxes("Frequency", "Phase", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);

                    ImPlot::PlotLine("TC Filter", filter_data.frequency_bins.data(),
                                     filter_data.tc_phase_response.data(), filter_data.tc_phase_response.size());

                    ImPlot::PopStyleVar();
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

    static bool show_octaves_bands = false;
    ImGui::Checkbox("Show Octave Bands", &show_octaves_bands);

    ImGui::DragFloatRange2("T60 decay range", &decay_db_start, &decay_db_end, 1.0, 0, 60);
    decay_db_start = std::clamp(decay_db_start, 0.0f, decay_db_end - 1.0f);
    decay_db_end = std::clamp(decay_db_end, decay_db_start + 1.0f, 60.0f);

    auto edc_data = fdn_analyzer_.GetEnergyDecayCurveData();

    auto t60_data = fdn_analyzer_.GetT60Data(-decay_db_start, -decay_db_end);

    constexpr std::array<const char*, 10> octave_band_names = {"32 Hz", "63 Hz", "125 Hz", "250 Hz", "500 Hz",
                                                               "1 kHz", "2 kHz", "4 kHz",  "8 kHz",  "16 kHz"};

    std::string edc_title = std::format("Energy Decay Curve (T60: {:.2f} s)", t60_data.overall_t60.t60);
    if (ImPlot::BeginPlot(edc_title.c_str(), ImVec2(-1, -1), ImPlotFlags_None))
    {
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);

        // ImPlot::SetupAxes("Sample", "Amplitude", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, fdn_analyzer_.GetTimeData().back());

        // Draw a horizontal line at the -60 dB point
        if (!show_octaves_bands)
        {
            ImPlot::PlotLine("Energy Decay Curve", fdn_analyzer_.GetTimeData().data(),
                             edc_data.energy_decay_curve.data(), edc_data.energy_decay_curve.size());

            std::array<float, 2> t60_x = {t60_data.overall_t60.decay_start_time, t60_data.overall_t60.decay_end_time};
            std::array<float, 2> t60_y = {
                (t60_data.overall_t60.decay_start_time * t60_data.overall_t60.slope) + t60_data.overall_t60.intercept,
                (t60_data.overall_t60.decay_end_time * t60_data.overall_t60.slope) + t60_data.overall_t60.intercept};
            ImPlot::PlotLine("T60 Decay", t60_x.data(), t60_y.data(), 2, ImPlotFlags_NoLegend);
        }
        else
        {
            for (auto [octave_band_name, edc] : std::views::zip(octave_band_names, edc_data.edc_octaves))
            {
                ImPlot::PlotLine(octave_band_name, fdn_analyzer_.GetTimeData().data(), edc.data(), edc.size());
            }
        }

        ImPlot::EndPlot();

        ImPlot::PopStyleVar();
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

    static float kRowRatios[] = {0.15f, 0.85f};
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
            ImPlot::SetupAxisLimits(ImAxis_Y1, -1, 1);
            ImPlot::PlotLine("Cepstrum", cepstrum_data.cepstrum.data(), cepstrum_data.cepstrum.size());

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

    auto echo_density_data = fdn_analyzer_.GetEchoDensityData(window_size_ms, hop_size_ms);

    if (ImPlot::BeginPlot("Impulse Response", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {

        ImPlot::SetupAxes("Sample", "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.5f, ImPlotCond_Once);

        auto ir = fdn_analyzer_.GetImpulseResponse();

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, ir.size() - 1);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 2.0f);

        ImPlot::PlotLine("Impulse Response", ir.data(), ir.size());

        ImPlot::PlotLine("Echo Density", echo_density_data.sparse_indices.data(), echo_density_data.echo_density.data(),
                         echo_density_data.sparse_indices.size());

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
    const float db_end = -20;

    if (ImPlot::BeginPlot("RT60s", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        auto t60_data = fdn_analyzer_.GetT60Data(db_start, db_end);

        ImPlot::SetupAxes("Frequency (Hz)", "RT60 (s)");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 3.0f, ImPlotCond_Once);

        std::vector<double> tick_labels;
        tick_labels.assign(t60_data.octave_band_frequencies.begin(), t60_data.octave_band_frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, tick_labels.data(), tick_labels.size(), nullptr, false);

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f);
        ImPlot::PlotLine("RT60s", t60_data.octave_band_frequencies.data(), t60_data.t60_octaves.data(),
                         t60_data.t60_octaves.size());

        ImPlot::EndPlot();
    }
    ImGui::End();
}

bool FDNToolboxApp::DrawToneCorrectionFilterDesigner(FDNConfig& fdn_config)
{
    if (!ImGui::Begin("Filter Designer", &show_tc_filter_designer_))
    {
        ImGui::End();
        return false;
    }

    bool config_changed = false;

    constexpr size_t kNBands = 10; // Number of bands in the filter designer
    static std::vector<float> gains(kNBands, 0.0f);
    static std::vector<float> frequencies(0, 0.f);

    // Oversampled vectors for plotting
    static std::vector<float> gains_plot;
    static std::vector<float> frequencies_plot;

    bool point_changed = false;

    // Initialize with default values if empty
    if (fdn_config.tc_gains.size() == 0)
    {
        fdn_config.tc_gains = gains;
    }
    else // Use the existing tc_gains from the config
    {
        gains = fdn_config.tc_gains;
    }

    if (frequencies.size() == 0)
    {
        frequencies.resize(kNBands);
        constexpr float kUpperLimit = 16000.0f;
        for (size_t i = 0; i < kNBands; ++i)
        {
            frequencies[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
        }
    }

    if (frequencies_plot.size() == 0) // Only runs on first call
    {
        frequencies_plot =
            utils::LogSpace(std::log10(frequencies[0] + 1e-6f), std::log10(frequencies.back() - 1.f), 512);
        gains_plot = utils::pchip(frequencies, gains, frequencies_plot);

        point_changed = true; // Force initial plot update
    }

    static std::vector<float> H;

    if (ImPlot::BeginPlot("Filter preview", ImVec2(-1, ImGui::GetWindowHeight() * 0.95f), ImPlotFlags_None))
    {
        ImPlot::SetupAxes("Frequency (Hz)", "Gain (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -10.0f, 10.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, Settings::Instance().SampleRate() / 2);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -60.f, 60.f);

        static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

        for (size_t i = 0; i < frequencies.size(); ++i)
        {
            double freq = frequencies[i]; // The frequency should stay constant
            double gain = gains[i];
            point_changed |= ImPlot::DragPoint(i, &freq, &gain, ImVec4(0, 0.9f, 0, 0), 10);
            gains[i] = std::clamp(static_cast<float>(gain), -10.f, 10.0f); // Update the gain value
        }

        if (point_changed)
        {
            gains_plot = utils::pchip(frequencies, gains, frequencies_plot);

            std::vector<float> sos = sfFDN::DesignGraphicEQ(gains, frequencies, Settings::Instance().SampleRate());
            H = utils::AbsFreqz(sos, frequencies_plot, Settings::Instance().SampleRate());

            // To db gain
            for (size_t i = 0; i < H.size(); ++i)
            {
                H[i] = 20.f * std::log10(H[i]);
            }
        }

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f);
        ImPlot::PlotScatter("RT60", frequencies.data(), gains.data(), gains.size());

        if (H.size() > 0)
        {
            ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.20f, 0.20f, 1.0f), 3.0f);
            ImPlot::PlotLine("Filter Response", frequencies_plot.data(), H.data(), frequencies_plot.size());
        }

        ImPlot::EndPlot();
    }

    fdn_config.tc_gains = gains;

    if (ImGui::Button("Apply"))
    {
        std::cout << "Applying filter design..." << std::endl;
        show_tc_filter_designer_ = false;
        fdn_config.tc_gains = gains;
        config_changed = true;
    }

    ImGui::End();
    return config_changed;
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
                std::cout << "Selected Audio Driver: " << supported_audio_drivers[i] << std::endl;
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
                std::cout << "Selected Output Device: " << output_devices[i] << "\n";
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
                std::cerr << "Failed to start audio stream" << std::endl;
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
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Running");
    }
    else
    {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Stopped");
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