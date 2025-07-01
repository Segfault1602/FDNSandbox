#include "app.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>

#include "imgui_internal.h"
#include <imgui.h>
#include <implot.h>
#include <sndfile.h>

#include <audio_metrics.h>
#include <fft_utils.h>

#include "utils.h"

#include <array_math.h>
#include <audio_buffer.h>
#include <delay_utils.h>
#include <fdn.h>
#include <filter_design.h>
#include <math_utils.h>
#include <matrix_gallery.h>
#include <parallel_gains.h>

namespace
{
constexpr size_t kSampleRate = 48000;     // Define a constant sample rate for audio processing
constexpr size_t kBlockSize = 64;         // Define a constant block size for audio processing
constexpr size_t kSystemBlockSize = 1024; // System block size for audio processing

#ifndef NDEBUG
constexpr int kFFTSize = 1024; // Size of FFT for spectrogram
constexpr int kSpectrogramWindowSize = 1024;
constexpr int kOverlap = 300; // Overlap size for spectrogram
#else
constexpr int kFFTSize = 4096; // Size of FFT for spectrogram
constexpr int kSpectrogramWindowSize = 1024;
constexpr int kOverlap = 900; // Overlap size for spectrogram
#endif

constexpr FFTWindowType kFFTWindowType = FFTWindowType::Hann; // Window type for FFT

std::vector<float> audio_loop;
size_t audio_loop_read_index = 0;

void Label(const std::string& label)
{
    ImGui::TextUnformatted(label.c_str());
    ImGui::SameLine();
}

bool DrawGainsTreeNode(std::vector<float>& gains)
{
    bool config_changed = false;
    const size_t N = gains.size();
    if (ImGui::Button("Distribute"))
    {
        config_changed = true;
        for (uint32_t i = 0; i < N; ++i)
        {
            gains[i] = 1.0f / N; // Distribute gains evenly
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Randomize"))
    {
        config_changed = true;
        std::random_device rd;                                    // Obtain a random number from hardware
        std::mt19937 eng(rd());                                   // Seed the generator
        std::uniform_real_distribution<float> distr(-1.0f, 1.0f); // Define the range

        for (uint32_t i = 0; i < N; ++i)
        {
            gains[i] = distr(eng); // Generate random gains
        }
    }

    ImGui::SameLine();
    if (ImGui::BeginPopupContextItem("Gains Popup"))
    {
        static float value = 0.0f; // Default value
        if (ImGui::Selectable("Set to 0.5"))
        {
            value = 0.5f;
            config_changed = true;
        }
        if (ImGui::Selectable("Set to -0.5"))
        {
            value = -0.5f;
            config_changed = true;
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::DragFloat("##Value", &value, 0.01f, -1.0f, 1.0f))
        {
            config_changed = true;
        }

        for (uint32_t i = 0; i < N; ++i)
        {
            gains[i] = value; // Set all gains to the specified value
        }

        ImGui::EndPopup();
    }

    if (ImGui::Button("Set all to..."))
    {
        config_changed = true;
        ImGui::OpenPopup("Gains Popup");
    }

    for (uint32_t i = 0; i < N; ++i)
    {
        std::string label = "Input Gain " + std::to_string(i + 1);
        config_changed |= ImGui::SliderFloat(label.c_str(), &gains[i], -1.f, 1.0f, "%.2f");
    }

    return config_changed;
}

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
            sfFDN::GetOnePoleAbsorption(config.t60_dc, config.t60_ny, kSampleRate, config.delays[i], b, a);
            filter->SetCoefficients(b, a);
            filter_bank->AddFilter(std::move(filter));
        }
        return filter_bank;
    }

    assert(config.delay_filter_type == DelayFilterType::TwoFilter);
    auto filter_bank = std::make_unique<sfFDN::FilterBank>();
    for (size_t i = 0; i < config.N; ++i)
    {
        std::vector<float> sos = sfFDN::GetTwoFilter(config.t60s, config.delays[i], kSampleRate);
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

    auto scalar_feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(config.N);
    scalar_feedback_matrix->SetMatrix(config.feedback_matrix);
    fdn->SetFeedbackMatrix(std::move(scalar_feedback_matrix));

    fdn->SetFilterBank(std::move(CreateFilterBank(config)));

    constexpr size_t kNBands = 10;
    std::vector<float> freqs(kNBands, 0.f);
    constexpr float kUpperLimit = 16000.0f;
    for (size_t i = 0; i < kNBands; ++i)
    {
        freqs[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
    }
    std::vector<float> tc_sos = sfFDN::aceq(config.tc_gains, freqs, kSampleRate);

    std::unique_ptr<sfFDN::CascadedBiquads> tc_filter = std::make_unique<sfFDN::CascadedBiquads>();
    const size_t num_stages = tc_sos.size() / 6;
    tc_filter->SetCoefficients(num_stages, tc_sos);
    fdn->SetTCFilter(std::move(tc_filter));

    return fdn;
}

} // namespace

FDNToolboxApp::FDNToolboxApp()
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
    if (!audio_manager_->start_audio_stream(audio_stream_option::kOutput,
                                            std::bind(&FDNToolboxApp::AudioCallback, this, std::placeholders::_1,
                                                      std::placeholders::_2, std::placeholders::_3),
                                            kSystemBlockSize))
    {
        throw std::runtime_error("Failed to start audio stream");
    }

    impulse_response_.resize(kSampleRate, 0.0f);

    spectrogram_info_.fft_size = kFFTSize;
    spectrogram_info_.overlap = kOverlap;
    spectrogram_info_.samplerate = kSampleRate;
    spectrogram_info_.window_size = kSpectrogramWindowSize;
    spectrogram_info_.window_type = kFFTWindowType;

    show_delay_filter_designer_ = false;
    show_tc_filter_designer_ = false;

    spectrogram_data_.resize(kFFTSize, -50.0f);

    fdn_config_.ir_duration = 1.0f; // Default IR duration
    fdn_config_.sample_rate = kSampleRate;
    fdn_config_.N = 4; // Default number of channels for FDN
    fdn_config_.input_gains.resize(fdn_config_.N, 0.5f);
    fdn_config_.output_gains.resize(fdn_config_.N, 0.5f);
    fdn_config_.delays = sfFDN::GetDelayLengths(fdn_config_.N, 500, 2000, sfFDN::DelayLengthType::Primes, 1);
    fdn_config_.feedback_matrix = sfFDN::GenerateMatrix(fdn_config_.N, sfFDN::ScalarMatrixType::Householder, 0);
    fdn_config_.delay_filter_type = DelayFilterType::Proportional; // Default delay filter type
    fdn_config_.feedback_gain = 0.9999f;
    fdn_config_.t60_dc = 2.f; // Default T60 for one-pole filters
    fdn_config_.t60_ny = 0.6f;
    fdn_config_.t60s.resize(10, 2.f);       // Default T60s for two-filter design
    fdn_config_.tc_gains.resize(10.f, 0.f); // Default tone correction gains

    // Preload the drum loop
    constexpr const char* drum_loop_path = "drumloop.wav";
    audio_loop = utils::ReadAudioFile(drum_loop_path);
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
        std::cerr << "Frame size mismatch: expected " << kSystemBlockSize << ", got " << frame_size << std::endl;
        return;
    }

    if (other_fdn_ != nullptr)
    {
        audio_fdn_ = std::move(other_fdn_);
        other_fdn_ = nullptr;
        audio_fdn_->SetDirectGain(0.f); // Direct gain is controlled by the dry/wet mix instead
    }

    if (audio_fdn_ == nullptr)
    {
        // If no FDN is configured, fill the output buffer with silence
        std::fill(output_buffer.begin(), output_buffer.end(), 0.0f);
        return;
    }

    float gain = audio_gain_.load();
    float dry_wet_mix = dry_wet_mix_.load();

    float input_data[kBlockSize] = {0.0f};
    static std::vector<float> output_data(kBlockSize * num_channels, 0.0f);
    if (output_data.size() != kBlockSize * num_channels)
    {
        std::cout << "Block size or number of channels changed, resizing output_data." << std::endl;
        output_data.resize(kBlockSize * num_channels, 0.0f);
    }

    const size_t block_count = frame_size / kBlockSize;
    if (block_count * kBlockSize != frame_size)
    {
        std::cerr << "Frame size is not a multiple of block size." << std::endl;
    }

    float* output_ptr = output_buffer.data();

    sfFDN::AudioBuffer input_audio_buffer(kBlockSize, 1, input_data);
    sfFDN::AudioBuffer output_audio_buffer(kBlockSize, num_channels, output_data);
    for (size_t i = 0; i < block_count; ++i)
    {
        std::fill(std::begin(input_data), std::end(input_data), 0.f);   // Reset input data for each block
        std::fill(std::begin(output_data), std::end(output_data), 0.f); // Reset output data for each block
        if (audio_state_ == AudioState::ImpulseRequested)
        {
            input_data[0] = 1.0f;            // Set the first sample to 1.0 for impulse response
            audio_state_ = AudioState::Idle; // Reset state after processing impulse response
        }

        audio_file_manager_->process_block(input_data, kBlockSize, 1);

        // Process the FDN for this block
        audio_fdn_->Process(input_audio_buffer, output_audio_buffer);

        // Copy to the interleaved output buffer
        for (size_t i = 0; i < kBlockSize; ++i)
        {
            for (size_t j = 0; j < num_channels; ++j)
            {
                *output_ptr++ =
                    (dry_wet_mix * output_audio_buffer.GetChannelSpan(j)[i] + (1.0f - dry_wet_mix) * input_data[i]) *
                    gain;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    const float allowed_time = (1e9 / kSampleRate) * frame_size; // in nanoseconds
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
    if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
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
        // ImGuiID dock_id_viz = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_None, 1.f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Impulse Response", dock_id_ir);
        ImGui::DockBuilderDockWindow("Audio Player", dock_id_ir);
        ImGui::DockBuilderDockWindow("FDN Configurator", dock_id_fdn);
        ImGui::DockBuilderDockWindow("Visualization", dock_main_id);
        ImGui::DockBuilderDockWindow("Spectrogram", dock_main_id);
        ImGui::DockBuilderDockWindow("Spectrum", dock_main_id);
        ImGui::DockBuilderDockWindow("Autocorrelation", dock_main_id);
        ImGui::DockBuilderDockWindow("Filter Response", dock_main_id);
        ImGui::DockBuilderDockWindow("Energy Decay Curve", dock_main_id);
        ImGui::DockBuilderDockWindow("Cepstrum", dock_main_id);
        ImGui::DockBuilderFinish(dockspace_id);
    }

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            ImGui::MenuItem("Save IR");
            constexpr const char* default_filename = "impulse_response.wav";
            utils::WriteAudioFile(default_filename, impulse_response_, kSampleRate);
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
    ImPlot::ShowStyleEditor();

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
    std::cout << "Configuration changed, updating FDN..." << std::endl;

    auto fdn = CreateFDN(fdn_config_);

    std::vector<float> input(kBlockSize, 0.0f);
    input[0] = 1.0f; // Set the first sample to 1.0 to create an impulse response

    const size_t kImpulseResponseSize = static_cast<size_t>(fdn_config_.ir_duration * kSampleRate);
    impulse_response_.clear();
    impulse_response_.resize(kImpulseResponseSize, 0.0f);

    const size_t kBlockCount = kImpulseResponseSize / kBlockSize;
    for (size_t i = 0; i < kBlockCount; ++i)
    {
        sfFDN::AudioBuffer input_buffer(kBlockSize, 1, input);
        sfFDN::AudioBuffer output_buffer(kBlockSize, 1, impulse_response_.data() + i * kBlockSize);
        fdn->Process(input_buffer, output_buffer);

        input[0] = 0.0f; // Reset the first sample to 0.0 for subsequent blocks
    }
    impulse_response_changed_.set();

    other_fdn_ = CreateFDN(fdn_config_);
}

bool FDNToolboxApp::DrawFDNConfigurator(FDNConfig& fdn_config)
{
    static bool first_time = true;
    static uint32_t sample_rate = 48000;
    static uint32_t random_seed = 0;

    if (!ImGui::Begin("FDN Configurator"))
    {
        ImGui::End();
        return false; // If the window is not open, return early
    }

    bool config_changed = first_time;
    first_time = false;

    bool shouldUpdateFeedbackMatrix = false;
    bool shouldUpdateDelays = false;

    static bool random_seed_checkbox = false;
    ImGui::Checkbox("Random Seed", &random_seed_checkbox);

    if (random_seed_checkbox)
    {
        ImGui::SameLine();
        if (ImGui::InputScalar("Seed", ImGuiDataType_U32, &random_seed, nullptr, nullptr, "%u"))
        {
            config_changed = true;             // Mark as changed if the seed is modified
            shouldUpdateFeedbackMatrix = true; // Update feedback matrix when seed changes
        }
    }
    else
    {
        random_seed = 0; // Reset to zero when unchecked
    }

    constexpr float kMinDuration = 1.f;
    constexpr float kMaxDuration = 10.f;
    bool ir_length_changed = ImGui::SliderScalar("IR Duration", ImGuiDataType_Float, &fdn_config.ir_duration,
                                                 &kMinDuration, &kMaxDuration, "%.2f");

    // Label("N:");
    constexpr uint32_t kNMin = 4;
    constexpr uint32_t kNMax = 32;
    config_changed |= ImGui::SliderScalar("N", ImGuiDataType_U32, (&fdn_config.N), &kNMin, &kNMax, nullptr,
                                          ImGuiSliderFlags_AlwaysClamp);
    if (config_changed)
    {
        shouldUpdateFeedbackMatrix = true; // Update feedback matrix when N changes
        shouldUpdateDelays = true;         // Update delays when N changes
    }

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

    // Label("Input Gains:");
    ImGui::PlotHistogram("Input Gains", fdn_config.input_gains.data(), fdn_config.input_gains.size(), 0, NULL, -1.0f,
                         1.0f, ImVec2(0, 40.0f));

    // Label("Output Gains:");
    ImGui::PlotHistogram("Output Gains", fdn_config.output_gains.data(), fdn_config.output_gains.size(), 0, NULL, -1.0f,
                         1.0f, ImVec2(0, 40.0f));

    // Delays
    static int min_delay = 400;
    static int max_delay = 2000;
    if (ImPlot::BeginPlot("##Delays", ImVec2(300, 100), ImPlotFlags_CanvasOnly))
    {
        constexpr ImPlotAxisFlags axes_flags =
            ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoGridLines;
        ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);

        ImPlot::SetupAxesLimits(-1, fdn_config.delays.size(), 0, max_delay, ImPlotCond_Always);

        ImPlot::PlotBars("Delays", fdn_config.delays.data(), fdn_config.delays.size(), 0.90, 0, ImPlotBarsFlags_None);
        ImPlot::EndPlot();
    }

    // Feedback Matrix
    static ImPlotColormap feedback_matrix_colormap = ImPlotColormap_Plasma;
    ImPlot::PushColormap(feedback_matrix_colormap);

    if (ImPlot::BeginPlot("Feedback Matrix", ImVec2(300, 300), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
        constexpr ImPlotAxisFlags axes_flags =
            ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels;
        ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
        const char* label_fmt = fdn_config.N < 10 ? "%.2f" : nullptr; // Adjust label format based on N size
        ImPlot::PlotHeatmap("heat", fdn_config.feedback_matrix.data(), fdn_config.N, fdn_config.N, -1, 1, label_fmt,
                            ImPlotPoint(0, 0), ImPlotPoint(1, 1), 0);

        ImPlot::EndPlot();
    }

    ImPlot::PopColormap();

    if (ImGui::TreeNode("Edit input gains"))
    {
        config_changed |= DrawGainsTreeNode(fdn_config.input_gains);
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Edit output gains"))
    {
        config_changed |= DrawGainsTreeNode(fdn_config.output_gains);
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Edit delays"))
    {
        constexpr int kMinDelay = 400; // Minimum delay in samples
        if (ImGui::DragIntRange2("Delay Range", &min_delay, &max_delay, 1, kMinDelay, 0, "%d samples", "%d samples",
                                 ImGuiSliderFlags_AlwaysClamp))
        {
            config_changed = true;
            shouldUpdateDelays = true;
        }
        // Need to manually clamp value incase of user input
        if (max_delay < kMinDelay)
        {
            max_delay = kMinDelay;
        }

        constexpr const char* kDelayLengthTypeNames[] = {"Random",  "Gaussian",    "Primes",
                                                         "Uniform", "Prime Power", "Steam Audio"};
        static int selected_delay_length_type = 0;
        const char* combo_preview_value = kDelayLengthTypeNames[selected_delay_length_type];
        if (ImGui::BeginCombo("Delay Length Type", combo_preview_value))
        {
            for (int i = 0; i < IM_ARRAYSIZE(kDelayLengthTypeNames); i++)
            {
                bool is_selected = (selected_delay_length_type == i);
                if (ImGui::Selectable(kDelayLengthTypeNames[i], is_selected))
                {
                    selected_delay_length_type = i;
                    config_changed = true;
                    shouldUpdateDelays = true; // Update delays when type changes
                }
            }
            ImGui::EndCombo();
        }

        if (shouldUpdateDelays)
        {
            shouldUpdateDelays = false;
            switch (selected_delay_length_type)
            {
            case 0: // Random
                fdn_config.delays = sfFDN::GetDelayLengths(fdn_config.N, min_delay, max_delay,
                                                           sfFDN::DelayLengthType::Random, random_seed);
                break;
            case 1: // Gaussian
                fdn_config.delays = sfFDN::GetDelayLengths(fdn_config.N, min_delay, max_delay,
                                                           sfFDN::DelayLengthType::Gaussian, random_seed);
                break;
            case 2: // Primes
                fdn_config.delays = sfFDN::GetDelayLengths(fdn_config.N, min_delay, max_delay,
                                                           sfFDN::DelayLengthType::Primes, random_seed);
                break;
            case 3: // Uniform
                fdn_config.delays = sfFDN::GetDelayLengths(fdn_config.N, min_delay, max_delay,
                                                           sfFDN::DelayLengthType::Uniform, random_seed);
                break;
            case 4: // Prime Power
                fdn_config.delays = sfFDN::GetDelayLengths(fdn_config.N, min_delay, max_delay,
                                                           sfFDN::DelayLengthType::PrimePower, random_seed);
                break;
            case 5: // Steam Audio
                fdn_config.delays = sfFDN::GetDelayLengths(fdn_config.N, min_delay, max_delay,
                                                           sfFDN::DelayLengthType::SteamAudio, random_seed);
                break;
            default:
                std::cerr << "Unknown delay length type selected: " << selected_delay_length_type << std::endl;
                break;
            }
        }

        static bool clamp_to_prime = false;
        config_changed |= ImGui::Checkbox("Clamp to Prime", &clamp_to_prime);

        for (uint32_t i = 0; i < fdn_config.N; ++i)
        {
            std::string label = "Delay " + std::to_string(i + 1);
            int delay = static_cast<int>(fdn_config.delays[i]);
            config_changed |= ImGui::SliderInt(label.c_str(), &delay, kBlockSize * 2, max_delay, nullptr);
            fdn_config.delays[i] = static_cast<size_t>(delay);
            if (fdn_config.delays[i] < kBlockSize * 2)
            {
                fdn_config.delays[i] = kBlockSize * 2; // Ensure minimum delay is at least kBlockSize * 2
            }

            if (clamp_to_prime)
            {
                fdn_config.delays[i] = utils::GetClosestPrime(fdn_config.delays[i]);
            }
        }
        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Edit matrix"))
    {
        static int selected_matrix_type = 0;
        const std::string combo_preview_value =
            utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type));
        if (ImGui::BeginCombo("Matrix Type", combo_preview_value.c_str()))
        {
            for (int i = 0; i < static_cast<int>(sfFDN::ScalarMatrixType::Count); i++)
            {
                bool is_selected = (selected_matrix_type == i);

                ImGuiSelectableFlags flags = ImGuiSelectableFlags_None;
                if (static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard)

                    if (!utils::IsPowerOfTwo(fdn_config.N) &&
                        static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard)
                    {
                        flags |= ImGuiSelectableFlags_Disabled;
                    }

                if (ImGui::Selectable(utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(i)).c_str(),
                                      is_selected, flags))
                {
                    selected_matrix_type = i;
                    config_changed = true;
                    shouldUpdateFeedbackMatrix = true;
                }

                if (!utils::IsPowerOfTwo(fdn_config.N) &&
                    static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard)
                {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), " only supported for N that is a power of 2.");
                }
            }
            ImGui::EndCombo();
        }

        if (shouldUpdateFeedbackMatrix)
        {
            shouldUpdateFeedbackMatrix = false;
            sfFDN::ScalarMatrixType matrix_type = static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type);
            if (matrix_type == sfFDN::ScalarMatrixType::Hadamard)
            {
                // Note: Hadamard matrix generation may not be supported for all N, ensure N is a power of 2
                if (utils::IsPowerOfTwo(fdn_config.N)) // Check if N is a power of 2
                {
                    fdn_config.feedback_matrix = sfFDN::GenerateMatrix(fdn_config.N, sfFDN::ScalarMatrixType::Hadamard);
                }
                else
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f),
                                       "Hadamard matrix is only supported for N that is a power of 2.");
                    fdn_config.feedback_matrix = sfFDN::GenerateMatrix(fdn_config.N, sfFDN::ScalarMatrixType::Identity);
                }
            }
            else if (matrix_type == sfFDN::ScalarMatrixType::NestedAllpass)
            {
                fdn_config.feedback_matrix = sfFDN::NestedAllpassMatrix(
                    fdn_config.N, random_seed, fdn_config.input_gains, fdn_config.output_gains);
            }
            else
            {
                fdn_config.feedback_matrix = sfFDN::GenerateMatrix(fdn_config.N, matrix_type, random_seed);
            }
        }

        if (ImGui::Button("Manual Edit"))
        {
            ImGui::OpenPopup("Matrix Edit Popup");
        }
        if (ImGui::BeginPopupModal("Matrix Edit Popup", NULL,
                                   ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize))
        {
            if (ImGui::Button("Clear"))
            {
                fdn_config.feedback_matrix.assign(fdn_config.N * fdn_config.N, 0.0f);
                config_changed = true;
            }

            ImGui::SameLine();
            if (ImGui::Button("Read from clipboard"))
            {
                const char* clipboard_text = ImGui::GetClipboardText();
                std::string clipboard_string(clipboard_text);

                // split by newline
                std::vector<std::string> lines;
                std::istringstream stream(clipboard_string);
                std::string line;
                while (std::getline(stream, line))
                {
                    lines.push_back(line);
                }

                size_t line_count = std::min(lines.size(), static_cast<size_t>(fdn_config.N));

                for (size_t i = 0; i < line_count; ++i)
                {
                    // Each line should have N values separated by spaces or tabs
                    std::istringstream line_stream(lines[i]);
                    std::vector<float> values;
                    float value;
                    while (line_stream >> value)
                    {
                        values.push_back(value);
                    }

                    // Only keep the first N values for each row
                    for (size_t j = 0; j < fdn_config.N && j < values.size(); ++j)
                    {
                        fdn_config.feedback_matrix[i * fdn_config.N + j] = values[j];
                    }
                }
            }

            ImGui::PushItemWidth(50);
            for (size_t i = 0; i < fdn_config.N; ++i)
            {
                for (size_t j = 0; j < fdn_config.N; ++j)
                {
                    if (j != 0)
                    {
                        ImGui::SameLine(); // Align inputs in a grid
                    }
                    ImGui::PushID(i * fdn_config.N + j);
                    config_changed |= ImGui::InputScalar("##MatrixValue", ImGuiDataType_Float,
                                                         &fdn_config.feedback_matrix[i * fdn_config.N + j], nullptr,
                                                         nullptr, "%.3f", 0);
                    ImGui::PopID();
                }
            }

            ImGui::PopItemWidth();

            if (ImGui::Button("Set"))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Delay Filters"))
    {
        constexpr const char* kFilterTypeNames[] = {"Proportional", "One Pole", "Octave Band Filter"};

        static int selected_filter_type = 0;
        const char* combo_preview_value = kFilterTypeNames[selected_filter_type];
        if (ImGui::BeginCombo("Filter Type", combo_preview_value))
        {
            for (int i = 0; i < IM_ARRAYSIZE(kFilterTypeNames); i++)
            {
                bool is_selected = (selected_filter_type == i);
                if (ImGui::Selectable(kFilterTypeNames[i], is_selected))
                {
                    selected_filter_type = i;
                    config_changed = true;
                }
            }
            ImGui::EndCombo();
        }

        // Proportinal feedback Gain
        if (selected_filter_type == 0) // Proportional
        {
            const float kFbGainStep = 0.00001f;
            const float kFbGainStepFast = 0.0001f;
            config_changed |= ImGui::InputScalar("Feedback Gain", ImGuiDataType_Float, &fdn_config.feedback_gain,
                                                 &kFbGainStep, &kFbGainStepFast, "%.5f", 0);
            fdn_config.feedback_gain =
                std::clamp(fdn_config.feedback_gain, -1.0f, 1.0f); // Ensure feedback gain is within [0, 1]
            fdn_config.delay_filter_type = DelayFilterType::Proportional;
        }
        else if (selected_filter_type == 1) // One Pole
        {
            constexpr float kOffsetFromStart = 125.f;
            ImGui::Text("RT60 DC: ");
            ImGui::SameLine(kOffsetFromStart);
            ImGui::SetNextItemWidth(200);
            if (ImGui::InputFloat("RT60 DC", &fdn_config.t60_dc, 0.01f, 0.1f, "%.2f"))
            {
                config_changed = true;
            }

            fdn_config.t60_dc = std::clamp(fdn_config.t60_dc, 0.01f, 10.0f);

            ImGui::Text("RT60 Nyquist: ");
            ImGui::SameLine(kOffsetFromStart);
            ImGui::SetNextItemWidth(200);
            if (ImGui::InputFloat("RT60 Nyquist", &fdn_config.t60_ny, 0.01f, 0.1f, "%.2f"))
            {
                config_changed = true;
            }

            fdn_config.t60_ny = std::clamp(fdn_config.t60_ny, 0.01f, 10.0f);
            fdn_config.delay_filter_type = DelayFilterType::OnePole;
        }
        else if (selected_filter_type == 2) // TwoFilter
        {
            if (ImGui::Button("Edit"))
            {
                std::cout << "Opening filter designer..." << std::endl;
                show_delay_filter_designer_ = true;
                config_changed = true;
            }
            fdn_config.delay_filter_type = DelayFilterType::TwoFilter;
        }

        if (show_delay_filter_designer_)
        {
            config_changed |= DrawFilterDesigner(fdn_config);
        }

        ImGui::TreePop();
    }

    if (ImGui::TreeNode("Tone Correction Filters"))
    {
        if (ImGui::Button("Edit"))
        {
            std::cout << "Opening filter designer..." << std::endl;
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
    return config_changed || ir_length_changed;
}

void FDNToolboxApp::DrawImpulseResponse()
{
    static ImPlotAxisFlags yaxis_flag;
    static bool is_clipping = false;

    if (!ImGui::Begin("Impulse Response"))
    {
        ImGui::End();
        return;
    }

    if (impulse_response_changed_.test(WindowType::ImpulseResponse))
    {
        std::cout << "Impulse response changed, updating plot..." << std::endl;
        impulse_response_changed_.reset(WindowType::ImpulseResponse);
        is_clipping = std::any_of(impulse_response_.begin(), impulse_response_.end(),
                                  [](float sample) { return sample < -1.0f || sample > 1.0f; });
    }

    if (ImPlot::BeginPlot("Impulse Response", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);

        if (is_clipping)
        {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 1.0f)); // Red for clipping
        }
        else
        {
            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));
        }

        ImPlot::SetupAxes("Sample", "Amplitude");
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Once);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, impulse_response_.size() - 1);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 1.0f);

        ImPlot::PlotLine("Impulse Response", impulse_response_.data(), impulse_response_.size());
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
        std::cout << "Playing impulse response..." << std::endl;
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
                std::cerr << "Failed to open audio file." << std::endl;
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
}

int MetricFormatter(double value, char* buff, int size, void* data)
{
    std::vector<float> mels = GetMelFrequencies(512, 0.f, kSampleRate / 2.f);
    size_t mel_index = static_cast<size_t>(value);
    if (mel_index >= mels.size())
    {
        mel_index = mels.size() - 1; // Clamp to the last index if out of bounds
    }
    uint32_t mel = static_cast<uint32_t>(mels[mel_index]);

    return snprintf(buff, size, "%u", mel);
}

void FDNToolboxApp::DrawSpectrogram()
{
    if (!ImGui::Begin("Spectrogram"))
    {
        ImGui::End();
        return;
    }

    static float min_dB = -50.f;
    static float max_dB = 10.f;
    constexpr size_t kNMels = 512;

    const double tmin = 0.f;
    const double tmax = impulse_response_.size() / static_cast<double>(kSampleRate);
    const float h = ImGui::GetWindowSize()[1];
    const float w = ImGui::GetWindowSize()[0];
    ImPlot::PushColormap(ImPlotColormap_Plasma);

    if (ImPlot::BeginPlot("##Spectrogram", ImVec2(-1, -1), ImPlotFlags_NoMouseText))
    {

        if (impulse_response_changed_.test(WindowType::Spectrogram))
        {
            impulse_response_changed_.reset(WindowType::Spectrogram);
            std::cout << "Impulse response changed, updating spectrogram..." << std::endl;
            spectrogram_data_ =
                MelSpectrogram(impulse_response_.data(), impulse_response_.size(), spectrogram_info_, kNMels);
        }

        ImPlot::SetupAxisLimits(ImAxis_X1, tmin, tmax, ImGuiCond_Always);
        // ImPlot::SetupAxisLimits(ImAxis_Y1, 0, kSampleRate / 2000, ImGuiCond_Always);
        // ImPlot::SetupAxisFormat(ImAxis_Y1, "%g kHz");
        ImPlot::SetupAxisFormat(ImAxis_Y1, MetricFormatter, nullptr);

        ImPlot::PlotHeatmap("##Heat", spectrogram_data_.data(), spectrogram_info_.num_freqs,
                            spectrogram_info_.num_frames, min_dB, max_dB, nullptr, {tmin, 0}, {tmax, kNMels});

        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();

    ImGui::SameLine();
    // ImPlot::ColormapScale("##Scale", g_min_dB, g_max_dB, {-1, 0.9f * h}, "%g dB");
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

    constexpr size_t kSpectrumNFFT = 48000;
    constexpr size_t kNumFrequencyBins = kSpectrumNFFT / 2 + 1; // NFFT/2 + 1 for real FFT
    static std::vector<float> spectrum_data(kSpectrumNFFT, 0.0f);
    static std::vector<float> freqs(kNumFrequencyBins, 0.0f);
    static std::vector<float> pks_db(kNumFrequencyBins, 0.0f);
    static std::vector<float> pks_freqs(kNumFrequencyBins, 0.0f);
    static int pks_count = 0;

    if (impulse_response_changed_.test(WindowType::Spectrum))
    {
        impulse_response_changed_.reset(WindowType::Spectrum);
        // copy the first 500 ms of the impulse response to the spectrum data
        size_t num_samples = static_cast<size_t>(0.5 * kSampleRate);
        if (num_samples > impulse_response_.size())
        {
            num_samples = impulse_response_.size();
        }
        std::fill(spectrum_data.begin(), spectrum_data.end(), 0.0f);
        std::copy(impulse_response_.begin(), impulse_response_.begin() + num_samples, spectrum_data.begin());

        spectrum_data = AbsFFT(std::span(impulse_response_.data(), num_samples), kSpectrumNFFT, true, true);

        // Generate frequency bins
        for (size_t i = 0; i < kNumFrequencyBins; ++i)
        {
            freqs[i] = static_cast<float>(i) * kSampleRate / kSpectrumNFFT;
        }

        // Find peaks in the spectrum
        pks_count = 0;
        for (size_t i = 1; i < kNumFrequencyBins - 1; ++i)
        {
            if (spectrum_data[i] > spectrum_data[i - 1] && spectrum_data[i] > spectrum_data[i + 1])
            {
                pks_db[pks_count] = spectrum_data[i];
                pks_freqs[pks_count] = freqs[i];
                pks_count++;
            }
        }
    }

    static bool show_peaks = false;
    ImGui::Checkbox("Show Peaks", &show_peaks);

    if (show_peaks)
    {
        ImGui::Text("Peaks found: %d", pks_count);
    }

    if (ImPlot::BeginPlot("Spectrum", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, freqs.back());
        if (show_peaks)
        {
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 1.0f);
            ImPlot::PlotScatter("Peaks", pks_freqs.data(), pks_db.data(), pks_count, ImPlotFlags_NoLegend);
        }
        else
        {
            ImPlot::PlotLine("Spectrum", freqs.data(), spectrum_data.data(), kNumFrequencyBins);
        }

        ImPlot::EndPlot();
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

    constexpr size_t time_corr_size = 0.25 * kSampleRate; // 250 ms autocorrelation
    constexpr size_t spectral_corr_size = 0.5 * kSampleRate;

    static std::vector<float> autocorr(time_corr_size, 0.0f);
    static std::vector<float> spectral_autocorr(spectral_corr_size, 0.0f);

    if (impulse_response_changed_.test(WindowType::Autocorrelation))
    {
        impulse_response_changed_.reset(WindowType::Autocorrelation);
        auto early_rir = std::span<const float>(impulse_response_.data(), time_corr_size);

        std::vector<float> spectrum = AbsFFT(early_rir, kSampleRate, true, false);

        spectral_autocorr = audio_utils::metrics::Autocorrelation(spectrum);
        autocorr = audio_utils::metrics::Autocorrelation(early_rir);
    }

    constexpr const char* autocorr_type[] = {"Time", "Spectral"};
    static int selected_autocorr_type = 0;
    const char* combo_preview_value = autocorr_type[selected_autocorr_type];

    if (ImGui::RadioButton("Time", selected_autocorr_type == 0))
    {
        selected_autocorr_type = 0;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Spectral", selected_autocorr_type == 1))
    {
        selected_autocorr_type = 1;
    }
    float* data_ptr = (selected_autocorr_type == 0) ? autocorr.data() : spectral_autocorr.data();
    size_t data_size = (selected_autocorr_type == 0) ? autocorr.size() : spectral_autocorr.size();

    if (ImPlot::BeginPlot("Autocorrelation", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));

        ImPlot::SetupAxisLimits(ImAxis_X1, -1000.0f, data_size, ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -1000, data_size + 100);

        ImPlot::PlotLine("Autocorrelation", data_ptr, data_size);
        ImPlot::EndPlot();

        ImPlot::PopStyleVar();
        ImPlot::PopStyleColor();
    }
    ImGui::End();
}

void FDNToolboxApp::DrawFilterResponse()
{
    // Delay filter
    static std::vector<std::vector<float>> filter_mag_responses;
    static std::vector<std::vector<float>> filter_phase_responses;

    // Tone Correction Filter
    static std::vector<float> tc_filter_mag_response;
    static std::vector<float> tc_filter_phase_response;

    constexpr size_t kNFFT = 8192; // Number of FFT points for filter response
    static std::vector<float> frequencies(0);
    if (frequencies.size() != kNFFT / 2 + 1)
    {
        frequencies.resize(kNFFT / 2 + 1);
        for (size_t i = 0; i < frequencies.size(); ++i)
        {
            frequencies[i] = static_cast<float>(i) * kSampleRate / kNFFT;
        }
    }

    if (!ImGui::Begin("Filter Response"))
    {
        ImGui::End();
        return;
    }

    bool reset_y_axis = false;

    if (impulse_response_changed_.test(WindowType::FilterResponse))
    {
        reset_y_axis = true;
        impulse_response_changed_.reset(WindowType::FilterResponse);

        auto filter_bank = CreateFilterBank(fdn_config_);
        filter_mag_responses.resize(fdn_config_.N);
        filter_phase_responses.resize(fdn_config_.N);

        std::vector<float> impulse(kNFFT * fdn_config_.N, 0.0f);
        // Create an impulse for each filter
        for (uint32_t i = 0; i < fdn_config_.N; ++i)
        {
            impulse[i * kNFFT] = 1.0f;
        }

        sfFDN::AudioBuffer inout_buffer(kNFFT, fdn_config_.N, impulse);
        filter_bank->Process(inout_buffer, inout_buffer);

        for (uint32_t i = 0; i < fdn_config_.N; ++i)
        {
            auto channel_span = inout_buffer.GetChannelSpan(i);
            filter_mag_responses[i].resize(channel_span.size());
            std::vector<std::complex<float>> mag_response = FFT(channel_span, kNFFT);

            // Compute the phase response
            filter_phase_responses[i].resize(kNFFT / 2 + 1);
            for (size_t j = 0; j < filter_phase_responses[i].size(); ++j)
            {
                filter_phase_responses[i][j] = 180.f / std::numbers::pi * std::arg(mag_response[j]);
            }

            filter_mag_responses[i].resize(kNFFT / 2 + 1);

            for (size_t j = 0; j < filter_mag_responses[i].size(); ++j)
            {
                // Convert to dB scale
                filter_mag_responses[i][j] = 20.0f * std::log10(std::abs(mag_response[j]));
                if (std::isinf(filter_mag_responses[i][j]) || std::isnan(filter_mag_responses[i][j]))
                {
                    filter_mag_responses[i][j] = -50.0f; // Set to a low value if the log is invalid
                }
            }
        }

        // Tone Correction Filter Response

        std::vector<float> frequencies_band(10, 0.f);
        constexpr float kUpperLimit = 16000.0f;
        for (size_t i = 0; i < 10; ++i)
        {
            frequencies_band[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(10 - 1 - i));
        }

        std::vector<float> sos = sfFDN::aceq(fdn_config_.tc_gains, frequencies_band, kSampleRate);
        tc_filter_mag_response = utils::AbsFreqz(sos, frequencies, kSampleRate);

        for (size_t i = 0; i < tc_filter_mag_response.size(); ++i)
        {
            // Convert to dB scale
            tc_filter_mag_response[i] = 20.0f * std::log10(tc_filter_mag_response[i]);
            if (std::isinf(tc_filter_mag_response[i]) || std::isnan(tc_filter_mag_response[i]))
            {
                tc_filter_mag_response[i] = -50.0f; // Set to a low value if the log is invalid
            }
        }
    }

    if (ImGui::BeginTabBar("Filter Responses"))
    {
        for (uint32_t i = 0; i < fdn_config_.N; ++i)
        {
            std::string tab_name = "Filter " + std::to_string(i + 1);
            if (ImGui::BeginTabItem(tab_name.c_str()))
            {
                if (ImPlot::BeginPlot(tab_name.c_str(), ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f),
                                      ImPlotFlags_NoLegend | ImPlotAxisFlags_AutoFit))
                {
                    ImPlot::SetupAxes("Frequency", "Magnitude (dB)");

                    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));

                    // if (reset_y_axis)
                    {
                        float min_y = *std::min_element(filter_mag_responses[i].begin(), filter_mag_responses[i].end());
                        min_y = std::min(min_y - 2.f, -6.f);
                        ImPlot::SetupAxisLimits(ImAxis_Y1, min_y, 1.0f, ImPlotCond_Always);
                    }

                    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, frequencies.back());
                    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.250f);
                    ImPlot::PlotShaded("##Mag1", frequencies.data(), filter_mag_responses[i].data(),
                                       filter_mag_responses[i].size(), -INFINITY);
                    ImPlot::PlotLine("Filter Response", frequencies.data(), filter_mag_responses[i].data(),
                                     filter_mag_responses[i].size());

                    ImPlot::PopStyleVar();
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }

                std::string phase_plot_name = "Phase " + std::to_string(i + 1);
                if (ImPlot::BeginPlot(phase_plot_name.c_str(), ImVec2(-1, -1),
                                      ImPlotFlags_NoLegend | ImPlotAxisFlags_AutoFit))
                {
                    ImPlot::SetupAxes("Frequency", "Phase");

                    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));

                    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -180.f, 180.f);
                    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, frequencies.back());

                    ImPlot::PlotLine("Phase Response", frequencies.data(), filter_phase_responses[i].data(),
                                     filter_phase_responses[i].size());

                    ImPlot::PopStyleVar();
                    ImPlot::PopStyleColor();
                    ImPlot::EndPlot();
                }
                ImGui::EndTabItem();
            }
        }

        if (ImGui::BeginTabItem("Tone Correction Filter"))
        {
            if (ImPlot::BeginPlot("Tone Correction Filter", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f),
                                  ImPlotFlags_NoLegend))
            {
                ImPlot::SetupAxes("Frequency", "Magnitude (dB)");

                ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
                ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));

                std::vector<double> frequencies_ticks(10, 0.f);
                constexpr double kUpperLimit = 16000.0f;
                for (size_t i = 0; i < 10; ++i)
                {
                    frequencies_ticks[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(10 - 1 - i));
                }
                ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr, false);

                ImPlot::SetupAxisZoomConstraints(ImAxis_Y1, 10.f, 100.f);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, kSampleRate / 2.f);
                ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.250f);
                ImPlot::PlotShaded("##tc_filter_mag", frequencies.data(), tc_filter_mag_response.data(),
                                   tc_filter_mag_response.size(), -INFINITY);
                ImPlot::PlotLine("Filter Response", frequencies.data(), tc_filter_mag_response.data(),
                                 tc_filter_mag_response.size());

                ImPlot::PopStyleVar();
                ImPlot::PopStyleColor();
                ImPlot::EndPlot();
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void FDNToolboxApp::DrawEnergyDecayCurve()
{
    static ImPlotAxisFlags yaxis_flag;
    static std::vector<float> energy_decay_curve(kSampleRate, 0.0f);
    static std::vector<float> x_axis_data(kSampleRate, 0.0f);
    static std::array<std::vector<float>, 10> edc_octaves;

    if (!ImGui::Begin("Energy Decay Curve"))
    {
        ImGui::End();
        return;
    }

    static float kEDC60db = 0.0f; // Maximum value for the energy decay curve

    bool autofit = false;
    if (impulse_response_changed_.test(WindowType::EnergyDecayCurve))
    {
        autofit = true;
        impulse_response_changed_.reset(WindowType::EnergyDecayCurve);
        energy_decay_curve = utils::EnergyDecayCurve(impulse_response_, true);

        kEDC60db = *std::max_element(energy_decay_curve.begin(), energy_decay_curve.end()) - 60.f;

        x_axis_data.resize(energy_decay_curve.size());
        for (size_t i = 0; i < x_axis_data.size(); ++i)
        {
            x_axis_data[i] = static_cast<float>(i) / static_cast<float>(kSampleRate); // Convert sample index to seconds
        }

        auto octave_bands_sos = utils::GetOctaveBandsSOS();
        assert(octave_bands_sos.size() == edc_octaves.size());

        for (size_t i = 0; i < edc_octaves.size(); ++i)
        {
            sfFDN::CascadedBiquads octave_filter;
            octave_filter.SetCoefficients(1, octave_bands_sos[i]);
            edc_octaves[i].resize(impulse_response_.size(), 0.0f);

            sfFDN::AudioBuffer input_buffer(impulse_response_.size(), 1, impulse_response_);
            sfFDN::AudioBuffer output_buffer(edc_octaves[i].size(), 1, edc_octaves[i]);
            octave_filter.Process(input_buffer, output_buffer);

            edc_octaves[i] = utils::EnergyDecayCurve(output_buffer.GetChannelSpan(0), true);
        }
    }

    static bool show_octaves_bands = false;
    ImGui::Checkbox("Show Octave Bands", &show_octaves_bands);

    constexpr const char* octave_band_names[] = {"32 Hz", "63 Hz", "125 Hz", "250 Hz", "500 Hz",
                                                 "1 kHz", "2 kHz", "4 kHz",  "8 kHz",  "16 kHz"};

    if (ImPlot::BeginPlot("Energy Decay Curve", ImVec2(-1, -1), ImPlotFlags_None))
    {
        ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);

        if (autofit)
        {
            ImPlot::SetupAxes("Sample", "Amplitude", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        }
        else
        {
            ImPlot::SetupAxes("Sample", "Amplitude");
        }

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, x_axis_data.back());

        // Draw a horizontal line at the -60 dB point
        if (!show_octaves_bands)
        {
            ImPlot::PlotLine("Energy Decay Curve", x_axis_data.data(), energy_decay_curve.data(),
                             energy_decay_curve.size());
            ImPlot::PlotInfLines("EDC -60 dB", &kEDC60db, 1, ImPlotInfLinesFlags_Horizontal);
        }
        else
        {
            for (size_t i = 0; i < edc_octaves.size(); ++i)
            {
                ImPlot::PlotLine(octave_band_names[i], x_axis_data.data(), edc_octaves[i].data(),
                                 edc_octaves[i].size());
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

    constexpr size_t kCepstrumNFFT = 24000;
    constexpr size_t kNumFrequencyBins = kCepstrumNFFT / 2 + 1; // NFFT/2 + 1 for real FFT
    static std::vector<float> cepstrum_data(kCepstrumNFFT, 0.0f);
    static std::vector<float> freqs(kNumFrequencyBins, 0.0f);

    if (impulse_response_changed_.test(WindowType::Cepstrum))
    {
        impulse_response_changed_.reset(WindowType::Cepstrum);
        // copy the first 500 ms of the impulse response to the cepstrum data
        size_t num_samples = static_cast<size_t>(kCepstrumNFFT);
        if (num_samples > impulse_response_.size())
        {
            num_samples = impulse_response_.size();
        }

        auto early_rir = std::span<const float>(impulse_response_.data(), 0.25f * kSampleRate);
        cepstrum_data = AbsCepstrum(early_rir, kCepstrumNFFT);
    }

    if (ImPlot::BeginPlot("Cepstrum", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        ImPlot::PlotLine("Cepstrum", cepstrum_data.data(), cepstrum_data.size());

        ImPlot::EndPlot();
    }

    ImGui::End();
}

bool FDNToolboxApp::DrawFilterDesigner(FDNConfig& fdn_config)
{
    if (!ImGui::Begin("Filter Designer", &show_delay_filter_designer_))
    {
        ImGui::End();
        return false;
    }

    bool config_changed = false;

    constexpr size_t kNBands = 10; // Number of bands in the filter designer
    static std::vector<float> t60s = {0.228581607341766f, 0.228581607341766f, 0.256176093220711f, 0.284963846206665f,
                                      0.268932670354843f, 0.321109890937805f, 0.329257786273956f, 0.340315490961075f,
                                      0.258857980370522f, 0.125823333859444f};
    static std::vector<float> frequencies(0, 0.f);
    std::vector<float> gains(kNBands, 0.0f);

    // Oversampled vectors for plotting
    static std::vector<float> gains_plot;
    static std::vector<float> t60s_plot;
    static std::vector<float> frequencies_plot;
    static std::vector<float> filter_freqs_plot;

    bool point_changed = false;

    // Initialize with default values if empty
    if (fdn_config.t60s.size() == 0)
    {
        fdn_config.t60s = t60s;
    }
    else // Use the existing t60s from the config
    {
        t60s = fdn_config.t60s;
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
            utils::LogSpace(std::log10(frequencies[0] + 1e-6f), std::log10(frequencies.back() - 1.f), 256);
        filter_freqs_plot = utils::LogSpace(std::log10(1.f), std::log10(kSampleRate / 2.f), 512);
        t60s_plot = utils::pchip(frequencies, t60s, frequencies_plot);

        gains = utils::T60ToGainsDb(t60s, kSampleRate);
        gains_plot = utils::pchip(frequencies, gains, frequencies_plot);
        point_changed = true; // Force initial plot update
    }

    if (ImPlot::BeginPlot("Filter Designer", ImVec2(-1, ImGui::GetWindowHeight() * 0.45f), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes("Frequency (Hz)", "RT60 (s)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.01f, 5.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, kSampleRate / 2);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0.f, 10.f);

        static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

        for (size_t i = 0; i < frequencies.size(); ++i)
        {
            double freq = frequencies[i]; // The frequency should stay constant
            double t60 = t60s[i];
            point_changed |= ImPlot::DragPoint(i, &freq, &t60, ImVec4(0, 0.9f, 0, 0), 10);
            t60s[i] = std::clamp(static_cast<float>(t60), 0.01f, 10.0f); // Update the t60 value
        }

        if (point_changed)
        {
            t60s_plot = utils::pchip(frequencies, t60s, frequencies_plot);
        }

        // Plot the RT60 values
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f);
        ImPlot::PlotScatter("RT60", frequencies.data(), t60s.data(), t60s.size());

        ImPlot::PlotLine("RT60 Line", frequencies_plot.data(), t60s_plot.data(), t60s_plot.size());
        ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.250f);
        ImPlot::PlotShaded("RT60 Area", frequencies_plot.data(), t60s_plot.data(), t60s_plot.size(), 0.f);

        ImPlot::EndPlot();
    }

    static std::vector<float> H;
    static float shelf_cutoff = 8000.f;
    point_changed |= ImGui::SliderFloat("Shelf Cutoff (Hz)", &shelf_cutoff, 1000.0f, 10000.0f, "%.0f Hz");

    if (point_changed)
    {
        gains = utils::T60ToGainsDb(t60s, kSampleRate);
        gains_plot = utils::pchip(frequencies, gains, frequencies_plot);
        std::vector<float> t60s_f(t60s.begin(), t60s.end());
        std::vector<float> sos = sfFDN::GetTwoFilter(t60s_f, 593.f, kSampleRate, shelf_cutoff);

        H = utils::AbsFreqz(sos, filter_freqs_plot, kSampleRate);

        // To db gain
        for (size_t i = 0; i < H.size(); ++i)
        {
            H[i] = 20.f * std::log10(H[i]);
        }
    }

    if (ImPlot::BeginPlot("Filter preview", ImVec2(-1, ImGui::GetWindowHeight() * 0.45f), ImPlotFlags_None))
    {

        ImPlot::SetupAxes("Frequency (Hz)", "Gain (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -10.0f, 1.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, kSampleRate / 2);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -60.f, 10.f);

        static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

        ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.70f, 0.20f, 1.0f), 4.0f);
        ImPlot::PlotLine("Target Gain", frequencies_plot.data(), gains_plot.data(), frequencies_plot.size());

        if (H.size() > 0)
        {
            ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.20f, 0.20f, 1.0f), 3.0f);
            ImPlot::PlotLine("Filter Response", filter_freqs_plot.data(), H.data(), filter_freqs_plot.size());
        }

        ImPlot::EndPlot();
    }

    fdn_config.t60s = t60s;

    if (ImGui::Button("Apply"))
    {
        std::cout << "Applying filter design..." << std::endl;
        show_delay_filter_designer_ = false;
        fdn_config.delay_filter_type = DelayFilterType::TwoFilter;
        fdn_config.t60s = t60s;
        config_changed = true;
    }

    ImGui::End();
    return config_changed;
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

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, kSampleRate / 2);
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

            std::vector<float> sos = sfFDN::aceq(gains, frequencies, kSampleRate);
            H = utils::AbsFreqz(sos, frequencies_plot, kSampleRate);

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
                std::cout << "Selected Output Device: " << output_devices[i] << std::endl;
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
            if (!audio_manager_->start_audio_stream(audio_stream_option::kOutput,
                                                    std::bind(&FDNToolboxApp::AudioCallback, this,
                                                              std::placeholders::_1, std::placeholders::_2,
                                                              std::placeholders::_3),
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