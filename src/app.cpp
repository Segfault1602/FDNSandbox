#include "app.h"

#include "fdn_analyzer.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "imgui_internal.h"
#include <boost/dll/runtime_symbol_info.hpp>
#include <nlohmann/json.hpp>
#include <quill/LogMacros.h>

#include "analysis/analysis.h"
#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft_utils.h>

#include "presets.h"
#include "settings.h"
#include "utils.h"
#include "widget.h"

#include <sffdn/sffdn.h>

namespace
{
constexpr size_t kSystemBlockSize = 512; // System block size for audio processing

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
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.3f, 0.6f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.3f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.3f, 0.8f, 0.8f));
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
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.3f, 0.6f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.3f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.3f, 0.8f, 0.8f));
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

void DrawOptimizationParamGui(const std::string_view optimization_name, fdn_optimization::OptimizationInfo& opt_info)
{
    ImGui::PushItemWidth(200);
    if (optimization_name == "Adam")
    {
        static float step_size = 0.6f;
        static float learning_rate_decay = 0.99f;

        ImGui::InputFloat("Step Size", &step_size, 0.01f, 2.0f, "%.3f");
        ImGui::InputFloat("Learning Rate Decay", &learning_rate_decay, 0.8f, 1.0f, "%.4f");

        opt_info.optimizer_params = fdn_optimization::AdamParameters{
            .step_size = step_size,
            .learning_rate_decay = learning_rate_decay,
        };
    }
    else if (optimization_name == "SPSA")
    {
        static float alpha = 0.01f;
        static float gamma = 0.101f;
        static float step_size = 0.9f;
        static float evaluation_step_size = 0.9f;
        static int max_iterations = 1000000;

        ImGui::InputFloat("Alpha", &alpha, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputFloat("Gamma", &gamma, 0.05f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputFloat("Step Size", &step_size, 0.1f, 2.0f, "%.3f");
        ImGui::InputFloat("Evaluation Step Size", &evaluation_step_size, 0.1f, 2.0f, "%.3f");
        ImGui::InputInt("Max Iterations", &max_iterations);

        opt_info.optimizer_params = fdn_optimization::SPSAParameters{
            .alpha = alpha,
            .gamma = gamma,
            .step_size = step_size,
            .evaluationStepSize = evaluation_step_size,
            .max_iterations = static_cast<size_t>(max_iterations),
        };
    }
    else if (optimization_name == "Simulated Annealing")
    {
        static fdn_optimization::SimulatedAnnealingParameters sa_params;
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &sa_params.max_iterations);

        constexpr std::array kInitialTempMinMax = {100.0, 20000.0};
        ImGui::InputScalar("Initial Temperature", ImGuiDataType_Double, &sa_params.initial_temperature,
                           &kInitialTempMinMax[0], &kInitialTempMinMax[1], "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputScalar("Initial Moves", ImGuiDataType_U64, &sa_params.init_moves);
        ImGui::InputScalar("Move Control Sweep", ImGuiDataType_U64, &sa_params.move_ctrl_sweep);
        ImGui::InputScalar("Max Tolerance Sweep", ImGuiDataType_U64, &sa_params.max_tolerance_sweep);

        constexpr std::array kMaxMoveCoefMinMax = {1.0, 50.0};
        ImGui::InputScalar("Max Move Coefficient", ImGuiDataType_Double, &sa_params.max_move_coef,
                           &kMaxMoveCoefMinMax[0], &kMaxMoveCoefMinMax[1], "%.2f");

        constexpr std::array kInitMoveCoefMinMax = {0.1, 10.0};
        ImGui::InputScalar("Initial Move Coefficient", ImGuiDataType_Double, &sa_params.init_move_coef,
                           &kInitMoveCoefMinMax[0], &kInitMoveCoefMinMax[1], "%.2f");
        constexpr std::array kGainMinMax = {0.1, 1.0};
        ImGui::InputScalar("Gain", ImGuiDataType_Double, &sa_params.gain, &kGainMinMax[0], &kGainMinMax[1], "%.2f");

        opt_info.optimizer_params = sa_params;
    }
    else if (optimization_name == "Differential Evolution")
    {
        static fdn_optimization::DifferentialEvolutionParameters de_params;
        ImGui::InputScalar("Population Size", ImGuiDataType_U64, &de_params.population_size);
        ImGui::InputScalar("Max Generations", ImGuiDataType_U64, &de_params.max_generation);
        constexpr std::array kCrossoverRateMinMax = {0.0, 1.0};
        ImGui::InputScalar("Crossover Rate", ImGuiDataType_Double, &de_params.crossover_rate, &kCrossoverRateMinMax[0],
                           &kCrossoverRateMinMax[1], "%.2f");
        constexpr std::array kDifferentialWeightMinMax = {0.0, 2.0};
        ImGui::InputScalar("Differential Weight", ImGuiDataType_Double, &de_params.differential_weight,
                           &kDifferentialWeightMinMax[0], &kDifferentialWeightMinMax[1], "%.2f");

        opt_info.optimizer_params = de_params;
    }
    else if (optimization_name == "Particle Swarm Optimization")
    {
        static fdn_optimization::PSOParameters pso_params;
        ImGui::InputScalar("Number of Particles", ImGuiDataType_U64, &pso_params.num_particles);
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &pso_params.max_iterations);
        ImGui::InputScalar("Horizon Size", ImGuiDataType_U64, &pso_params.horizon_size);

        constexpr std::array kExploitationFactorMinMax = {0.1, 5.0};
        ImGui::InputScalar("Exploitation Factor", ImGuiDataType_Double, &pso_params.exploitation_factor,
                           &kExploitationFactorMinMax[0], &kExploitationFactorMinMax[1], "%.2f");
        constexpr std::array kExplorationFactorMinMax = {0.1, 5.0};
        ImGui::InputScalar("Exploration Factor", ImGuiDataType_Double, &pso_params.exploration_factor,
                           &kExplorationFactorMinMax[0], &kExplorationFactorMinMax[1], "%.2f");

        opt_info.optimizer_params = pso_params;
    }
    else if (optimization_name == "Random Search")
    {
        static fdn_optimization::RandomSearchParameters rs_params;
        // No parameters for random search currently

        opt_info.optimizer_params = rs_params;
    }
    else if (optimization_name == "L-BFGS")
    {
        static fdn_optimization::L_BFGSParameters lbfgs_params;
        ImGui::InputScalar("Number of Basis", ImGuiDataType_U64, &lbfgs_params.num_basis);

        constexpr std::array<size_t, 2> kMaxIterationsSteps = {100, 10000};
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &lbfgs_params.max_iterations, &kMaxIterationsSteps[0],
                           &kMaxIterationsSteps[1]);

        ImGui::InputScalar("Wolfe Parameter", ImGuiDataType_Double, &lbfgs_params.wolfe, nullptr, nullptr, "%.2f");
        ImGui::InputScalar("Min Gradient Norm", ImGuiDataType_Double, &lbfgs_params.min_gradient_norm, nullptr, nullptr,
                           "%.1e");
        ImGui::InputScalar("Factor", ImGuiDataType_Double, &lbfgs_params.factor, nullptr, nullptr, "%.1e");
        ImGui::InputScalar("Max Line Search Trials", ImGuiDataType_U64, &lbfgs_params.max_line_search_trials);
        ImGui::InputScalar("Min Step", ImGuiDataType_Double, &lbfgs_params.min_step, nullptr, nullptr, "%.1e");
        ImGui::InputScalar("Max Step", ImGuiDataType_Double, &lbfgs_params.max_step, nullptr, nullptr, "%.1e");

        opt_info.optimizer_params = lbfgs_params;
    }
    else if (optimization_name == "Gradient Descent")
    {
        static float step_size = 0.01f;
        static int max_iterations = 1000000;
        static float tolerance = 1e-5f;

        ImGui::InputFloat("Step Size", &step_size, 0.001f, 1.0f, "%.4f");
        ImGui::InputInt("Max Iterations", &max_iterations, 0, 0);
        ImGui::InputFloat("Tolerance", &tolerance, 0.f, 0.f, "%.1ef");

        opt_info.optimizer_params = fdn_optimization::GradientDescentParameters{
            .step_size = step_size,
            .max_iterations = static_cast<size_t>(max_iterations),
            .tolerance = tolerance,
        };
    }
    else if (optimization_name == "CMA-ES")
    {
        static size_t population_size = 0;
        static size_t max_iterations = 1000;
        static double tolerance = 1e-5;
        static double step_size = 0;

        ImGui::InputScalar("Population Size", ImGuiDataType_U64, &population_size);
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &max_iterations);
        ImGui::InputScalar("Tolerance", ImGuiDataType_Double, &tolerance);
        ImGui::InputScalar("Step Size", ImGuiDataType_Double, &step_size);

        opt_info.optimizer_params = fdn_optimization::CMAESParameters{
            .population_size = population_size,
            .max_iterations = max_iterations,
            .tolerance = tolerance,
            .step_size = step_size,
        };
    }
    else
    {
        ImGui::Text("No parameters available for this optimizer.");
    }
    ImGui::PopItemWidth();
}

} // namespace

FDNToolboxApp::FDNToolboxApp()
    : fdn_analyzer_(Settings::Instance().SampleRate(), Settings::Instance().GetLogger())
    , fdn_optimizer_(Settings::Instance().GetLogger())
    , save_ir_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_config_browser(0)
    , save_config_browser(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
{
    LOG_INFO(Settings::Instance().GetLogger(), "Starting FDN Toolbox");

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

    audio_output_buffer_.resize(Settings::Instance().SampleRate() * 0.5f);

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

    save_ir_browser.SetTitle("Save Impulse Response");
    save_ir_browser.SetTypeFilters({".wav"});

    load_config_browser.SetTitle("Load FDN Configuration");
    load_config_browser.SetTypeFilters({".json"});

    save_config_browser.SetTitle("Save FDN Configuration");
    save_config_browser.SetTypeFilters({".json"});

    show_tc_filter_designer_ = false;
    fdn_config_ = presets::kDefaultFDNConfig;
    fdn_config_A_ = presets::kDefaultFDNConfig;
    fdn_config_B_ = presets::kDefaultFDNConfig;

    spectrogram_info_ = {
#ifndef NDEBUG
        .fft_size = 1024,
        .overlap = 300,
        .window_size = 1024,
#else
        .fft_size = 2048,
        .overlap = 400,
        .window_size = 512,
#endif
        .samplerate = Settings::Instance().SampleRate(),
        .window_type = audio_utils::FFTWindowType::Hann};

    gui_fdn_ = presets::CreateDefaultFDN();
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
        LOG_ERROR(Settings::Instance().GetLogger(), "Frame size mismatch: expected {}, got {}", kSystemBlockSize,
                  frame_size);
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
        input_data[0] = 1.0f; // Set the first sample to 1.0 for impulse response
    }

    audio_file_manager_->process_block(input_data, kSystemBlockSize, 1);

    // Process the FDN for this block
    audio_fdn_->Process(input_audio_buffer, output_audio_buffer);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();

    if (audio_state_ == AudioState::ImpulseRequested)
    {
        input_data[0] -= 1.0f; // Clear the impulse sample after processing
        audio_state_ = AudioState::Idle;
    }

    // Copy to the interleaved output buffer
    const float wet_mix = dry_wet_mix * gain;
    const float dry_mix = (1.f - dry_wet_mix) * gain;

    // static float phase = 0.f;
    for (size_t i = 0; i < kSystemBlockSize; ++i)
    {
        output_data[i] = (wet_mix * output_data[i]) + (dry_mix * input_data[i]);
        // output_data[i] = std::sin(phase) * gain;
        // phase += 300.f * (2.f * std::numbers::pi / Settings::Instance().SampleRate());
    }

    audio_output_buffer_.write(output_data.data(), output_data.size());

    for (size_t i = 0; i < kSystemBlockSize; ++i)
    {
        int idx_offset = i * num_channels;
        for (size_t j = 0; j < num_channels; ++j)
        {
            output_buffer[idx_offset + j] = output_data[i];
        }
    }

    const float allowed_time = (1e9 / Settings::Instance().SampleRate()) * frame_size; // in nanoseconds
    float cpu_usage = static_cast<float>(duration) / allowed_time;
    fdn_cpu_usage_.store(cpu_usage);

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
        ImGuiID dock_id_ir = ImGui::DockBuilderSplitNode(dock_main_id, ImGuiDir_Up, 0.25f, nullptr, &dock_main_id);

        ImGui::DockBuilderDockWindow("Impulse Response", dock_id_ir);
        ImGui::DockBuilderDockWindow("Audio Player", dock_id_ir);
        ImGui::DockBuilderDockWindow("Settings", dock_id_ir);
        ImGui::DockBuilderDockWindow("Optimization", dock_id_ir);
        ImGui::DockBuilderDockWindow("FDN Configurator", dock_id_fdn);
        ImGui::DockBuilderDockWindow("Extras", dock_id_fdn);
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

    DrawMainMenuBar();

    bool config_changed = false;
    config_changed = DrawFDNConfigurator();
    config_changed |= DrawFDNExtras(config_changed);

    if (config_changed)
    {
        UpdateFDN();
    }

    DrawImpulseResponse();

    DrawAudioPlayer();

    DrawSettingsWindow();

    DrawOptimizationWindow();

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
    LOG_INFO(Settings::Instance().GetLogger(), "Configuration changed, updating FDN...");
    auto start = std::chrono::high_resolution_clock::now();

    gui_fdn_ = sfFDN::CreateFDNFromConfig(fdn_config_, Settings::Instance().SampleRate());
    gui_fdn_->SetDirectGain(0.f);

    fdn_analyzer_.SetFDN(sfFDN::CreateFDNFromConfig(fdn_config_, Settings::Instance().SampleRate()));

    other_fdn_ = sfFDN::CreateFDNFromConfig(fdn_config_, Settings::Instance().SampleRate());

    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    LOG_INFO(Settings::Instance().GetLogger(), "FDN updated in {} milliseconds", duration.count());
}

void FDNToolboxApp::DrawMainMenuBar()
{
    static bool show_audio_config_window = false;
    if (ImGui::BeginMainMenuBar())
    {
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
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "FPS: %.1f", fps);
        }
        else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "FPS: %.1f", fps);
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
            sfFDN::FDNConfig::LoadFromFile(filename, fdn_config_);
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

        try
        {
            sfFDN::FDNConfig::SaveToFile(filename, fdn_config_);
        }
        catch (const std::exception& e)
        {
            LOG_ERROR(Settings::Instance().GetLogger(), "Error saving configuration: {}", e.what());
        }
        save_config_browser.ClearSelected();
    }
}

bool FDNToolboxApp::DrawFDNConfigurator()
{
    static uint32_t random_seed = 0;

    static int min_delay = Settings::Instance().BlockSize() * 2;
    static int max_delay = 6000;

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

    DrawInputOutputGainsPlot(fdn_config_, gui_fdn_.get());
    DrawDelaysPlot(fdn_config_, max_delay);
    DrawFeedbackMatrixPlot(fdn_config_);

    uint32_t N = fdn_config_.N;
    bool fdn_size_changed = false;
    fdn_size_changed =
        ImGui::SliderScalar("N", ImGuiDataType_U32, (&N), &kNMin, &kNMax, nullptr, ImGuiSliderFlags_AlwaysClamp);

    config_changed |= fdn_size_changed;
    fdn_config_.N = N;

    static bool transpose = false;
    if (ImGui::Checkbox("Transpose", &transpose))
    {
        fdn_config_.transposed = transpose;
        config_changed = true;
    }

    config_changed |= DrawInputGainsWidget(fdn_config_);
    config_changed |= DrawOutputGainsWidget(fdn_config_);
    config_changed |= DrawDelayLengthsWidget(fdn_config_, min_delay, max_delay, random_seed);
    config_changed |= DrawScalarMatrixWidget(fdn_config_, random_seed);
    config_changed |= DrawDelayFilterWidget(fdn_config_);
    config_changed |= DrawToneCorrectionFilterDesigner(fdn_config_);

    ImGui::End();
    return config_changed;
}

bool FDNToolboxApp::DrawFDNExtras(bool force_update)
{
    bool config_changed = force_update;
    if (!ImGui::Begin("Extras"))
    {
        ImGui::End();
        return config_changed; // If the window is not open, return early
    }

    ImGui::SeparatorText("Input Gain Stage");
    ImGui::Separator();
    ImGui::TextWrapped("The following extras are added before the input gain stage.");

    bool use_input_delays = fdn_config_.input_velvet_decorrelator.has_value();
    config_changed |= ImGui::Checkbox("Velvet Decorrelator", &use_input_delays);
    if (use_input_delays)
    {
        sfFDN::VelvetNoiseDecorrelatorConfig vn_config;
        if (fdn_config_.input_velvet_decorrelator.has_value())
        {
            vn_config = *(fdn_config_.input_velvet_decorrelator);
        }

        config_changed |= DrawInputVelvetNoiseDecorrelator(vn_config, config_changed);

        fdn_config_.input_velvet_decorrelator = vn_config;
    }
    else
    {
        fdn_config_.input_velvet_decorrelator.reset();
    }

    bool use_series_schroeder = fdn_config_.input_series_schroeder_config.has_value();
    config_changed |= ImGui::Checkbox("Schroeder Allpass", &use_series_schroeder);
    if (use_series_schroeder)
    {
        sfFDN::SchroederAllpassConfig schroeder_config;
        if (fdn_config_.input_series_schroeder_config.has_value())
        {
            schroeder_config = *(fdn_config_.input_series_schroeder_config);
        }
        config_changed |= DrawInputSeriesSchroederAllpassWidget(schroeder_config, config_changed);

        fdn_config_.input_series_schroeder_config = schroeder_config;
    }
    else
    {
        fdn_config_.input_series_schroeder_config.reset();
    }

    ImGui::Separator();
    ImGui::TextWrapped(
        "The following extras are added after the input gain stage but before entering the feedback loop.");

    config_changed |= ImGui::Checkbox("Extra Delays", &fdn_config_.use_extra_delays);

    if (fdn_config_.use_extra_delays)
    {
        config_changed |= DrawExtraDelayWidget(fdn_config_, config_changed);
    }
    else
    {
        fdn_config_.input_stage_delays.clear();
    }

    bool use_schroeder_allpass = fdn_config_.input_schroeder_allpass_config.has_value();
    config_changed |= ImGui::Checkbox("Extra Schroeder Allpass", &use_schroeder_allpass);

    if (use_schroeder_allpass)
    {
        sfFDN::SchroederAllpassConfig schroeder_config;
        if (fdn_config_.input_schroeder_allpass_config.has_value())
        {
            schroeder_config = *(fdn_config_.input_schroeder_allpass_config);
        }

        config_changed |= DrawExtraSchroederAllpassWidget(schroeder_config, fdn_config_.N, config_changed);
        fdn_config_.input_schroeder_allpass_config = schroeder_config;
    }
    else
    {
        fdn_config_.input_schroeder_allpass_config.reset();
    }

    bool use_diffuser = fdn_config_.input_diffuser.has_value();
    config_changed |= ImGui::Checkbox("Input Diffuser", &use_diffuser);
    if (use_diffuser)
    {
        config_changed |= DrawDiffuserWidget(fdn_config_, config_changed);
    }
    else
    {
        fdn_config_.input_diffuser.reset();
    }

    bool use_mc_velvet = fdn_config_.input_velvet_decorrelator_mc.has_value();
    config_changed |= ImGui::Checkbox("Velvet Decorrelator MC", &use_mc_velvet);
    if (use_mc_velvet)
    {
        sfFDN::VelvetNoiseDecorrelatorConfig vn_config;
        if (fdn_config_.input_velvet_decorrelator_mc.has_value())
        {
            vn_config = *(fdn_config_.input_velvet_decorrelator_mc);
        }

        static uint32_t selected_ovn_input = 0;
        config_changed |= DrawInputVelvetNoiseDecorrelatorMultiChannel(vn_config, selected_ovn_input, config_changed);

        fdn_config_.input_velvet_decorrelator_mc = vn_config;
    }
    else
    {
        fdn_config_.input_velvet_decorrelator_mc.reset();
    }

    ImGui::SeparatorText("Feedback Loop");
    ImGui::Separator();
    ImGui::TextWrapped(
        "The following extras are added inside the feedback loop, between the delay lines and the attenuation filter.");

    bool use_feedback_schroeder = fdn_config_.feedback_schroeder_allpass_config.has_value();
    config_changed |= ImGui::Checkbox("Feedback Schroeder Allpass", &use_feedback_schroeder);
    if (use_feedback_schroeder)
    {
        sfFDN::SchroederAllpassConfig schroeder_config;
        if (fdn_config_.feedback_schroeder_allpass_config.has_value())
        {
            schroeder_config = *(fdn_config_.feedback_schroeder_allpass_config);
        }
        config_changed |= DrawExtraSchroederAllpassWidget(schroeder_config, fdn_config_.N, config_changed);

        fdn_config_.feedback_schroeder_allpass_config = schroeder_config;
    }
    else
    {
        fdn_config_.feedback_schroeder_allpass_config.reset();
    }

    bool use_time_varying_delays = fdn_config_.time_varying_delays.has_value();
    config_changed |= ImGui::Checkbox("Time-Varying Delays", &use_time_varying_delays);
    if (use_time_varying_delays)
    {
        sfFDN::TimeVaryingDelayConfig tvd_config;
        if (fdn_config_.time_varying_delays.has_value())
        {
            tvd_config = *(fdn_config_.time_varying_delays);
        }

        config_changed |= DrawTimeVaryingDelayWidget(tvd_config, fdn_config_.N, config_changed);

        fdn_config_.time_varying_delays = tvd_config;
    }
    else
    {
        fdn_config_.time_varying_delays.reset();
    }

    ImGui::SeparatorText("Output Gain Stage");
    ImGui::Separator();
    ImGui::TextWrapped("The following extras are added before the output gain stage, outside the feedback loop.");

    bool use_output_velvet_mc = fdn_config_.output_velvet_decorrelator_mc.has_value();
    config_changed |= ImGui::Checkbox("Velvet Decorrelator MC ##Output", &use_output_velvet_mc);
    if (use_output_velvet_mc)
    {
        sfFDN::VelvetNoiseDecorrelatorConfig vn_config;
        if (fdn_config_.output_velvet_decorrelator_mc.has_value())
        {
            vn_config = *(fdn_config_.output_velvet_decorrelator_mc);
        }

        static uint32_t selected_ovn_output = 0;
        config_changed |= DrawInputVelvetNoiseDecorrelatorMultiChannel(vn_config, selected_ovn_output, config_changed);

        fdn_config_.output_velvet_decorrelator_mc = vn_config;
    }
    else
    {
        fdn_config_.output_velvet_decorrelator_mc.reset();
    }

    ImGui::Separator();
    ImGui::TextWrapped("The following extras are added after the output gain stage.");

    bool use_output_schroeder = fdn_config_.output_schroeder_allpass_config.has_value();
    config_changed |= ImGui::Checkbox("Output Schroeder Allpass", &use_output_schroeder);
    if (use_output_schroeder)
    {
        sfFDN::SchroederAllpassConfig schroeder_config;
        if (fdn_config_.output_schroeder_allpass_config.has_value())
        {
            schroeder_config = *(fdn_config_.output_schroeder_allpass_config);
        }
        config_changed |= DrawInputSeriesSchroederAllpassWidget(schroeder_config, config_changed);

        fdn_config_.output_schroeder_allpass_config = schroeder_config;
    }
    else
    {
        fdn_config_.output_schroeder_allpass_config.reset();
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
        auto imp_response = fdn_analyzer_.GetImpulseResponse();

        if (fdn_analyzer_.IsClipping())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Warning: Impulse response is clipping!");
        }

        ImPlot::SetupAxes("Sample", "Amplitude", ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Once);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, imp_response.size() - 1);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 1.0f);

        ImPlot::PlotLine("Impulse Response", imp_response.data(), imp_response.size());
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

    static float gain = 1.0f;
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("Gain", &gain, 0.0f, 10.0f, "%.2f"))
    {
        audio_gain_ = gain; // Update the audio gain
    }

    static float mix = 0.5f;
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("Dry/Wet", &mix, 0.0f, 1.0f, "%.2f"))
    {
        dry_wet_mix_ = mix; // Update the dry/wet mix
    }

    static sfFDN::OnePoleFilter rms_filter;
    rms_filter.SetPole(0.90f);
    constexpr uint32_t kRMSBlockSize = 1024;
    static std::vector<float> rms_buffer(kRMSBlockSize);
    auto read_available = audio_output_buffer_.get_read_available();
    static float rms_value = 0.0f;
    static bool clipping_warning_displayed = false;
    static float clipping_debounce_timer = 0.0f;

    clipping_debounce_timer += ImGui::GetIO().DeltaTime;

    while (read_available >= kRMSBlockSize)
    {
        size_t sample_read = kRMSBlockSize;
        audio_output_buffer_.read(rms_buffer.data(), sample_read);
        if (sample_read > 0)
        {
            auto rms = utils::ComputeRMS(rms_buffer, kRMSBlockSize, kRMSBlockSize);
            for (const auto& sample : rms)
            {
                rms_value = rms_filter.Tick(sample);
            }

            for (const auto& sample : rms)
            {
                if (std::abs(sample) >= 1.0f)
                {
                    clipping_warning_displayed = true;
                    clipping_debounce_timer = 0.f;
                    break;
                }
            }
        }
        read_available -= sample_read;
    }

    if (clipping_warning_displayed && clipping_debounce_timer >= 1.0f)
    {
        clipping_warning_displayed = false;
    }

    ImGui::Text("RMS Level (dB): %.2f", rms_value);
    float rms_db = 20.f * std::log10(rms_value + 1e-10f);
    ImGui::Text("RMS Level (dB FS): %.2f dB FS", rms_db);

    ImGui::Text("Level:");
    ImGui::SameLine();
    if (clipping_warning_displayed)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
    }
    else if (rms_value < 0.5f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
    }
    else if (rms_value >= 0.5f && rms_value < 0.6f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
    }
    ImGui::ProgressBar(rms_value, ImVec2(200.0f, 0.0f));

    ImGui::PopStyleColor();

    static sfFDN::OnePoleFilter cpu_usage_filter;
    cpu_usage_filter.SetPole(0.99f); // Set the pole for the low-pass filter
    float cpu_usage = fdn_cpu_usage_.load();
    cpu_usage = cpu_usage_filter.Tick(cpu_usage); // Apply low-pass filter to CPU usage
    ImGui::Text("CPU Usage: %.2f%%", cpu_usage * 100.0f);

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
        spectrogram_info_.fft_size = 512 * (1 << selected_fft_size_index);
        if (spectrogram_info_.fft_size < spectrogram_info_.window_size)
        {
            spectrogram_info_.window_size = spectrogram_info_.fft_size;
            spectrogram_info_.overlap = static_cast<uint32_t>(overlap * spectrogram_info_.window_size);
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
        spectrogram_info_.window_size = 256 * (1 << selected_window_size_index);
        spectrogram_info_.overlap = static_cast<uint32_t>(overlap * spectrogram_info_.window_size);
        if (spectrogram_info_.window_size > spectrogram_info_.fft_size)
        {
            spectrogram_info_.fft_size = spectrogram_info_.window_size;
            selected_fft_size_index = selected_window_size_index - 1;
        }
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Overlap:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderFloat("##Overlap", &overlap, 0.01f, 0.95f, "%.2f"))
    {
        spectrogram_info_.overlap = static_cast<uint32_t>(overlap * spectrogram_info_.window_size);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Window Type:");
    ImGui::SameLine(kColOffset);
    static int selected_window_type = 2; // Default to Hann
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowType", &selected_window_type, "Rectangular\0Hamming\0Hann\0Blackman\0"))
    {
        spectrogram_info_.window_type = static_cast<audio_utils::FFTWindowType>(selected_window_type);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::SeparatorText("Style Settings");

    ImPlotStyle& style = ImPlot::GetStyle();

    float line_weight = style.LineWeight;
    ImGui::SetNextItemWidth(200);
    if (ImGui::SliderFloat("Line Weight", &line_weight, 1.0f, 5.0f))
    {
        style.LineWeight = line_weight;
    }

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

    float content_region_width = ImGui::GetContentRegionAvail().x;

    {
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
        ImGui::BeginChild("Optimization Parameters", ImVec2(content_region_width * 0.25f, -1), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_None);
        ImGui::SeparatorText("Setup");
        static bool optimize_gains = false;
        static bool optimize_matrix = false;
        static bool optimize_delays = false;
        ImGui::Checkbox("Optimize Gains", &optimize_gains);
        ImGui::Checkbox("Optimize Matrix", &optimize_matrix);
        ImGui::Checkbox("Optimize Delays", &optimize_delays);

        bool is_optimizing = fdn_optimizer_.GetStatus() == fdn_optimization::OptimizationStatus::Running;
        ImGui::BeginDisabled(is_optimizing);

        ImGui::SeparatorText("Objective Weights");
        static float spectral_flatness_weight = 1.0;
        static float sparsity_weight = 0.5;
        static float power_envelope_weight = 1.0;

        ImGui::PushItemWidth(100);

        ImGui::InputFloat("Spectral Flatness Weight", &spectral_flatness_weight, 0.0, 10.0, "%.2f");
        ImGui::InputFloat("Sparsity Weight", &sparsity_weight, 0.0, 10.0, "%.2f");
        ImGui::InputFloat("Power Envelope Weight", &power_envelope_weight, 0.0, 10.0, "%.2f");

        ImGui::PopItemWidth();

        ImGui::SeparatorText("Optimization Parameters");

        constexpr std::array<std::string_view, 9> kOptimizationAlgorithms = {"Adam",
                                                                             "SPSA",
                                                                             "Simulated Annealing",
                                                                             "Differential Evolution",
                                                                             "Particle Swarm Optimization",
                                                                             "Random Search",
                                                                             "L-BFGS",
                                                                             "Gradient Descent",
                                                                             "CMA-ES"};
        static int selected_algorithm = 0;
        if (ImGui::BeginCombo("Algorithm", kOptimizationAlgorithms[selected_algorithm].data()))
        {
            for (int i = 0; i < static_cast<int>(kOptimizationAlgorithms.size()); ++i)
            {
                bool is_selected = (selected_algorithm == i);
                if (ImGui::Selectable(kOptimizationAlgorithms[i].data(), is_selected))
                {
                    selected_algorithm = i;
                }
            }
            ImGui::EndCombo();
        }

        static fdn_optimization::OptimizationInfo opt_info;

        DrawOptimizationParamGui(kOptimizationAlgorithms[selected_algorithm], opt_info);

        ImGui::PushItemWidth(200);
        ImGui::SeparatorText("Gradient Settings");
        constexpr std::array<std::string_view, 2> kGradientMethods = {"Central Difference", "Forward Difference"};
        static int selected_gradient_method = 0;
        static fdn_optimization::GradientMethod gradient_method = fdn_optimization::GradientMethod::CentralDifferences;
        if (ImGui::BeginCombo("Gradient Method", kGradientMethods[selected_gradient_method].data()))
        {
            for (int i = 0; i < static_cast<int>(kGradientMethods.size()); ++i)
            {
                bool is_selected = (selected_gradient_method == i);
                if (ImGui::Selectable(kGradientMethods[i].data(), is_selected))
                {
                    selected_gradient_method = i;
                    gradient_method = (i == 0) ? fdn_optimization::GradientMethod::CentralDifferences
                                               : fdn_optimization::GradientMethod::ForwardDifferences;
                }
            }
            ImGui::EndCombo();
        }

        static float gradient_step_size = 1e-4f;
        ImGui::InputFloat("Gradient Step Size", &gradient_step_size, 0.0f, 1.0f, "%.1ef");
        ImGui::PopItemWidth();

        if (ImGui::Button("Run Optimization"))
        {
            if (fdn_optimizer_.GetStatus() != fdn_optimization::OptimizationStatus::Running)
            {
                std::vector<fdn_optimization::OptimizationParamType> params_to_optimize;
                if (optimize_gains)
                {
                    params_to_optimize.push_back(fdn_optimization::OptimizationParamType::Gains);
                }
                if (optimize_matrix)
                {
                    params_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix);
                }
                if (optimize_delays)
                {
                    params_to_optimize.push_back(fdn_optimization::OptimizationParamType::Delays);
                }

                if (!params_to_optimize.empty())
                {

                    opt_info.parameters_to_optimize = params_to_optimize;
                    opt_info.initial_fdn_config = fdn_config_;
                    opt_info.ir_size = Settings::Instance().SampleRate();
                    opt_info.gradient_method = gradient_method;
                    opt_info.gradient_delta = gradient_step_size;
                    fdn_optimizer_.StartOptimization(opt_info);
                    ImGui::OpenPopup("Optimization Progress");
                }
            }
        }
        ImGui::EndDisabled();

        ImGui::SetNextWindowSize(ImVec2(600, -1), ImGuiCond_Always);
        if (ImGui::BeginPopupModal("Optimization Progress", nullptr, ImGuiWindowFlags_None))
        {
            fdn_optimization::OptimizationProgressInfo progress = fdn_optimizer_.GetProgress();

            std::chrono::duration<double, std::chrono::seconds::period> elapsed_seconds = progress.elapsed_time;
            ImGui::Text("Elapsed Time: %.2f seconds", elapsed_seconds.count());

            uint32_t eval_count = progress.evaluation_count;
            ImGui::Text("Evaluations: %u", eval_count);

            if (!progress.loss_history.empty() &&
                ImPlot::BeginPlot("Loss Progress", ImVec2(-1, 250), ImPlotAxisFlags_None))
            {
                const std::vector<double>& loss_history_vec = progress.loss_history[0];
                ImPlot::SetupAxes("Evaluation", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(loss_history_vec.size() * 1.25),
                                        ImPlotCond_Always);

                double max_loss = loss_history_vec.empty()
                                      ? 1.0
                                      : *std::max_element(loss_history_vec.begin(), loss_history_vec.end());
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_loss * 1.25, ImPlotCond_Always);

                ImPlot::PlotLine("Loss", loss_history_vec.data(), static_cast<int>(loss_history_vec.size()));

                for (auto i = 1u; i < progress.loss_history.size(); ++i)
                {
                    const std::vector<double>& other_loss_history = progress.loss_history[i];
                    ImPlot::PlotLine(("Loss " + std::to_string(i)).c_str(), other_loss_history.data(),
                                     static_cast<int>(other_loss_history.size()));
                }

                ImPlot::EndPlot();
            }

            ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-1, 0.0f), "Optimizing...");
            if (ImGui::Button("Cancel"))
            {
                fdn_optimizer_.CancelOptimization();
            }

            if (fdn_optimizer_.GetStatus() != fdn_optimization::OptimizationStatus::Running)
            {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("Optimization Results", ImVec2(content_region_width * 0.25f, -1), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_None);
        ImGui::SeparatorText("Results");

        static double elapsed_time_sec = 0;
        static uint32_t evaluation_count = 0;
        static double initial_loss = 0.0;
        static double final_loss = 0.0;
        static std::vector<double> loss_history;
        static std::vector<std::vector<double>> all_loss_histories;
        static std::vector<std::string> loss_names;

        if (fdn_optimizer_.GetStatus() == fdn_optimization::OptimizationStatus::Completed ||
            fdn_optimizer_.GetStatus() == fdn_optimization::OptimizationStatus::Canceled)
        {
            fdn_optimization::OptimizationResult result = fdn_optimizer_.GetResult();
            elapsed_time_sec = std::chrono::duration<double, std::chrono::seconds::period>(result.total_time).count();
            evaluation_count = result.total_evaluations;
            initial_loss = result.loss_history[0].front();
            final_loss = result.loss_history[0].back();
            loss_history = result.loss_history[0];
            all_loss_histories = result.loss_history;
            loss_names = result.loss_names;
            if (optimize_gains)
            {
                fdn_config_.input_gains = result.optimized_fdn_config.input_gains;
                fdn_config_.output_gains = result.optimized_fdn_config.output_gains;
            }
            if (optimize_matrix)
            {
                fdn_config_.matrix_info = result.optimized_fdn_config.matrix_info;
            }
            if (optimize_delays)
            {
                fdn_config_.delays = result.optimized_fdn_config.delays;
            }

            UpdateFDN();
        }
        auto last_status = fdn_optimizer_.GetStatus();
        if (last_status == fdn_optimization::OptimizationStatus::Canceled ||
            last_status == fdn_optimization::OptimizationStatus::Completed)
        {
            fdn_optimizer_.ResetStatus();
        }

        ImGui::Text("Elapsed Time: %.2f seconds", elapsed_time_sec);
        ImGui::Text("Evaluations: %u", evaluation_count);
        ImGui::Text("Initial Loss: %.6f", initial_loss);
        ImGui::Text("Final Loss: %.6f", final_loss);

        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("Loss Plot", ImVec2(-1, -1), ImGuiChildFlags_Borders, ImGuiWindowFlags_None);

        if (ImPlot::BeginPlot("Loss Progress", ImVec2(-1, -1), ImPlotAxisFlags_None))
        {
            ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
            ImPlot::SetupAxes("Evaluation", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(loss_history.size() * 1.25), ImPlotCond_Once);

            double max_loss = loss_history.empty() ? 1.0 : *std::max_element(loss_history.begin(), loss_history.end());
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, max_loss * 1.25, ImPlotCond_Once);

            ImPlot::PlotLine("Total Loss", loss_history.data(), static_cast<int>(loss_history.size()));

            for (auto i = 1u; i < all_loss_histories.size(); ++i)
            {
                const std::vector<double>& other_loss_history = all_loss_histories[i];
                ImPlot::PlotLine((loss_names[i - 1]).c_str(), other_loss_history.data(),
                                 static_cast<int>(other_loss_history.size()));
            }
            ImPlot::EndPlot();
        }
        ImGui::EndChild();

        ImGui::PopStyleVar();
    }

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

    static float min_dB = -80.0;
    static float max_dB = 10.0;
    static std::vector<float> mels{};
    static MelFormatterContext ctx{.mel_frequencies = mels};

    ImPlot::PushColormap(ImPlotColormap_Plasma);

    if (ImPlot::BeginPlot("##Spectrogram", ImVec2(ImGui::GetCurrentWindow()->Size[0] * 0.92f, -1),
                          ImPlotFlags_NoMouseText))
    {
        fdn_analysis::SpectrogramData spectrogram_data =
            fdn_analyzer_.GetSpectrogram(spectrogram_info_, spectrogram_type_ == SpectrogramType::Mel);

        double bin_count = Settings::Instance().SampleRate() / 2.0;

        if (spectrogram_type_ == SpectrogramType::Mel)
        {
            mels = audio_utils::GetMelFrequencies(spectrogram_data.bin_count, 0.f,
                                                  Settings::Instance().SampleRate() / 2.f);
            ctx.mel_frequencies = mels;
            ImPlot::SetupAxisFormat(ImAxis_Y1, MelFormatter, &ctx);
            bin_count = spectrogram_data.bin_count;
        }

        const double tmin = 0.f;
        const double tmax =
            fdn_analyzer_.GetImpulseResponseSize() / static_cast<double>(Settings::Instance().SampleRate());
        ImPlot::SetupAxisLimits(ImAxis_X1, tmin, tmax, ImGuiCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.f, bin_count, ImGuiCond_Always);

        ImPlot::PlotHeatmap("##Spectrogram", spectrogram_data.data.data(), spectrogram_data.bin_count,
                            spectrogram_data.frame_count, min_dB, max_dB, nullptr, {tmin, 0}, {tmax, bin_count},
                            ImPlotHeatmapFlags_ColMajor);

        // Add colorbar

        ImPlot::EndPlot();
    }

    ImGui::SameLine();

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
    ImPlot::PopColormap();

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
            ImPlotAxisFlags x_axis_flags = ImPlotAxisFlags_None;
            ImPlotAxisFlags y_axis_flags = ImPlotAxisFlags_None;

            // if (plot_type_changed && !lock_freq_range)
            // {
            //     x_axis_flags |= ImPlotAxisFlags_AutoFit;
            //     y_axis_flags |= ImPlotAxisFlags_AutoFit;
            // }

            if (peak_radio < 3)
            {
                ImPlot::SetupAxes("Frequency (Hz)", "Magnitude (dB)", x_axis_flags, y_axis_flags);
                // ImPlot::SetupAxesLimits(frequency_range_min, frequency_range_max, -60.0, 0.0,
                //                         (lock_freq_range) ? ImPlotCond_Always : ImPlotCond_Once);

                ImPlot::SetupAxisLimits(ImAxis_Y1, -60.0f, 10.f, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.f, Settings::Instance().SampleRate() / 2.f, ImPlotCond_Once);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, spectrum_data.frequency_bins.back());
            }

            if (peak_radio == 0 || peak_radio == 2)
            {
                ImPlot::PlotLine("Spectrum", spectrum_data.frequency_bins.data(), spectrum_data.spectrum.data(),
                                 spectrum_data.spectrum.size());
            }
            if (peak_radio == 1 || peak_radio == 2)
            {
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Asterisk, 2.0f);
                ImPlot::PlotScatter("Peaks", spectrum_data.peaks_freqs.data(), spectrum_data.peaks.data(),
                                    spectrum_data.peaks.size(), ImPlotScatterFlags_None);
            }

            if (peak_radio == 3)
            {
                ImPlot::SetupAxes("Magnitude (dB)", "Count");
                ImPlot::SetupAxesLimits(-60.0, 0.0, 0.0, 600.0, ImPlotCond_Always);

                ImPlot::PlotHistogram("Histogram", spectrum_data.peaks.data(), spectrum_data.peaks.size(),
                                      ImPlotBin_Sqrt, 0.95f, ImPlotRange(-60, 0), ImPlotHistogramFlags_None);
            }

            // if (!lock_freq_range)
            // {
            //     auto plot_limit = ImPlot::GetPlotLimits();
            //     frequency_range_min = static_cast<float>(plot_limit.X.Min);
            //     frequency_range_max = static_cast<float>(plot_limit.X.Max);
            // }

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
            ImPlot::SetupAxisLimits(ImAxis_X1, -1000.0f, xcorr_span.size(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -1000, xcorr_span.size() + 100);

            ImPlot::PlotLine("Autocorrelation", xcorr_span.data(), xcorr_span.size());
            ImPlot::EndPlot();
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

    if (ImGui::BeginTabBar("Filter Responses"))
    {
        auto filter_data = fdn_analyzer_.GetFilterData();
        float min_mag = 0.f;
        for (const auto& mag_response : filter_data.mag_responses)
        {
            auto min_response = *std::ranges::min_element(mag_response);
            min_mag = std::min(min_response, min_mag);
        }

        if (ImGui::BeginTabItem("Attenuation filters"))
        {
            if (ImPlot::BeginSubplots("Attenuation Filters Mag/Phase", 2, 1, ImVec2(-1, -1),
                                      ImPlotSubplotFlags_LinkAllX))
            {
                if (ImPlot::BeginPlot("Magnitude", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f),
                                      ImPlotFlags_None))
                {
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
                    // ImPlot::SetupAxes("Frequency", "Magnitude (dB)", ImPlotAxisFlags_AutoFit,
                    // ImPlotAxisFlags_AutoFit);

                    // ImPlot::SetupAxisLimits(ImAxis_Y1, 0, min_mag * 1.5f, ImPlotCond_Once);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    for (uint32_t i = 0; i < filter_data.mag_responses.size(); ++i)
                    {
                        auto mag_response = filter_data.mag_responses[i];
                        std::string line_name = "Delay filter " + std::to_string(i + 1);
                        ImPlot::PlotLine(line_name.c_str(), filter_data.frequency_bins.data(), mag_response.data(),
                                         mag_response.size());
                    }
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_None))
                {
                    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_None);
                    ImPlot::SetupAxes("Frequency", "Phase", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    for (uint32_t i = 0; i < filter_data.mag_responses.size(); ++i)
                    {
                        auto phase_response = filter_data.phase_responses[i];
                        std::string line_name = "Delay filter " + std::to_string(i + 1);
                        ImPlot::PlotLine(line_name.c_str(), filter_data.frequency_bins.data(), phase_response.data(),
                                         phase_response.size());
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
                    ImPlot::SetupAxes("Frequency", "Magnitude (dB)", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    ImPlot::PlotLine("TC Filter", filter_data.frequency_bins.data(), filter_data.tc_mag_response.data(),
                                     filter_data.tc_mag_response.size());
                    ImPlot::EndPlot();
                }

                if (ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_NoLegend))
                {
                    ImPlot::SetupAxes("Frequency", "Phase", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

                    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                    ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_ticks.data(), frequencies_ticks.size(), nullptr,
                                           false);

                    ImPlot::PlotLine("TC Filter", filter_data.frequency_bins.data(),
                                     filter_data.tc_phase_response.data(), filter_data.tc_phase_response.size());

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
            for (auto i = 0; i < octave_band_names.size(); ++i)
            {
                constexpr uint32_t kDownsampleFactor = 4;
                ImPlot::PlotLine(octave_band_names[i], fdn_analyzer_.GetTimeData().data(),
                                 edc_data.edc_octaves[i].data(), edc_data.edc_octaves[i].size() / kDownsampleFactor,
                                 ImPlotLineFlags_None, 0, kDownsampleFactor * sizeof(float));
            }
        }

        ImPlot::EndPlot();
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
        auto ir = fdn_analyzer_.GetImpulseResponse();
        auto time_data = fdn_analyzer_.GetTimeData();

        ImPlot::SetupAxes("Sample", "Amplitude", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.5f, ImPlotCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.01f, time_data.back(), ImPlotCond_Once);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -0.01, time_data.back());
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 2.0f);

        ImPlot::PlotLine("Impulse Response", time_data.data(), ir.data(), ir.size());

        ImPlot::PlotLine("Echo Density", echo_density_data.sparse_indices.data(), echo_density_data.echo_density.data(),
                         echo_density_data.sparse_indices.size());

        if (echo_density_data.mixing_time < time_data.back())
        {
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), 2.0f);
            ImPlot::PlotInfLines("Mixing Time", &echo_density_data.mixing_time, 1, ImPlotInfLinesFlags_None);
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