#include "optimization_gui.h"

#include <imgui.h>
#include <implot.h>
#include <quill/LogMacros.h>
#include <quill/std/Vector.h>

#include <sffdn/sffdn.h>

#include "optimizer.h"
#include "settings.h"

#include <span>

namespace
{
fdn_optimization::OptimizationAlgoParams DrawOptimizationParamGui(fdn_optimization::OptimizationAlgoType algo_type)
{
    ImGui::PushItemWidth(200);
    if (algo_type == fdn_optimization::OptimizationAlgoType::Adam)
    {
        static fdn_optimization::AdamParameters params;

        ImGui::InputFloat("Step Size", &params.step_size, 0.01f, 2.0f, "%.3f");
        ImGui::InputFloat("Learning Rate Decay", &params.learning_rate_decay, 0.8f, 1.0f, "%.4f");
        ImGui::InputInt("Decay Step Size", reinterpret_cast<int*>(&params.decay_step_size));
        ImGui::InputInt("Epoch Restarts", reinterpret_cast<int*>(&params.epoch_restarts));
        ImGui::InputInt("Max Restarts", reinterpret_cast<int*>(&params.max_restarts));
        ImGui::InputFloat("Tolerance", &params.tolerance, 1e-6f, 1e-3f, "%.1e", ImGuiSliderFlags_Logarithmic);

        params.decay_step_size = std::max(1, static_cast<int>(params.decay_step_size));
        params.epoch_restarts = std::max(0, static_cast<int>(params.epoch_restarts));
        params.max_restarts = std::max(0, static_cast<int>(params.max_restarts));

        return params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::SPSA)
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

        return fdn_optimization::SPSAParameters{
            .alpha = alpha,
            .gamma = gamma,
            .step_size = step_size,
            .evaluationStepSize = evaluation_step_size,
            .max_iterations = static_cast<size_t>(max_iterations),
        };
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::SimulatedAnnealing)
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

        return sa_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::DifferentialEvolution)
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

        return de_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::PSO)
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

        return pso_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::RandomSearch)
    {
        static fdn_optimization::RandomSearchParameters rs_params;
        // No parameters for random search currently

        return rs_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::L_BFGS)
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

        return lbfgs_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::GradientDescent)
    {
        static float step_size = 0.01f;
        static int max_iterations = 1000000;
        static float tolerance = 1e-5f;

        ImGui::InputFloat("Step Size", &step_size, 0.001f, 1.0f, "%.4f");
        ImGui::InputInt("Max Iterations", &max_iterations, 0, 0);
        ImGui::InputFloat("Tolerance", &tolerance, 0.f, 0.f, "%.1ef");

        return fdn_optimization::GradientDescentParameters{
            .step_size = step_size,
            .max_iterations = static_cast<size_t>(max_iterations),
            .tolerance = tolerance,
        };
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::CMAES)
    {
        static size_t population_size = 0;
        static size_t max_iterations = 1000;
        static double tolerance = 1e-5;
        static double step_size = 0;

        ImGui::InputScalar("Population Size", ImGuiDataType_U64, &population_size);
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &max_iterations);
        ImGui::InputScalar("Tolerance", ImGuiDataType_Double, &tolerance);
        ImGui::InputScalar("Step Size", ImGuiDataType_Double, &step_size);

        return fdn_optimization::CMAESParameters{
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

    return {};
}

} // namespace

OptimizationGUI::OptimizationGUI(quill::Logger* logger)
    : fdn_optimizer_(logger)
    , logger_(logger)
{
}

bool OptimizationGUI::Draw(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
{
    bool updated_fdn = false;
    float content_region_width = ImGui::GetContentRegionAvail().x;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild("Optimization Parameters", ImVec2(content_region_width * 0.25f, -1), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_None);
    ImGui::SeparatorText("Setup");

    opt_info_.parameters_to_optimize.clear();
    ImGui::Checkbox("Optimize Gains", &optimize_gains_checkbox_);

    if (optimize_gains_checkbox_)
    {
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Gains);
    }

    ImGui::Checkbox("Optimize Matrix", &optimize_matrix_checkbox_);
    if (optimize_matrix_checkbox_)
    {
        static int matrix_type = 0;
        ImGui::RadioButton("Random", &matrix_type, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Householder", &matrix_type, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Circulant", &matrix_type, 2);

        if (matrix_type == 0)
        {
            opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix);
        }
        else if (matrix_type == 1)
        {
            opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix_Householder);
        }
        else if (matrix_type == 2)
        {
            opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix_Circulant);
        }
    }

    const bool has_target_rir = !target_rir.empty();
    ImGui::BeginDisabled(!has_target_rir);
    ImGui::Checkbox("Optimize Filters", &optimize_filters_checkbox_);
    ImGui::EndDisabled();

    if (optimize_filters_checkbox_)
    {
        optimize_gains_checkbox_ = false;
        optimize_matrix_checkbox_ = false;
        opt_info_.parameters_to_optimize.clear();
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::AttenuationFilters);
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::TonecorrectionFilters);
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::OverallGain);
    }

    constexpr double kMaxWeight = 2.0;
    constexpr double kMinWeight = 0.0;

    if (optimize_gains_checkbox_ || optimize_matrix_checkbox_)
    {
        ImGui::SeparatorText("Objective Weights");

        ImGui::PushItemWidth(100);

        ImGui::SliderScalar("Spectral Flatness Weight", ImGuiDataType_Double, &opt_info_.spectral_flatness_weight,
                            &kMinWeight, &kMaxWeight, "%.2f");
        ImGui::SliderScalar("Sparsity Weight", ImGuiDataType_Double, &opt_info_.sparsity_weight, &kMinWeight,
                            &kMaxWeight, "%.2f");
        ImGui::SliderScalar("Power Envelope Weight", ImGuiDataType_Double, &opt_info_.power_envelope_weight,
                            &kMinWeight, &kMaxWeight, "%.2f");

        ImGui::PopItemWidth();
    }
    else if (optimize_filters_checkbox_)
    {
        ImGui::SeparatorText("RIR Match Weights");

        ImGui::PushItemWidth(100);

        ImGui::SliderScalar("EDC Weight", ImGuiDataType_Double, &opt_info_.edc_weight, &kMinWeight, &kMaxWeight,
                            "%.2f");
        ImGui::SliderScalar("Mel EDR Weight", ImGuiDataType_Double, &opt_info_.mel_edr_weight, &kMinWeight, &kMaxWeight,
                            "%.2f");

        constexpr std::array kFFTSizeOptions = {"512", "1024", "2048", "4096", "8192"};
        static int selected_fft_size_index = 3; // Default to 4096
        if (ImGui::BeginCombo("Mel EDR FFT Size", kFFTSizeOptions[selected_fft_size_index]))
        {
            for (int i = 0; i < static_cast<int>(kFFTSizeOptions.size()); ++i)
            {
                bool is_selected = (selected_fft_size_index == i);
                if (ImGui::Selectable(kFFTSizeOptions[i], is_selected))
                {
                    selected_fft_size_index = i;
                    opt_info_.mel_edr_fft_length = 512 * static_cast<uint32_t>(1 << i);
                }
            }
            ImGui::EndCombo();
        }

        ImGui::InputScalar("Mel EDR Hop Size", ImGuiDataType_U32, &opt_info_.mel_edr_hop_size);
        ImGui::InputScalar("Mel EDR Window Size", ImGuiDataType_U32, &opt_info_.mel_edr_window_size);
        ImGui::InputScalar("Mel EDR Num Bands", ImGuiDataType_U32, &opt_info_.mel_edr_num_bands);

        opt_info_.mel_edr_window_size = std::clamp(opt_info_.mel_edr_window_size, 256u, opt_info_.mel_edr_fft_length);
        opt_info_.mel_edr_hop_size = std::clamp(opt_info_.mel_edr_hop_size, 32u, opt_info_.mel_edr_window_size - 1);
        opt_info_.mel_edr_num_bands = std::clamp(opt_info_.mel_edr_num_bands, 8u, 128u);

        ImGui::PopItemWidth();
    }

    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("Optimization Results", ImVec2(content_region_width * 0.25f, -1), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_None);

    ImGui::SeparatorText("Optimization Parameters");

    static fdn_optimization::OptimizationAlgoType selected_algorithm = fdn_optimization::OptimizationAlgoType::Adam;
    if (ImGui::BeginCombo("Algorithm", fdn_optimization::OptimizationAlgoTypeToString(selected_algorithm)))
    {
        for (int i = 0; i < static_cast<int>(fdn_optimization::OptimizationAlgoType::Count); ++i)
        {
            bool is_selected = (selected_algorithm == static_cast<fdn_optimization::OptimizationAlgoType>(i));
            if (ImGui::Selectable(fdn_optimization::OptimizationAlgoTypeToString(
                                      static_cast<fdn_optimization::OptimizationAlgoType>(i)),
                                  is_selected))
            {
                selected_algorithm = static_cast<fdn_optimization::OptimizationAlgoType>(i);
            }
        }
        ImGui::EndCombo();
    }

    opt_info_.optimizer_params = DrawOptimizationParamGui(selected_algorithm);

    if (selected_algorithm >= fdn_optimization::OptimizationAlgoType::Adam)
    {
        ImGui::PushItemWidth(200);
        ImGui::SeparatorText("Gradient Settings");
        constexpr std::array<std::string_view, 2> kGradientMethods = {"Central Difference", "Forward Difference"};
        static int selected_gradient_method = 0;
        if (ImGui::BeginCombo("Gradient Method", kGradientMethods[selected_gradient_method].data()))
        {
            for (int i = 0; i < static_cast<int>(kGradientMethods.size()); ++i)
            {
                bool is_selected = (selected_gradient_method == i);
                if (ImGui::Selectable(kGradientMethods[i].data(), is_selected))
                {
                    selected_gradient_method = i;
                    opt_info_.gradient_method = (i == 0) ? fdn_optimization::GradientMethod::CentralDifferences
                                                         : fdn_optimization::GradientMethod::ForwardDifferences;
                }
            }
            ImGui::EndCombo();
        }

        ImGui::InputScalar("Gradient Step Size", ImGuiDataType_Double, &opt_info_.gradient_delta, nullptr, nullptr,
                           "%.1e");
        ImGui::PopItemWidth();
    }

    bool start_optimization = ImGui::Button("Run Optimization");

    if (start_optimization)
    {
        if (fdn_optimizer_.GetStatus() != fdn_optimization::OptimizationStatus::Running)
        {
            opt_info_.initial_fdn_config = fdn_config;
            opt_info_.ir_size = Settings::Instance().SampleRate();
            if (has_target_rir)
            {
                opt_info_.target_rir.clear();
                opt_info_.target_rir.reserve(target_rir.size());
                std::ranges::copy(target_rir, std::back_inserter(opt_info_.target_rir));
            }

            fdn_optimizer_.StartOptimization(opt_info_);
            ImGui::OpenPopup("Optimization Progress");
        }
    }

    ImGui::SetNextWindowSize(ImVec2(600, -1), ImGuiCond_Always);
    if (ImGui::BeginPopupModal("Optimization Progress", nullptr, ImGuiWindowFlags_None))
    {
        fdn_optimization::OptimizationProgressInfo progress = fdn_optimizer_.GetProgress();

        std::chrono::duration<double, std::chrono::seconds::period> elapsed_seconds = progress.elapsed_time;
        ImGui::Text("Elapsed Time: %.2f seconds", elapsed_seconds.count());

        uint32_t eval_count = progress.evaluation_count;
        ImGui::Text("Evaluations: %u", eval_count);

        if (!progress.loss_history.empty() && ImPlot::BeginPlot("Loss Progress", ImVec2(-1, 250), ImPlotAxisFlags_None))
        {
            const std::vector<double>& loss_history_vec = progress.loss_history[0];
            ImPlot::SetupAxes("Evaluation", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(loss_history_vec.size() * 1.25),
                                    ImPlotCond_Always);

            double max_loss =
                loss_history_vec.empty() ? 1.0 : *std::max_element(loss_history_vec.begin(), loss_history_vec.end());
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

        if (!result.loss_history.empty())
        {
            initial_loss = result.loss_history[0].front();
            final_loss = result.loss_history[0].back();
            loss_history = result.loss_history[0];
            all_loss_histories = result.loss_history;
            loss_names = result.loss_names;
        }

        fdn_config = result.optimized_fdn_config;

        if (optimize_filters_checkbox_)
        {
            // fdn_config.attenuation_t60s = result.optimized_fdn_config.attenuation_t60s;
            LOG_INFO(logger_, "Optimized T60s: {}", fdn_config.attenuation_t60s);

            // fdn_config.tc_gains = result.optimized_fdn_config.tc_gains;
            // fdn_config.tc_frequencies = result.optimized_fdn_config.tc_frequencies;
            LOG_INFO(logger_, "Optimized Tone Correction Gains: {}", fdn_config.tc_gains);
        }

        updated_fdn = true;
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
        ImPlot::SetupAxes("Evaluation", "Loss", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
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

    return updated_fdn;
}
