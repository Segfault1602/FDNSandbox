#include "optimization_gui.h"

#include <imgui.h>
#include <implot.h>
#include <quill/LogMacros.h>
#include <quill/std/Array.h>
#include <quill/std/Vector.h>

#include <sffdn/sffdn.h>

#include "icons.h"
#include "optimizer.h"
#include "plot_ui.h"
#include "settings.h"
#include "theme.h"
#include "utils.h"
#include "views/window_ids.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

namespace
{
constexpr double kMaxWeight = 2.0;
constexpr double kMinWeight = 0.0;

bool IsOptimizationActive(fdn_optimization::OptimizationStatus status)
{
    return status == fdn_optimization::OptimizationStatus::StartRequested ||
           status == fdn_optimization::OptimizationStatus::Running ||
           status == fdn_optimization::OptimizationStatus::CancelRequested;
}

fdn_optimization::OptimizationAlgoParams DrawOptimizationParamGui(fdn_optimization::OptimizationAlgoType algo_type)
{
    ImGui::PushItemWidth(200);
    if (algo_type == fdn_optimization::OptimizationAlgoType::Adam)
    {
        static fdn_optimization::AdamParameters params;

        ImGui::InputFloat("Step Size", &params.step_size, 0.01f, 2.0f, "%.3f");
        ImGui::InputFloat("Beta 1", &params.beta1, 0.001f, 0.999f, "%.3f");
        ImGui::InputFloat("Beta 2", &params.beta2, 0.001f, 0.999f, "%.3f");
        ImGui::InputDouble("Gradient Delta", &params.gradient_delta, 1e-6f, 1e-1f, "%.1e",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::InputFloat("Tolerance", &params.tolerance, 1e-6f, 1e-3f, "%.1e", ImGuiSliderFlags_Logarithmic);

        params.decay_step_size = std::max(1, params.decay_step_size);
        params.epoch_restarts = std::max(0, params.epoch_restarts);
        params.max_restarts = std::max(0, params.max_restarts);

        return params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::SPSA)
    {
        static fdn_optimization::SPSAParameters spsa_params;

        ImGui::InputDouble("Alpha", &spsa_params.alpha, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputDouble("Gamma", &spsa_params.gamma, 0.05f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputDouble("Step Size", &spsa_params.step_size, 0.1f, 2.0f, "%.3f");
        ImGui::InputDouble("Evaluation Step Size", &spsa_params.evaluationStepSize, 0.1f, 2.0f, "%.3f");
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &spsa_params.max_iterations);

        return spsa_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::SimulatedAnnealing)
    {
        static fdn_optimization::SimulatedAnnealingParameters sa_params;
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &sa_params.max_iterations);

        constexpr std::array kInitialTempMinMax = {100.0, 20000.0};
        ImGui::InputScalar("Initial Temperature", ImGuiDataType_Double, &sa_params.initial_temperature,
                           kInitialTempMinMax.data(), &kInitialTempMinMax[1], "%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputScalar("Initial Moves", ImGuiDataType_U64, &sa_params.init_moves);
        ImGui::InputScalar("Move Control Sweep", ImGuiDataType_U64, &sa_params.move_ctrl_sweep);
        ImGui::InputScalar("Max Tolerance Sweep", ImGuiDataType_U64, &sa_params.max_tolerance_sweep);

        constexpr std::array kMaxMoveCoefMinMax = {1.0, 50.0};
        ImGui::InputScalar("Max Move Coefficient", ImGuiDataType_Double, &sa_params.max_move_coef,
                           kMaxMoveCoefMinMax.data(), &kMaxMoveCoefMinMax[1], "%.2f");

        constexpr std::array kInitMoveCoefMinMax = {0.1, 10.0};
        ImGui::InputScalar("Initial Move Coefficient", ImGuiDataType_Double, &sa_params.init_move_coef,
                           kInitMoveCoefMinMax.data(), &kInitMoveCoefMinMax[1], "%.2f");
        constexpr std::array kGainMinMax = {0.1, 1.0};
        ImGui::InputScalar("Gain", ImGuiDataType_Double, &sa_params.gain, kGainMinMax.data(), &kGainMinMax[1], "%.2f");

        return sa_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::DifferentialEvolution)
    {
        static fdn_optimization::DifferentialEvolutionParameters de_params;
        ImGui::InputScalar("Population Size", ImGuiDataType_U64, &de_params.population_size);
        ImGui::InputScalar("Max Generations", ImGuiDataType_U64, &de_params.max_generation);
        constexpr std::array kCrossoverRateMinMax = {0.0, 1.0};
        ImGui::InputScalar("Crossover Rate", ImGuiDataType_Double, &de_params.crossover_rate,
                           kCrossoverRateMinMax.data(), &kCrossoverRateMinMax[1], "%.2f");
        constexpr std::array kDifferentialWeightMinMax = {0.0, 2.0};
        ImGui::InputScalar("Differential Weight", ImGuiDataType_Double, &de_params.differential_weight,
                           kDifferentialWeightMinMax.data(), &kDifferentialWeightMinMax[1], "%.2f");

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
                           kExploitationFactorMinMax.data(), &kExploitationFactorMinMax[1], "%.2f");
        constexpr std::array kExplorationFactorMinMax = {0.1, 5.0};
        ImGui::InputScalar("Exploration Factor", ImGuiDataType_Double, &pso_params.exploration_factor,
                           kExplorationFactorMinMax.data(), &kExplorationFactorMinMax[1], "%.2f");

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
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &lbfgs_params.max_iterations,
                           kMaxIterationsSteps.data(), &kMaxIterationsSteps[1]);

        ImGui::InputScalar("Wolfe Parameter", ImGuiDataType_Double, &lbfgs_params.wolfe, nullptr, nullptr, "%.2f");
        ImGui::InputScalar("Min Gradient Norm", ImGuiDataType_Double, &lbfgs_params.min_gradient_norm, nullptr, nullptr,
                           "%.1e");
        ImGui::InputScalar("Factor", ImGuiDataType_Double, &lbfgs_params.factor, nullptr, nullptr, "%.1e");
        ImGui::InputScalar("Max Line Search Trials", ImGuiDataType_U64, &lbfgs_params.max_line_search_trials);
        ImGui::InputScalar("Min Step", ImGuiDataType_Double, &lbfgs_params.min_step, nullptr, nullptr, "%.1e");
        ImGui::InputScalar("Max Step", ImGuiDataType_Double, &lbfgs_params.max_step, nullptr, nullptr, "%.1e");
        ImGui::InputDouble("Gradient Delta", &lbfgs_params.gradient_delta, 1e-6f, 1e-1f, "%.1e",
                           ImGuiSliderFlags_Logarithmic);

        return lbfgs_params;
    }
    else if (algo_type == fdn_optimization::OptimizationAlgoType::GradientDescent)
    {
        static fdn_optimization::GradientDescentParameters gd_params;

        ImGui::InputDouble("Step Size", &gd_params.step_size, 0.001f, 1.0f, "%.4f");
        ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &gd_params.max_iterations, nullptr, nullptr, "%llu");
        ImGui::InputDouble("Tolerance", &gd_params.tolerance, 0.f, 0.f, "%.1ef");
        ImGui::InputDouble("Kappa", &gd_params.kappa, 0.f, 0.f, "%.2f");
        ImGui::InputDouble("Phi", &gd_params.phi, 0.f, 0.f, "%.2f");
        ImGui::InputDouble("Momentum", &gd_params.momentum, 0.f, 0.f, "%.3f");
        ImGui::InputDouble("Min Gain", &gd_params.min_gain, 1e-4f, 1.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::InputDouble("Gradient Delta", &gd_params.gradient_delta, 1e-6f, 1e-1f, "%.1e",
                           ImGuiSliderFlags_Logarithmic);

        return gd_params;
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
    , previous_loss_plot_statistics_{.min_loss = std::numeric_limits<double>::quiet_NaN(),
                                     .max_loss = std::numeric_limits<double>::quiet_NaN()}
{
}

bool OptimizationGUI::IsActive() const
{
    return IsOptimizationActive(fdn_optimizer_.GetStatus());
}

std::optional<OptimizationGUI::Event> OptimizationGUI::ConsumeEvent()
{
    std::optional<Event> event = std::move(pending_event_);
    pending_event_.reset();
    return event;
}

bool OptimizationGUI::Draw(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
{
    const float content_region_width = ImGui::GetContentRegionAvail().x;

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::BeginChild(fdn_sandbox::views::ids::kOptimizationParametersChild, ImVec2(content_region_width * 0.25f, -1),
                      ImGuiChildFlags_Borders, ImGuiWindowFlags_None);
    DrawSetupPanel(target_rir);
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild(fdn_sandbox::views::ids::kOptimizationResultsChild, ImVec2(content_region_width * 0.25f, -1),
                      ImGuiChildFlags_Borders, ImGuiWindowFlags_None);
    const bool updated_fdn = DrawOptimizationPanel(fdn_config, target_rir);
    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild(fdn_sandbox::views::ids::kOptimizationLossPlotChild, ImVec2(-1, -1), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_None);
    PlotLossHistory();
    ImGui::EndChild();

    ImGui::PopStyleVar();

    return updated_fdn;
}

void OptimizationGUI::DrawSetupPanel(std::span<const float> target_rir)
{
    ImGui::SeparatorText("Setup");

    opt_info_.parameters_to_optimize.clear();

    ImGui::RadioButton("Colorless Optimization", &optimize_type_, 0);
    ImGui::SameLine();
    ImGui::BeginDisabled(target_rir.empty());
    ImGui::RadioButton("RIR Match Optimization", &optimize_type_, 1);
    ImGui::EndDisabled();

    ImGui::Separator();

    if (optimize_type_ == 0)
    {
        DrawColorlessSettings();
    }
    else if (optimize_type_ == 1)
    {
        ImGui::BeginDisabled(target_rir.empty());
        DrawRIRMatchSettings();
        ImGui::EndDisabled();
    }
    else if (optimize_filters_checkbox_)
    {
    }
}

bool OptimizationGUI::DrawOptimizationPanel(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
{
    ImGui::SeparatorText("Optimization Parameters");
    DrawAlgorithmSettings();

    if (ImGui::Button(fdn_sandbox::icons::RunOptimization) && !IsOptimizationActive(fdn_optimizer_.GetStatus()))
    {
        StartOptimization(fdn_config, target_rir);
    }

    DrawOptimizationProgressPopup();

    ImGui::SeparatorText("Results");
    const bool updated_fdn = HandleOptimizationResult(fdn_config);
    DrawResultSummary();
    return updated_fdn;
}

void OptimizationGUI::DrawAlgorithmSettings()
{
    DrawAlgorithmSelector();
    opt_info_.optimizer_params = DrawOptimizationParamGui(selected_algorithm_);

    if (selected_algorithm_ >= fdn_optimization::OptimizationAlgoType::Adam)
    {
        DrawGradientSettings();
    }
}

void OptimizationGUI::DrawAlgorithmSelector()
{
    if (!ImGui::BeginCombo("Algorithm", fdn_optimization::OptimizationAlgoTypeToString(selected_algorithm_)))
    {
        return;
    }

    for (int i = 0; i < static_cast<int>(fdn_optimization::OptimizationAlgoType::Count); ++i)
    {
        const auto algorithm = static_cast<fdn_optimization::OptimizationAlgoType>(i);
        const bool is_selected = selected_algorithm_ == algorithm;
        if (ImGui::Selectable(fdn_optimization::OptimizationAlgoTypeToString(algorithm), is_selected))
        {
            selected_algorithm_ = algorithm;
        }
    }
    ImGui::EndCombo();
}

void OptimizationGUI::DrawGradientSettings()
{
    ImGui::PushItemWidth(200);
    ImGui::SeparatorText("Gradient Settings");
    constexpr std::array<const char*, 2> kGradientMethods = {"Central Difference", "Forward Difference"};
    if (ImGui::BeginCombo("Gradient Method", kGradientMethods[selected_gradient_method_]))
    {
        for (int i = 0; std::cmp_less(i, kGradientMethods.size()); ++i)
        {
            const bool is_selected = selected_gradient_method_ == i;
            if (!ImGui::Selectable(kGradientMethods[i], is_selected))
            {
                continue;
            }

            selected_gradient_method_ = i;
            opt_info_.gradient_method = i == 0 ? fdn_optimization::GradientMethod::CentralDifferences
                                               : fdn_optimization::GradientMethod::ForwardDifferences;
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
}

void OptimizationGUI::StartOptimization(const sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
{
    opt_info_.initial_fdn_config = fdn_config;
    utils::NormalizeAttenuationFilterBank(opt_info_.initial_fdn_config);
    opt_info_.ir_size = Settings::Instance().SampleRate();
    if (!target_rir.empty())
    {
        opt_info_.target_rir.assign(target_rir.begin(), target_rir.end());
    }

    LossFunctions loss_functions = CreateLossFunctions(target_rir);
    fdn_optimizer_.SetLossFunctions(loss_functions);
    fdn_optimizer_.StartOptimization(opt_info_);
    ImGui::OpenPopup(fdn_sandbox::views::ids::kOptimizationProgressModal);
}

OptimizationGUI::LossFunctions OptimizationGUI::CreateLossFunctions(std::span<const float> target_rir) const
{
    LossFunctions loss_functions;
    if (optimize_gains_checkbox_ || optimize_matrix_checkbox_)
    {
        AppendColorlessLossFunctions(loss_functions);
    }
    else if (optimize_filters_checkbox_)
    {
        AppendRIRMatchLossFunctions(loss_functions, target_rir);
    }
    return loss_functions;
}

void OptimizationGUI::AppendColorlessLossFunctions(LossFunctions& loss_functions) const
{
    if (spectral_flatness_weight_ > 0.0)
    {
        constexpr float kTargetSpectralFlatness = 0.5575f;
        loss_functions.push_back(std::make_shared<fdn_optimization::SpectralFlatnessLoss>(kTargetSpectralFlatness,
                                                                                          spectral_flatness_weight_));
    }

    if (sparsity_weight_ > 0.0)
    {
        loss_functions.push_back(std::make_shared<fdn_optimization::TimeDomainSparsityLoss>(sparsity_weight_));
    }
}

void OptimizationGUI::AppendRIRMatchLossFunctions(LossFunctions& loss_functions,
                                                  std::span<const float> target_rir) const
{
    if (edc_weight_ > 0.0)
    {
        loss_functions.push_back(std::make_shared<fdn_optimization::EnergyDecayCurveLoss>(target_rir, edc_weight_));
    }

    if (mel_edr_weight_ <= 0.0)
    {
        return;
    }

    const audio_utils::analysis::EnergyDecayReliefOptions edr_options{
        .fft_length = mel_edr_fft_length_,
        .hop_size = mel_edr_hop_size_,
        .window_size = mel_edr_window_size_,
        .window_type = audio_utils::FFTWindowType::Hann,
        .n_mels = mel_edr_num_bands_,
        .to_db = true,
    };
    loss_functions.push_back(
        std::make_shared<fdn_optimization::WeightedEDRLoss>(target_rir, edr_options, -20.0f, mel_edr_weight_));
}

void OptimizationGUI::DrawOptimizationProgressPopup()
{
    ImGui::SetNextWindowSize(ImVec2(600, -1), ImGuiCond_Always);
    if (!ImGui::BeginPopupModal(fdn_sandbox::views::ids::kOptimizationProgressModal, nullptr, ImGuiWindowFlags_None))
    {
        return;
    }

    const fdn_optimization::OptimizationProgressInfo progress = fdn_optimizer_.GetProgress();
    const std::chrono::duration<double, std::chrono::seconds::period> elapsed_seconds = progress.elapsed_time;
    ImGui::Text("Elapsed Time: %.2f seconds", elapsed_seconds.count());
    ImGui::Text("Evaluations: %u", progress.evaluation_count);
    DrawProgressLossPlot(progress);

    const auto status = fdn_optimizer_.GetStatus();
    const char* progress_label = "Optimizing...";
    if (status == fdn_optimization::OptimizationStatus::StartRequested)
    {
        progress_label = "Starting...";
    }
    else if (status == fdn_optimization::OptimizationStatus::CancelRequested)
    {
        progress_label = "Canceling...";
    }

    ImGui::ProgressBar(-1.0f * static_cast<float>(ImGui::GetTime()), ImVec2(-1, 0.0f), progress_label);
    ImGui::BeginDisabled(status == fdn_optimization::OptimizationStatus::CancelRequested);
    if (ImGui::Button(fdn_sandbox::icons::CancelOptimization))
    {
        fdn_optimizer_.CancelOptimization();
    }
    ImGui::EndDisabled();

    if (!IsOptimizationActive(fdn_optimizer_.GetStatus()))
    {
        ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
}

void OptimizationGUI::DrawProgressLossPlot(const fdn_optimization::OptimizationProgressInfo& progress)
{
    if (progress.loss_history.empty())
    {
        return;
    }

    ImPlot::SetNextAxisToFit(ImAxis_Y1);
    if (!ImPlot::BeginPlot("Loss Progress", ImVec2(-1, 250), ImPlotAxisFlags_None))
    {
        return;
    }

    const std::vector<double>& total_loss_history = progress.loss_history.front();
    ImPlot::SetupAxes("Evaluation", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, static_cast<double>(total_loss_history.size()) * 1.25, ImPlotCond_Always);
    ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    ImPlot::PlotLine("Total Loss", total_loss_history.data(), static_cast<int>(total_loss_history.size()), 1.0, 0.0,
                     fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Optimization, 1.8f));

    for (size_t i = 1; i < progress.loss_history.size(); ++i)
    {
        const std::vector<double>& component_history = progress.loss_history[i];
        ImPlotSpec component_spec{};
        component_spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor({.index = i - 1});
        component_spec.LineWeight = 1.1f;
        ImPlot::PlotLine(("Component " + std::to_string(i)).c_str(), component_history.data(),
                         static_cast<int>(component_history.size()), 1.0, 0.0, component_spec);
    }

    if (!total_loss_history.empty())
    {
        const double best_loss = *std::ranges::min_element(total_loss_history);
        const ImVec4 status_color = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusOk);
        fdn_sandbox::plot_ui::DrawHorizontalGuide("Best Loss", best_loss, status_color, 1.0f);
        ImPlot::Annotation(static_cast<double>(total_loss_history.size() - 1), best_loss, status_color,
                           ImVec2(6.0f, -6.0f), true, "Best %.4g", best_loss);
    }

    ImPlot::EndPlot();
}

bool OptimizationGUI::HandleOptimizationResult(sfFDN::FDNConfig& fdn_config)
{
    const auto status = fdn_optimizer_.GetStatus();
    if (status == fdn_optimization::OptimizationStatus::Failed)
    {
        pending_event_ =
            Event{.type = EventType::Failed, .message = "Optimization failed. Check the application log for details."};
        fdn_optimizer_.ResetStatus();
        return false;
    }
    if (status != fdn_optimization::OptimizationStatus::Completed &&
        status != fdn_optimization::OptimizationStatus::Canceled)
    {
        return false;
    }

    const fdn_optimization::OptimizationResult result = fdn_optimizer_.GetResult();
    result_summary_.elapsed_time_sec =
        std::chrono::duration<double, std::chrono::seconds::period>(result.total_time).count();
    result_summary_.evaluation_count = result.total_evaluations;

    if (!result.loss_history.empty())
    {
        result_summary_.initial_loss = result.loss_history.front().front();
        result_summary_.final_loss = result.loss_history.front().back();
        loss_history_.losses = result.loss_history;
        loss_history_.loss_names = result.loss_names;
        loss_history_.loss_names.insert(loss_history_.loss_names.begin(), "Total Loss");
    }

    fdn_config = result.optimized_fdn_config;
    pending_event_ =
        status == fdn_optimization::OptimizationStatus::Completed
            ? Event{.type = EventType::Completed,
                    .message = std::format("Completed in {:.2f} s after {} evaluations.",
                                           result_summary_.elapsed_time_sec, result_summary_.evaluation_count)}
            : Event{.type = EventType::Canceled, .message = "Optimization was canceled."};
    fdn_optimizer_.ResetStatus();
    return true;
}

void OptimizationGUI::DrawResultSummary() const
{
    ImGui::Text("Elapsed Time: %.2f seconds", result_summary_.elapsed_time_sec);
    ImGui::Text("Evaluations: %u", result_summary_.evaluation_count);
    ImGui::Text("Initial Loss: %.6f", result_summary_.initial_loss);
    ImGui::Text("Final Loss: %.6f", result_summary_.final_loss);
}

void OptimizationGUI::PlotLossHistory()
{
    const size_t loss_count = loss_history_.losses.size();
    assert(loss_count == loss_history_.loss_names.size());
    const LossPlotStatistics statistics = CalculateLossPlotStatistics();
    const bool history_bounds_changed = HaveLossPlotBoundsChanged(statistics);

    if (!ImPlot::BeginPlot("Loss Progress", ImVec2(-1, -1), ImPlotAxisFlags_None))
    {
        return;
    }

    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
    ImPlot::SetupAxes("Evaluation", "Loss", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
    ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);

    if (loss_count == 0)
    {
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 1.0, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
        fdn_sandbox::plot_ui::DrawEmptyState("No optimization history available");
    }
    else
    {
        DrawPopulatedLossPlot(statistics, history_bounds_changed);
    }

    ImPlot::EndPlot();
    previous_loss_plot_statistics_ = statistics;
}

OptimizationGUI::LossPlotStatistics OptimizationGUI::CalculateLossPlotStatistics() const
{
    LossPlotStatistics statistics{
        .min_loss = std::numeric_limits<double>::infinity(),
        .max_loss = -std::numeric_limits<double>::infinity(),
    };
    for (const auto& losses : loss_history_.losses)
    {
        statistics.total_point_count += losses.size();
        statistics.max_history_size = std::max(statistics.max_history_size, losses.size());
        for (const double loss : losses)
        {
            if (std::isfinite(loss))
            {
                statistics.min_loss = std::min(statistics.min_loss, loss);
                statistics.max_loss = std::max(statistics.max_loss, loss);
            }
        }
    }
    return statistics;
}

bool OptimizationGUI::HaveLossPlotBoundsChanged(const LossPlotStatistics& statistics) const
{
    return statistics.total_point_count != previous_loss_plot_statistics_.total_point_count ||
           statistics.max_history_size != previous_loss_plot_statistics_.max_history_size ||
           statistics.min_loss != previous_loss_plot_statistics_.min_loss ||
           statistics.max_loss != previous_loss_plot_statistics_.max_loss;
}

void OptimizationGUI::DrawPopulatedLossPlot(const LossPlotStatistics& statistics, bool history_bounds_changed) const
{
    const double x_max =
        statistics.max_history_size > 1 ? static_cast<double>(statistics.max_history_size - 1) * 1.05 : 1.0;
    ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, x_max, ImPlotCond_Always);

    if (history_bounds_changed && std::isfinite(statistics.min_loss) && std::isfinite(statistics.max_loss))
    {
        const double range = statistics.max_loss - statistics.min_loss;
        const double padding = std::max({range * 0.1, std::abs(statistics.max_loss) * 0.05, 1e-9});
        ImPlot::SetupAxisLimits(ImAxis_Y1, statistics.min_loss - padding, statistics.max_loss + padding,
                                ImPlotCond_Always);
    }

    DrawLossSeries();
    DrawBestLossGuide();
}

void OptimizationGUI::DrawLossSeries() const
{
    for (size_t i = 0; i < loss_history_.losses.size(); ++i)
    {
        ImPlotSpec spec{};
        if (i == 0)
        {
            spec = fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Optimization, 1.8f);
        }
        else
        {
            spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor({.index = i - 1});
            spec.LineWeight = 1.1f;
        }
        ImPlot::PlotLine(loss_history_.loss_names[i].c_str(), loss_history_.losses[i].data(),
                         static_cast<int>(loss_history_.losses[i].size()), 1.0, 0.0, spec);
    }
}

void OptimizationGUI::DrawBestLossGuide() const
{
    if (loss_history_.losses.front().empty())
    {
        return;
    }

    const double best_loss = *std::ranges::min_element(loss_history_.losses.front());
    fdn_sandbox::plot_ui::DrawHorizontalGuide("Best Loss", best_loss,
                                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusOk), 1.0f);
}

void OptimizationGUI::DrawColorlessSettings()
{
    ImGui::Checkbox("Optimize Gains", &optimize_gains_checkbox_);

    if (optimize_gains_checkbox_)
    {
        optimize_filters_checkbox_ = false;
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Gains);
    }

    ImGui::Checkbox("Optimize Matrix", &optimize_matrix_checkbox_);
    if (optimize_matrix_checkbox_)
    {
        ImGui::RadioButton("Random", &matrix_type_, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Householder", &matrix_type_, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Circulant", &matrix_type_, 2);

        if (matrix_type_ == 0)
        {
            opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix);
        }
        else if (matrix_type_ == 1)
        {
            opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix_Householder);
        }
        else if (matrix_type_ == 2)
        {
            opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::Matrix_Circulant);
        }
    }

    ImGui::SeparatorText("Objective Weights");

    ImGui::PushItemWidth(100);

    ImGui::SliderScalar("Spectral Flatness Weight", ImGuiDataType_Double, &spectral_flatness_weight_, &kMinWeight,
                        &kMaxWeight, "%.2f");
    ImGui::SliderScalar("Sparsity Weight", ImGuiDataType_Double, &sparsity_weight_, &kMinWeight, &kMaxWeight, "%.2f");

    ImGui::PopItemWidth();
}

void OptimizationGUI::DrawRIRMatchSettings()
{
    optimize_gains_checkbox_ = false;
    optimize_matrix_checkbox_ = false;
    optimize_filters_checkbox_ = true;
    opt_info_.parameters_to_optimize.clear();

    ImGui::RadioButton("10 band", &filter_type_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("3 band", &filter_type_, 1);

    if (filter_type_ == 0)
    {
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::AttenuationFilters);
    }
    else if (filter_type_ == 1)
    {
        opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::AttenuationFilters_3Band);
    }

    opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::TonecorrectionFilters);
    opt_info_.parameters_to_optimize.push_back(fdn_optimization::OptimizationParamType::OverallGain);

    ImGui::SeparatorText("RIR Match Weights");

    ImGui::PushItemWidth(100);

    ImGui::SliderScalar("EDC Weight", ImGuiDataType_Double, &edc_weight_, &kMinWeight, &kMaxWeight, "%.2f");
    ImGui::SliderScalar("Mel EDR Weight", ImGuiDataType_Double, &mel_edr_weight_, &kMinWeight, &kMaxWeight, "%.2f");
    constexpr std::array kFFTSizeOptions = {"512", "1024", "2048", "4096", "8192"};
    if (ImGui::BeginCombo("Mel EDR FFT Size", kFFTSizeOptions[selected_fft_size_index_]))
    {
        for (int i = 0; std::cmp_less(i, kFFTSizeOptions.size()); ++i)
        {
            const bool is_selected = selected_fft_size_index_ == i;
            if (ImGui::Selectable(kFFTSizeOptions[i], is_selected))
            {
                selected_fft_size_index_ = i;
                mel_edr_fft_length_ = 512u << i;
            }
        }
        ImGui::EndCombo();
    }

    ImGui::InputScalar("Mel EDR Hop Size", ImGuiDataType_U32, &mel_edr_hop_size_);
    ImGui::InputScalar("Mel EDR Window Size", ImGuiDataType_U32, &mel_edr_window_size_);
    ImGui::InputScalar("Mel EDR Num Bands", ImGuiDataType_U32, &mel_edr_num_bands_);

    mel_edr_window_size_ = std::clamp(mel_edr_window_size_, 256u, mel_edr_fft_length_);
    mel_edr_hop_size_ = std::clamp(mel_edr_hop_size_, 32u, mel_edr_window_size_ - 1);

    ImGui::PopItemWidth();
}