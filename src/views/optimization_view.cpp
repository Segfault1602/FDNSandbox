#include "views/optimization_view.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <format>
#include <imgui.h>
#include <implot.h>
#include <memory>
#include <optional>
#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include "audio_loss.h"
#include "audio_utils/audio_analysis.h"
#include "audio_utils/fft_utils.h"
#include "icons.h"
#include "initialization.h"
#include "optim_types.h"
#include "optimizer.h"
#include "plot_ui.h"
#include "settings.h"
#include "sffdn/fdn_config.h"
#include "theme.h"
#include "utils.h"
#include "views/window_ids.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

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

/// Restores the ImGui item-width stack on every exit path.
class ItemWidthScope
{
  public:
    explicit ItemWidthScope(float width)
    {
        ImGui::PushItemWidth(width);
    }
    ~ItemWidthScope()
    {
        ImGui::PopItemWidth();
    }

    ItemWidthScope(const ItemWidthScope&) = delete;
    ItemWidthScope& operator=(const ItemWidthScope&) = delete;
    ItemWidthScope(ItemWidthScope&&) = delete;
    ItemWidthScope& operator=(ItemWidthScope&&) = delete;
};

/// Draws a combo bound to an enum, using `to_string` for the labels.
template <typename Enum, std::size_t N>
void DrawEnumCombo(const char* label, Enum& value, const std::array<Enum, N>& options, const char* (*to_string)(Enum))
{
    if (!ImGui::BeginCombo(label, to_string(value)))
    {
        return;
    }
    for (const Enum option : options)
    {
        const bool is_selected = value == option;
        if (ImGui::Selectable(to_string(option), is_selected))
        {
            value = option;
        }
    }
    ImGui::EndCombo();
}

void DrawAdamParams(fdn_optimization::AdamParameters& params)
{
    ImGui::InputFloat("Step Size", &params.step_size, 0.01f, 2.0f, "%.4f");
    ImGui::InputFloat("Beta 1", &params.beta1, 0.001f, 0.999f, "%.3f");
    ImGui::InputFloat("Beta 2", &params.beta2, 0.001f, 0.999f, "%.3f");
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Gradient Delta", &params.gradient_delta, 1e-6f, 1e-1f, "%.1e", ImGuiSliderFlags_Logarithmic);
    ImGui::InputFloat("Tolerance", &params.tolerance, 1e-6f, 1e-3f, "%.1e", ImGuiSliderFlags_Logarithmic);

    params.decay_step_size = std::max(1, params.decay_step_size);
    params.epoch_restarts = std::max(0, params.epoch_restarts);
    params.max_restarts = std::max(0, params.max_restarts);
}

void DrawSPSAParams(fdn_optimization::SPSAParameters& params)
{
    ImGui::InputDouble("Alpha", &params.alpha, 0.001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::InputDouble("Gamma", &params.gamma, 0.05f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::InputDouble("Step Size", &params.step_size, 0.1f, 2.0f, "%.4f");
    ImGui::InputDouble("Evaluation Step Size", &params.evaluationStepSize, 0.1f, 2.0f, "%.4f");
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
}

void DrawBlockSPSAParams(fdn_optimization::BlockSPSAParameters& params)
{
    constexpr std::array kModes = {fdn_optimization::BlockSPSAMode::SnapshotSweepAll,
                                   fdn_optimization::BlockSPSAMode::RandomOne};
    DrawEnumCombo("Mode", params.mode, kModes, fdn_optimization::BlockSPSAModeToString);

    constexpr std::array kStrategies = {fdn_optimization::ParameterBlockStrategy::Semantic,
                                        fdn_optimization::ParameterBlockStrategy::FixedContiguous};
    DrawEnumCombo("Block Strategy", params.block_strategy, kStrategies,
                  fdn_optimization::ParameterBlockStrategyToString);

    if (params.block_strategy == fdn_optimization::ParameterBlockStrategy::FixedContiguous)
    {
        ImGui::InputScalar("Fixed Block Size", ImGuiDataType_U64, &params.contiguous_block_size);
    }

    if (params.mode == fdn_optimization::BlockSPSAMode::RandomOne)
    {
        constexpr std::array kSchedules = {fdn_optimization::RandomBlockSchedule::ShuffledSweep,
                                           fdn_optimization::RandomBlockSchedule::IndependentUniform};
        DrawEnumCombo("Random Schedule", params.random_schedule, kSchedules,
                      fdn_optimization::RandomBlockScheduleToString);
    }

    constexpr std::array kNormalizations = {fdn_optimization::ProbeRadiusNormalization::None,
                                            fdn_optimization::ProbeRadiusNormalization::SqrtDimension};
    DrawEnumCombo("Probe Radius Normalization", params.probe_radius_normalization, kNormalizations,
                  fdn_optimization::ProbeRadiusNormalizationToString);

    ImGui::InputDouble("Step Size", &params.step_size, 0.1f, 2.0f, "%.4f");
    ImGui::InputDouble("Evaluation Step Size", &params.evaluation_step_size, 0.01f, 0.5f, "%.4f");
    ImGui::InputDouble("Alpha", &params.alpha, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Gamma", &params.gamma, 0.01f, 0.1f, "%.4f");

    // An unset stability constant lets the optimizer derive one; expose that as an explicit toggle
    // rather than overloading a sentinel value.
    bool has_stability_constant = params.stability_constant.has_value();
    if (ImGui::Checkbox("Override Stability Constant", &has_stability_constant))
    {
        params.stability_constant = has_stability_constant ? std::optional<double>(1.0) : std::nullopt;
    }
    if (params.stability_constant)
    {
        ImGui::InputDouble("Stability Constant", &*params.stability_constant, 1.0f, 10.0f, "%.4f");
    }

    ImGui::InputScalar("Directions Per Block", ImGuiDataType_U64, &params.directions_per_block);
    params.directions_per_block = std::max<std::size_t>(1, params.directions_per_block);

    ImGui::InputScalar("Accepted Eval Interval", ImGuiDataType_U64, &params.accepted_evaluation_interval);
    params.accepted_evaluation_interval = std::max<std::size_t>(1, params.accepted_evaluation_interval);

    ImGui::InputDouble("Max Step Norm", &params.max_step_norm, 0.1f, 1.0f, "%.4f");
    params.max_step_norm = std::max(0.0, params.max_step_norm);

    bool has_stall_window = params.stall_window.has_value();
    if (ImGui::Checkbox("Enable Stall Window", &has_stall_window))
    {
        params.stall_window = has_stall_window ? std::optional<std::size_t>(320) : std::nullopt;
    }
    if (params.stall_window)
    {
        ImGui::InputScalar("Stall Window", ImGuiDataType_U64, &*params.stall_window);
    }

    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);

    // Block gain scales are carried through from the tuned defaults. They are a list of per-class
    // (a, c) multipliers whose valid classes depend on the block strategy, so editing them here
    // would need validation the optimizer already performs on the CLI side.
    if (!params.block_scales.empty())
    {
        ImGui::TextDisabled("%zu tuned block gain scale(s) applied", params.block_scales.size());
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip("Per-class learning-rate and perturbation multipliers from the tuned configuration.");
        }
    }
}

void DrawSimulatedAnnealingParams(fdn_optimization::SimulatedAnnealingParameters& params)
{
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Initial Temperature", &params.initial_temperature, 0.01f, 1.0f, "%.4f");
    ImGui::InputScalar("Initial Moves", ImGuiDataType_U64, &params.init_moves);
    ImGui::InputScalar("Move Control Sweep", ImGuiDataType_U64, &params.move_ctrl_sweep);
    ImGui::InputScalar("Max Tolerance Sweep", ImGuiDataType_U64, &params.max_tolerance_sweep);
    ImGui::InputDouble("Max Move Coefficient", &params.max_move_coef, 1.0f, 10.0f, "%.4f");
    ImGui::InputDouble("Initial Move Coefficient", &params.init_move_coef, 0.01f, 1.0f, "%.4f");
    ImGui::InputDouble("Gain", &params.gain, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
}

void DrawCNEParams(fdn_optimization::CNEParameters& params)
{
    ImGui::InputScalar("Population Size", ImGuiDataType_U64, &params.population_size);
    ImGui::InputScalar("Max Generations", ImGuiDataType_U64, &params.max_generations);
    ImGui::InputDouble("Mutation Probability", &params.mutation_probability, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Mutation Size", &params.mutation_size, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Select Percent", &params.select_percent, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
}

void DrawDifferentialEvolutionParams(fdn_optimization::DifferentialEvolutionParameters& params)
{
    ImGui::InputScalar("Population Size", ImGuiDataType_U64, &params.population_size);
    ImGui::InputScalar("Max Generations", ImGuiDataType_U64, &params.max_generation);
    ImGui::InputDouble("Crossover Rate", &params.crossover_rate, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Differential Weight", &params.differential_weight, 0.01f, 0.1f, "%.4f");
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
}

void DrawPSOParams(fdn_optimization::PSOParameters& params)
{
    ImGui::InputScalar("Number of Particles", ImGuiDataType_U64, &params.num_particles);
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputScalar("Horizon Size", ImGuiDataType_U64, &params.horizon_size);
    ImGui::InputDouble("Exploitation Factor", &params.exploitation_factor, 0.1f, 1.0f, "%.4f");
    ImGui::InputDouble("Exploration Factor", &params.exploration_factor, 0.1f, 1.0f, "%.4f");
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
}

void DrawLBFGSParams(fdn_optimization::L_BFGSParameters& params)
{
    ImGui::InputScalar("Number of Basis", ImGuiDataType_U64, &params.num_basis);
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Wolfe Parameter", &params.wolfe, 0.0, 0.0, "%.4f");
    ImGui::InputDouble("Min Gradient Norm", &params.min_gradient_norm, 0.0, 0.0, "%.1e");
    ImGui::InputDouble("Factor", &params.factor, 0.0, 0.0, "%.1e");
    ImGui::InputScalar("Max Line Search Trials", ImGuiDataType_U64, &params.max_line_search_trials);
    ImGui::InputDouble("Min Step", &params.min_step, 0.0, 0.0, "%.1e");
    ImGui::InputDouble("Max Step", &params.max_step, 0.0, 0.0, "%.1e");
    ImGui::InputDouble("Gradient Delta", &params.gradient_delta, 1e-6f, 1e-1f, "%.1e", ImGuiSliderFlags_Logarithmic);
}

void DrawGradientDescentParams(fdn_optimization::GradientDescentParameters& params)
{
    ImGui::InputDouble("Step Size", &params.step_size, 0.001f, 1.0f, "%.4f");
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
    ImGui::InputDouble("Kappa", &params.kappa, 0.0, 0.0, "%.4f");
    ImGui::InputDouble("Phi", &params.phi, 0.0, 0.0, "%.4f");
    ImGui::InputDouble("Momentum", &params.momentum, 0.0, 0.0, "%.4f");
    ImGui::InputDouble("Min Gain", &params.min_gain, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
    ImGui::InputDouble("Gradient Delta", &params.gradient_delta, 1e-6f, 1e-1f, "%.1e", ImGuiSliderFlags_Logarithmic);
    ImGui::InputDouble("Max Step Norm", &params.max_step_norm, 0.1f, 1.0f, "%.4f");
    params.max_step_norm = std::max(0.0, params.max_step_norm);
}

void DrawCMAESParams(fdn_optimization::CMAESParameters& params)
{
    ImGui::InputScalar("Population Size", ImGuiDataType_U64, &params.population_size);
    ImGui::InputScalar("Max Iterations", ImGuiDataType_U64, &params.max_iterations);
    ImGui::InputDouble("Tolerance", &params.tolerance, 0.0, 0.0, "%.1e", ImGuiSliderFlags_Logarithmic);
    ImGui::InputDouble("Step Size", &params.step_size, 0.01f, 0.1f, "%.4f");
}

void DrawRandomSearchParams(fdn_optimization::RandomSearchParameters& params)
{
    ImGui::InputDouble("Time Limit (s)", &params.time_limit_seconds, 1.0f, 10.0f, "%.1f");
}

} // namespace

namespace fdn_sandbox::views
{

std::size_t OptimizationView::AlgorithmParameterStore::Index(fdn_optimization::OptimizationObjective objective)
{
    return objective == fdn_optimization::OptimizationObjective::RirMatching ? 1U : 0U;
}

fdn_optimization::OptimizationAlgoParams& OptimizationView::AlgorithmParameterStore::Entry(
    fdn_optimization::OptimizationObjective objective, fdn_optimization::OptimizationAlgoType algorithm)
{
    const auto algorithm_index = static_cast<std::size_t>(algorithm);
    assert(algorithm_index < kAlgorithmCount);
    Slot& slot = slots_[Index(objective)][algorithm_index];
    if (!slot.seeded)
    {
        slot.params = fdn_optimization::TunedDefaultsFor(algorithm, objective);
        slot.seeded = true;
    }
    return slot.params;
}

void OptimizationView::AlgorithmParameterStore::Reset(fdn_optimization::OptimizationObjective objective,
                                                      fdn_optimization::OptimizationAlgoType algorithm)
{
    const auto algorithm_index = static_cast<std::size_t>(algorithm);
    assert(algorithm_index < kAlgorithmCount);
    Slot& slot = slots_[Index(objective)][algorithm_index];
    slot.params = fdn_optimization::TunedDefaultsFor(algorithm, objective);
    slot.seeded = true;
}

OptimizationView::OptimizationView(quill::Logger* logger)
    : fdn_optimizer_(logger)
    , previous_loss_plot_statistics_{.min_loss = std::numeric_limits<double>::quiet_NaN(),
                                     .max_loss = std::numeric_limits<double>::quiet_NaN()}
{
    ApplyTunedLossDefaults(SelectedObjective());

    // The tuned RIR-matching campaign is thesis-faithful 3-band matching, so the filter selector
    // must start there rather than on the 10-band default it previously shipped with.
    const auto matching = fdn_optimization::TunedMatchingDefaults();
    filter_type_ =
        matching.attenuation_filter_type == fdn_optimization::OptimizationParamType::AttenuationFilters_3Band ? 1 : 0;
    tuned_matching_initialization_ = matching.initialization;
    opt_info_.matching_parameters.initialization = matching.initialization;
}

fdn_optimization::OptimizationObjective OptimizationView::SelectedObjective() const noexcept
{
    return optimize_type_ == 1 ? fdn_optimization::OptimizationObjective::RirMatching
                               : fdn_optimization::OptimizationObjective::Colorless;
}

void OptimizationView::ApplyTunedLossDefaults(fdn_optimization::OptimizationObjective objective)
{
    // Each objective edits a disjoint pair of weights (colorless: flatness and sparsity; matching:
    // EDC and mel-EDR), so seeding all four once is safe and switching objectives never has to
    // discard the other objective's edits.
    const auto losses = fdn_optimization::TunedLossDefaultsFor(objective);
    spectral_flatness_weight_ = losses.spectral_flatness_weight;
    sparsity_weight_ = losses.sparsity_weight;
    edc_weight_ = losses.edc_weight;
    mel_edr_weight_ = losses.mel_edr_weight;
}

bool OptimizationView::IsActive() const noexcept
{
    return IsOptimizationActive(fdn_optimizer_.GetStatus());
}

std::optional<OptimizationView::Event> OptimizationView::ConsumeEvent()
{
    std::optional<Event> event = std::move(pending_event_);
    pending_event_.reset();
    return event;
}

bool OptimizationView::Draw(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
{
    if (!ImGui::Begin(ids::kOptimization))
    {
        ImGui::End();
        return false;
    }

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

    ImGui::End();
    return updated_fdn;
}

void OptimizationView::DrawSetupPanel(std::span<const float> target_rir)
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

bool OptimizationView::DrawOptimizationPanel(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
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

void OptimizationView::DrawParameterEditor(fdn_optimization::OptimizationAlgoParams& params)
{
    const ItemWidthScope item_width(200);

    // Dispatching on the stored alternative rather than on `selected_algorithm_` keeps the editor
    // and the parameters that get submitted in agreement even if the two ever drift apart.
    std::visit(
        [](auto& value) {
            using Params = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Params, fdn_optimization::AdamParameters>)
            {
                DrawAdamParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::SPSAParameters>)
            {
                DrawSPSAParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::BlockSPSAParameters>)
            {
                DrawBlockSPSAParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::SimulatedAnnealingParameters>)
            {
                DrawSimulatedAnnealingParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::CNEParameters>)
            {
                DrawCNEParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::DifferentialEvolutionParameters>)
            {
                DrawDifferentialEvolutionParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::PSOParameters>)
            {
                DrawPSOParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::L_BFGSParameters>)
            {
                DrawLBFGSParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::GradientDescentParameters>)
            {
                DrawGradientDescentParams(value);
            }
            else if constexpr (std::is_same_v<Params, fdn_optimization::CMAESParameters>)
            {
                DrawCMAESParams(value);
            }
            else
            {
                DrawRandomSearchParams(value);
            }
        },
        params);
}

void OptimizationView::DrawAlgorithmSettings()
{
    DrawAlgorithmSelector();

    const fdn_optimization::OptimizationObjective objective = SelectedObjective();
    fdn_optimization::OptimizationAlgoParams& params = parameter_store_.Entry(objective, selected_algorithm_);
    DrawParameterEditor(params);

    if (ImGui::Button("Reset to tuned defaults"))
    {
        parameter_store_.Reset(objective, selected_algorithm_);
        ApplyTunedLossDefaults(objective);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Restores the %s hyperparameters tuned for the %s objective.",
                          fdn_optimization::OptimizationAlgoTypeToString(selected_algorithm_),
                          fdn_optimization::OptimizationObjectiveToString(objective));
    }

    opt_info_.optimizer_params = params;

    if (selected_algorithm_ >= fdn_optimization::OptimizationAlgoType::Adam)
    {
        DrawGradientSettings();
    }
}

void OptimizationView::DrawAlgorithmSelector()
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

void OptimizationView::DrawGradientSettings()
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

void OptimizationView::StartOptimization(const sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir)
{
    opt_info_.initial_fdn_config = fdn_config;
    utils::NormalizeAttenuationFilterBank(opt_info_.initial_fdn_config);
    opt_info_.ir_size = Settings::Instance().SampleRate();
    if (!target_rir.empty())
    {
        opt_info_.target_rir.assign(target_rir.begin(), target_rir.end());
    }

    // Target-derived initialization is silently ignored unless the estimates are supplied, so they
    // are measured here where both the target RIR and the selected band layout are known. Measuring
    // per start rather than per frame keeps the analysis off the UI thread's hot path.
    opt_info_.t60_estimates.clear();
    if (opt_info_.matching_parameters.initialization == fdn_optimization::MatchingInitialization::TargetDerived &&
        !target_rir.empty())
    {
        const auto attenuation_type = filter_type_ == 1
                                          ? fdn_optimization::OptimizationParamType::AttenuationFilters_3Band
                                          : fdn_optimization::OptimizationParamType::AttenuationFilters;
        opt_info_.t60_estimates = fdn_optimization::EstimateMatchingT60s(target_rir, attenuation_type,
                                                                         Settings::Instance().SampleRateAs<float>());
        LOG_INFO(Settings::Instance().GetLogger(), "Estimated {} starting T60s from the target RIR",
                 opt_info_.t60_estimates.size());
    }

    LossFunctions loss_functions = CreateLossFunctions(target_rir);
    fdn_optimizer_.SetLossFunctions(loss_functions);
    fdn_optimizer_.StartOptimization(opt_info_);
    ImGui::OpenPopup(fdn_sandbox::views::ids::kOptimizationProgressModal);
}

OptimizationView::LossFunctions OptimizationView::CreateLossFunctions(std::span<const float> target_rir) const
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

void OptimizationView::AppendColorlessLossFunctions(LossFunctions& loss_functions) const
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

void OptimizationView::AppendRIRMatchLossFunctions(LossFunctions& loss_functions,
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

void OptimizationView::DrawOptimizationProgressPopup()
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

void OptimizationView::DrawProgressLossPlot(const fdn_optimization::OptimizationProgressInfo& progress)
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

bool OptimizationView::HandleOptimizationResult(sfFDN::FDNConfig& fdn_config)
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

void OptimizationView::DrawResultSummary() const
{
    ImGui::Text("Elapsed Time: %.2f seconds", result_summary_.elapsed_time_sec);
    ImGui::Text("Evaluations: %u", result_summary_.evaluation_count);
    ImGui::Text("Initial Loss: %.6f", result_summary_.initial_loss);
    ImGui::Text("Final Loss: %.6f", result_summary_.final_loss);
}

void OptimizationView::PlotLossHistory()
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

OptimizationView::LossPlotStatistics OptimizationView::CalculateLossPlotStatistics() const
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

bool OptimizationView::HaveLossPlotBoundsChanged(const LossPlotStatistics& statistics) const
{
    return statistics.total_point_count != previous_loss_plot_statistics_.total_point_count ||
           statistics.max_history_size != previous_loss_plot_statistics_.max_history_size ||
           statistics.min_loss != previous_loss_plot_statistics_.min_loss ||
           statistics.max_loss != previous_loss_plot_statistics_.max_loss;
}

void OptimizationView::DrawPopulatedLossPlot(const LossPlotStatistics& statistics, bool history_bounds_changed) const
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

void OptimizationView::DrawLossSeries() const
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

void OptimizationView::DrawBestLossGuide() const
{
    if (loss_history_.losses.front().empty())
    {
        return;
    }

    const double best_loss = *std::ranges::min_element(loss_history_.losses.front());
    fdn_sandbox::plot_ui::DrawHorizontalGuide("Best Loss", best_loss,
                                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusOk), 1.0f);
}

void OptimizationView::DrawColorlessSettings()
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

void OptimizationView::DrawRIRMatchSettings()
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

    // Choosing target-derived initialization only records the intent; the estimates themselves are
    // measured from the target RIR in StartOptimization, since they depend on the band layout.
    ImGui::Checkbox("Start from estimated T60s", &use_estimated_t60s_);
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Measures per-band T60s from the target RIR and uses them as the optimizer's starting\n"
                          "point instead of the tuned random initialization.");
    }
    opt_info_.matching_parameters.initialization =
        use_estimated_t60s_ ? fdn_optimization::MatchingInitialization::TargetDerived : tuned_matching_initialization_;

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

} // namespace fdn_sandbox::views