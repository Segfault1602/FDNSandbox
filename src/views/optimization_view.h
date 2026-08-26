#pragma once

#include "optimizer.h"
#include "tuned_defaults.h"

#include "quill/Logger.h"
#include <audio_loss.h>
#include <sffdn/sffdn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace fdn_sandbox::views
{

/// Owns the optimization window, its UI state, and the optimizer workflow.
class OptimizationView final
{
  public:
    enum class EventType
    {
        Completed,
        Canceled,
        Failed,
    };

    struct Event
    {
        EventType type;
        std::string message;
    };

    OptimizationView(const OptimizationView&) = delete;
    OptimizationView& operator=(const OptimizationView&) = delete;
    OptimizationView(OptimizationView&&) = delete;
    OptimizationView& operator=(OptimizationView&&) = delete;

    explicit OptimizationView(quill::Logger* logger);
    ~OptimizationView() = default;

    bool Draw(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);

    [[nodiscard]] bool IsActive() const noexcept;
    std::optional<Event> ConsumeEvent();

  private:
    using LossFunctions = std::vector<std::shared_ptr<fdn_optimization::AudioLoss>>;

    static constexpr std::size_t kObjectiveCount = 2;
    static constexpr std::size_t kAlgorithmCount =
        static_cast<std::size_t>(fdn_optimization::OptimizationAlgoType::Count);

    /// Holds one editable parameter set per (objective, algorithm) pairing.
    ///
    /// The two tuning campaigns produce hyperparameters that differ by orders of magnitude, so a
    /// single shared set cannot serve both. Entries are seeded from the library's tuned defaults on
    /// first use, which also guarantees the stored variant always holds the alternative matching its
    /// algorithm.
    class AlgorithmParameterStore
    {
      public:
        [[nodiscard]] fdn_optimization::OptimizationAlgoParams& Entry(fdn_optimization::OptimizationObjective objective,
                                                                      fdn_optimization::OptimizationAlgoType algorithm);

        void Reset(fdn_optimization::OptimizationObjective objective, fdn_optimization::OptimizationAlgoType algorithm);

      private:
        struct Slot
        {
            fdn_optimization::OptimizationAlgoParams params;
            bool seeded = false;
        };

        static std::size_t Index(fdn_optimization::OptimizationObjective objective);

        std::array<std::array<Slot, kAlgorithmCount>, kObjectiveCount> slots_;
    };

    struct LossHistory
    {
        std::vector<std::vector<double>> losses;
        std::vector<std::string> loss_names;
    };

    struct ResultSummary
    {
        double elapsed_time_sec = 0.0;
        uint32_t evaluation_count = 0;
        double initial_loss = 0.0;
        double final_loss = 0.0;
    };

    struct LossPlotStatistics
    {
        size_t total_point_count = 0;
        size_t max_history_size = 0;
        double min_loss = 0.0;
        double max_loss = 0.0;
    };

    fdn_optimization::FDNOptimizer fdn_optimizer_;

    bool optimize_gains_checkbox_ = false;
    bool optimize_matrix_checkbox_ = false;
    bool optimize_filters_checkbox_ = false;
    int optimize_type_ = 0;
    int matrix_type_ = 0;
    int filter_type_ = 0;

    // Seeded from the tuned campaign defaults by ApplyTunedLossDefaults in the constructor. The
    // zeros only exist so the object is never in an indeterminate state; they are never observed.
    double spectral_flatness_weight_ = 0.0;
    double sparsity_weight_ = 0.0;

    double edc_weight_ = 0.0;
    double mel_edr_weight_ = 0.0;

    uint32_t mel_edr_fft_length_ = 4096;
    uint32_t mel_edr_hop_size_ = 512;
    uint32_t mel_edr_window_size_ = 1024;
    uint32_t mel_edr_num_bands_ = 32;
    int selected_fft_size_index_ = 3;

    // Off by default so the tuned campaign initialization stays the shipped behaviour; the tuned
    // RIR-matching defaults were measured with random RT60 initialization.
    bool use_estimated_t60s_ = false;
    fdn_optimization::MatchingInitialization tuned_matching_initialization_ =
        fdn_optimization::MatchingInitialization::SeededRandom;

    fdn_optimization::OptimizationAlgoType selected_algorithm_ = fdn_optimization::OptimizationAlgoType::Adam;
    int selected_gradient_method_ = 0;

    AlgorithmParameterStore parameter_store_;

    fdn_optimization::OptimizationInfo opt_info_;
    LossHistory loss_history_;
    ResultSummary result_summary_;
    LossPlotStatistics previous_loss_plot_statistics_;
    std::optional<Event> pending_event_;

    [[nodiscard]] fdn_optimization::OptimizationObjective SelectedObjective() const noexcept;
    void ApplyTunedLossDefaults(fdn_optimization::OptimizationObjective objective);

    void DrawSetupPanel(std::span<const float> target_rir);
    bool DrawOptimizationPanel(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);
    void DrawAlgorithmSettings();
    void DrawAlgorithmSelector();
    void DrawGradientSettings();
    void DrawParameterEditor(fdn_optimization::OptimizationAlgoParams& params);
    void StartOptimization(const sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);
    [[nodiscard]] LossFunctions CreateLossFunctions(std::span<const float> target_rir) const;
    void AppendColorlessLossFunctions(LossFunctions& loss_functions) const;
    void AppendRIRMatchLossFunctions(LossFunctions& loss_functions, std::span<const float> target_rir) const;
    void DrawOptimizationProgressPopup();
    static void DrawProgressLossPlot(const fdn_optimization::OptimizationProgressInfo& progress);
    bool HandleOptimizationResult(sfFDN::FDNConfig& fdn_config);
    void DrawResultSummary() const;
    void PlotLossHistory();
    [[nodiscard]] LossPlotStatistics CalculateLossPlotStatistics() const;
    [[nodiscard]] bool HaveLossPlotBoundsChanged(const LossPlotStatistics& statistics) const;
    void DrawPopulatedLossPlot(const LossPlotStatistics& statistics, bool history_bounds_changed) const;
    void DrawLossSeries() const;
    void DrawBestLossGuide() const;
    void DrawColorlessSettings();
    void DrawRIRMatchSettings();
};

} // namespace fdn_sandbox::views