#pragma once

#include "optimizer.h"

#include "quill/Logger.h"
#include <audio_loss.h>
#include <sffdn/sffdn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct LossHistory
{
    std::vector<std::vector<double>> losses;
    std::vector<std::string> loss_names;
};

class OptimizationGUI
{
  public:
    OptimizationGUI(quill::Logger* logger);
    ~OptimizationGUI() = default;

    bool Draw(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);

    void PlotLossHistory();

  private:
    using LossFunctions = std::vector<std::shared_ptr<fdn_optimization::AudioLoss>>;

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

    double spectral_flatness_weight_ = 1.0;
    double sparsity_weight_ = 1.0;

    double edc_weight_ = 1.0;
    double mel_edr_weight_ = 1.0;

    uint32_t mel_edr_fft_length_ = 4096;
    uint32_t mel_edr_hop_size_ = 512;
    uint32_t mel_edr_window_size_ = 1024;
    uint32_t mel_edr_num_bands_ = 32;
    int selected_fft_size_index_ = 3;

    fdn_optimization::OptimizationAlgoType selected_algorithm_ = fdn_optimization::OptimizationAlgoType::Adam;
    int selected_gradient_method_ = 0;

    fdn_optimization::OptimizationInfo opt_info_;
    LossHistory loss_history_;
    ResultSummary result_summary_;
    LossPlotStatistics previous_loss_plot_statistics_;

    void DrawSetupPanel(std::span<const float> target_rir);
    bool DrawOptimizationPanel(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);
    void DrawAlgorithmSettings();
    void DrawAlgorithmSelector();
    void DrawGradientSettings();
    void StartOptimization(const sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);
    [[nodiscard]] LossFunctions CreateLossFunctions(std::span<const float> target_rir) const;
    void AppendColorlessLossFunctions(LossFunctions& loss_functions) const;
    void AppendRIRMatchLossFunctions(LossFunctions& loss_functions, std::span<const float> target_rir) const;
    void DrawOptimizationProgressPopup();
    static void DrawProgressLossPlot(const fdn_optimization::OptimizationProgressInfo& progress);
    bool HandleOptimizationResult(sfFDN::FDNConfig& fdn_config);
    void DrawResultSummary() const;
    [[nodiscard]] LossPlotStatistics CalculateLossPlotStatistics() const;
    [[nodiscard]] bool HaveLossPlotBoundsChanged(const LossPlotStatistics& statistics) const;
    void DrawPopulatedLossPlot(const LossPlotStatistics& statistics, bool history_bounds_changed) const;
    void DrawLossSeries() const;
    void DrawBestLossGuide() const;
    void DrawColorlessSettings();
    void DrawRIRMatchSettings();
};