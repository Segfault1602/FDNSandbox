#pragma once

#include "optimizer.h"

#include "quill/Logger.h"
#include <sffdn/sffdn.h>

#include <span>

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
    fdn_optimization::FDNOptimizer fdn_optimizer_;

    bool optimize_gains_checkbox_ = false;
    bool optimize_matrix_checkbox_ = false;
    bool optimize_filters_checkbox_ = false;

    double spectral_flatness_weight_ = 1.0;
    double sparsity_weight_ = 1.0;

    double edc_weight_ = 1.0;
    double mel_edr_weight_ = 1.0;

    uint32_t mel_edr_fft_length_ = 4096;
    uint32_t mel_edr_hop_size_ = 512;
    uint32_t mel_edr_window_size_ = 1024;
    uint32_t mel_edr_num_bands_ = 32;

    fdn_optimization::OptimizationInfo opt_info_;
    LossHistory loss_history_;

    void DrawColorlessSettings();
    void DrawRIRMatchSettings(bool has_target_rir);
};