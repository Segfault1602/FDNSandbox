#pragma once

#include "optimizer.h"

#include "quill/Logger.h"
#include <sffdn/sffdn.h>

#include <span>

class OptimizationGUI
{
  public:
    OptimizationGUI(quill::Logger* logger);
    ~OptimizationGUI() = default;

    bool Draw(sfFDN::FDNConfig& fdn_config, std::span<const float> target_rir);

  private:
    fdn_optimization::FDNOptimizer fdn_optimizer_;
    quill::Logger* logger_;

    bool optimize_gains_checkbox_ = false;
    bool optimize_matrix_checkbox_ = false;
    bool optimize_filters_checkbox_ = false;

    fdn_optimization::OptimizationInfo opt_info_;
};