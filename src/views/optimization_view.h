#pragma once

#include "optimization_gui.h"

#include <quill/Logger.h>
#include <sffdn/sffdn.h>

#include <optional>
#include <span>

namespace fdn_sandbox::views
{

/// Owns the optimizer UI and exposes its configuration changes and completion events.
class OptimizationView final
{
  public:
    explicit OptimizationView(quill::Logger* logger);

    bool Draw(sfFDN::FDNConfig& config, std::span<const float> reference_ir);
    bool IsActive() const noexcept;
    std::optional<OptimizationGUI::Event> ConsumeEvent();

  private:
    OptimizationGUI gui_;
};

} // namespace fdn_sandbox::views
