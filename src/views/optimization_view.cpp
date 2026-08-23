#include "views/optimization_view.h"

#include "views/window_ids.h"

#include <imgui.h>

namespace fdn_sandbox::views
{

OptimizationView::OptimizationView(quill::Logger* logger)
    : gui_(logger)
{
}

bool OptimizationView::Draw(sfFDN::FDNConfig& config, std::span<const float> reference_ir)
{
    if (!ImGui::Begin(ids::kOptimization))
    {
        ImGui::End();
        return false;
    }
    const bool changed = gui_.Draw(config, reference_ir);
    ImGui::End();
    return changed;
}

bool OptimizationView::IsActive() const noexcept
{
    return gui_.IsActive();
}

std::optional<OptimizationGUI::Event> OptimizationView::ConsumeEvent()
{
    return gui_.ConsumeEvent();
}

} // namespace fdn_sandbox::views
