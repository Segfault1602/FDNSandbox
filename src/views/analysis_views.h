#pragma once

#include "analysis/analysis_workspace.h"

#include <memory>

namespace fdn_sandbox::views
{

/**
 * @brief Renders all analysis windows with per-instance UI state.
 *
 * Analyzer results are borrowed only while drawing and are never retained
 * across workspace invalidation.
 */
class AnalysisViews final
{
  public:
    AnalysisViews();
    ~AnalysisViews();

    AnalysisViews(const AnalysisViews&) = delete;
    AnalysisViews& operator=(const AnalysisViews&) = delete;
    AnalysisViews(AnalysisViews&&) noexcept;
    AnalysisViews& operator=(AnalysisViews&&) noexcept;

    void DrawImpulseResponse(analysis::AnalysisWorkspace& workspace);
    void DrawAll(analysis::AnalysisWorkspace& workspace);

  private:
    struct State;

    void DrawSpectrogram(analysis::AnalysisWorkspace& workspace);
    void DrawSpectrum(analysis::AnalysisWorkspace& workspace);
    void DrawAutocorrelation(analysis::AnalysisWorkspace& workspace);
    static void DrawFilterResponse(analysis::AnalysisWorkspace& workspace);
    void DrawEnergyDecayCurve(analysis::AnalysisWorkspace& workspace);
    void DrawEnergyDecayRelief(analysis::AnalysisWorkspace& workspace);
    void DrawCepstrum(analysis::AnalysisWorkspace& workspace);
    void DrawEchoDensity(analysis::AnalysisWorkspace& workspace);
    void DrawT60s(analysis::AnalysisWorkspace& workspace);

    std::unique_ptr<State> state_;
};

} // namespace fdn_sandbox::views
