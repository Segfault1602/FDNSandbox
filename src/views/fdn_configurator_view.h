#pragma once

#include "fdn_widget.h"
#include "session/fdn_session.h"

#include <cstdint>

namespace fdn_sandbox::views
{

/**
 * @brief Renders the FDN editor and owns its persistent interaction state.
 *
 * The view edits the session draft and reports whether the app should realize
 * and publish a new FDN.
 */
class FdnConfiguratorView final
{
  public:
    bool Draw(session::FdnSession& session);

  private:
    struct StructureSectionResult
    {
        bool config_changed = false;
        bool fdn_size_changed = false;
    };

    StructureSectionResult DrawStructureSection(session::FdnSession& session);
    void DrawVisualOverviewSection(session::FdnSession& session, bool fdn_size_changed);
    bool DrawInputStageSection(session::FdnSession& session);
    bool DrawDelayNetworkSection(session::FdnSession& session);
    bool DrawFeedbackMatrixSection(session::FdnSession& session);
    bool DrawLoopFiltersSection(session::FdnSession& session);
    bool DrawOutputStageSection(session::FdnSession& session);
    bool DrawToneCorrectionSection(session::FdnSession& session);

    std::uint32_t random_seed_ = 0;
    bool initialized_ = false;
    bool random_seed_enabled_ = false;
    bool transpose_ = false;
    bool independent_t60s_ = false;
    FDNWidgetState widget_state_;
};

} // namespace fdn_sandbox::views
