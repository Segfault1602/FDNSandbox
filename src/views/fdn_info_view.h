#pragma once

#include "analysis/analysis_workspace.h"
#include "session/fdn_session.h"

namespace fdn_sandbox::views
{

/// Renders a read-only summary of the active FDN session and analysis results.
class FdnInfoView final
{
  public:
    static void Draw(const session::FdnSession& session, analysis::AnalysisWorkspace& workspace, float cpu_usage);
};

} // namespace fdn_sandbox::views
