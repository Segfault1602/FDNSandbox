#pragma once

#include "analysis/analysis_workspace.h"

namespace fdn_sandbox::views
{

/// Renders application and spectrogram settings with per-instance control state.
class SettingsView final
{
  public:
    void Draw(analysis::AnalysisWorkspace& workspace);

  private:
    int selected_spectrogram_type_ = 0;
    int selected_fft_size_index_ = 1;
    int selected_window_size_index_ = 2;
    int selected_window_type_ = 2;
    float overlap_ = 0.5f;
    bool initialized_ = false;
};

} // namespace fdn_sandbox::views
