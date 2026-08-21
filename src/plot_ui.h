#pragma once

#include <implot.h>

#include <cstddef>
#include <string_view>

namespace fdn_sandbox::plot_ui
{
enum class SeriesRole
{
    Fdn,
    Reference,
    Optimization,
    Annotation,
};

ImPlotSpec LineSpec(SeriesRole role, float weight = 1.6f);
ImPlotSpec MarkerSpec(SeriesRole role, ImPlotMarker marker, float marker_size = 5.0f, float weight = 1.4f);

void SetupFrequencyAxis(ImAxis axis, bool logarithmic, double nyquist, ImPlotAxisFlags flags = ImPlotAxisFlags_None,
                        ImPlotCond limits_condition = ImPlotCond_Once);
void DrawHorizontalGuide(const char* label, double value, const ImVec4& color, float weight = 1.0f);
void DrawVerticalGuide(const char* label, double value, const ImVec4& color, float weight = 1.0f);
void DrawEmptyState(std::string_view message);

ImPlotColormap OctaveBandColormap();
ImVec4 OctaveBandColor(size_t index, float alpha = 1.0f);
} // namespace fdn_sandbox::plot_ui
