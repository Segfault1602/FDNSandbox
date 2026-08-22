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

struct MarkerSpecOptions
{
    SeriesRole role = SeriesRole::Fdn;
    ImPlotMarker marker = ImPlotMarker_Circle;
    float marker_size = 5.0f;
    float weight = 1.4f;
};

struct FrequencyAxisOptions
{
    ImAxis axis = ImAxis_X1;
    bool logarithmic = true;
    double nyquist = 0.0;
    ImPlotAxisFlags flags = ImPlotAxisFlags_None;
    ImPlotCond limits_condition = ImPlotCond_Once;
};

struct OctaveBandColorOptions
{
    size_t index = 0;
    float alpha = 1.0f;
};

ImPlotSpec LineSpec(SeriesRole role, float weight = 1.6f);
ImPlotSpec MarkerSpec(MarkerSpecOptions options);

void SetupFrequencyAxis(const FrequencyAxisOptions& options);
void DrawHorizontalGuide(const char* label, double value, const ImVec4& color, float weight = 1.0f);
void DrawVerticalGuide(const char* label, double value, const ImVec4& color, float weight = 1.0f);
void DrawEmptyState(std::string_view message);

ImPlotColormap OctaveBandColormap();
ImVec4 OctaveBandColor(OctaveBandColorOptions options);
} // namespace fdn_sandbox::plot_ui
