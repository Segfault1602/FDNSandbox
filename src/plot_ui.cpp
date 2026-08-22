#include "plot_ui.h"

#include "theme.h"

#include <algorithm>
#include <array>
#include <string>

namespace fdn_sandbox::plot_ui
{
namespace
{
constexpr std::array<double, 12> kAudioFrequencyTicks = {
    20.0, 31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0,
};

constexpr std::array<const char*, 12> kAudioFrequencyLabels = {
    "20", "31", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k", "20k",
};

const std::array<ImVec4, 9> kOctaveBandColors = {
    ImVec4(0.800f, 0.400f, 0.467f, 1.0f), ImVec4(0.200f, 0.133f, 0.533f, 1.0f), ImVec4(0.867f, 0.800f, 0.467f, 1.0f),
    ImVec4(0.067f, 0.467f, 0.200f, 1.0f), ImVec4(0.533f, 0.800f, 0.933f, 1.0f), ImVec4(0.533f, 0.133f, 0.333f, 1.0f),
    ImVec4(0.267f, 0.667f, 0.600f, 1.0f), ImVec4(0.600f, 0.600f, 0.200f, 1.0f), ImVec4(0.667f, 0.267f, 0.600f, 1.0f),
};

ImVec4 SeriesColor(SeriesRole role)
{
    using fdn_sandbox::theme::Color;
    using fdn_sandbox::theme::ColorRole;

    switch (role)
    {
    case SeriesRole::Fdn:
        return Color(ColorRole::PlotFdn);
    case SeriesRole::Reference:
        return Color(ColorRole::PlotReference);
    case SeriesRole::Optimization:
        return Color(ColorRole::PlotOptimization);
    case SeriesRole::Annotation:
        return Color(ColorRole::TextSecondary);
    }
    return Color(ColorRole::TextPrimary);
}
} // namespace

ImPlotSpec LineSpec(SeriesRole role, float weight)
{
    ImPlotSpec spec{};
    spec.LineColor = SeriesColor(role);
    spec.LineWeight = weight;
    return spec;
}

ImPlotSpec MarkerSpec(SeriesRole role, ImPlotMarker marker, float marker_size, float weight)
{
    ImPlotSpec spec = LineSpec(role, weight);
    spec.Marker = marker;
    spec.MarkerSize = marker_size;
    spec.MarkerLineColor = spec.LineColor;
    spec.MarkerFillColor = spec.LineColor;
    return spec;
}

void SetupFrequencyAxis(ImAxis axis, bool logarithmic, double nyquist, ImPlotAxisFlags flags,
                        ImPlotCond limits_condition)
{
    const double max_frequency = std::max(20.0, nyquist);
    ImPlot::SetupAxis(axis, "Frequency (Hz)", flags);
    ImPlot::SetupAxisScale(axis, logarithmic ? ImPlotScale_Log10 : ImPlotScale_Linear);
    ImPlot::SetupAxisLimits(axis, logarithmic ? 20.0 : 0.0, max_frequency, limits_condition);
    ImPlot::SetupAxisLimitsConstraints(axis, logarithmic ? 10.0 : 0.0, max_frequency);

    if (logarithmic)
    {
        size_t tick_count = 0;
        while (tick_count < kAudioFrequencyTicks.size() && kAudioFrequencyTicks[tick_count] <= max_frequency)
        {
            ++tick_count;
        }
        ImPlot::SetupAxisTicks(axis, kAudioFrequencyTicks.data(), static_cast<int>(tick_count),
                               kAudioFrequencyLabels.data(), false);
    }
}

void DrawHorizontalGuide(const char* label, double value, const ImVec4& color, float weight)
{
    ImPlotSpec spec{};
    spec.LineColor = color;
    spec.LineWeight = weight;
    spec.Flags = ImPlotInfLinesFlags_Horizontal;
    ImPlot::PlotInfLines(label, &value, 1, spec);
}

void DrawVerticalGuide(const char* label, double value, const ImVec4& color, float weight)
{
    ImPlotSpec spec{};
    spec.LineColor = color;
    spec.LineWeight = weight;
    ImPlot::PlotInfLines(label, &value, 1, spec);
}

void DrawEmptyState(std::string_view message)
{
    const ImPlotRect limits = ImPlot::GetPlotLimits();
    const std::string message_text(message);
    ImPlotSpec spec{};
    spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary);
    ImPlot::PlotText(message_text.c_str(), limits.X.Min + (limits.X.Size() * 0.5),
                     limits.Y.Min + (limits.Y.Size() * 0.5), ImVec2(0.0f, 0.0f), spec);
}

ImPlotColormap OctaveBandColormap()
{
    static const ImPlotColormap colormap =
        ImPlot::AddColormap("FDNSandbox Octaves", kOctaveBandColors.data(), static_cast<int>(kOctaveBandColors.size()));
    return colormap;
}

ImVec4 OctaveBandColor(size_t index, float alpha)
{
    ImVec4 color = kOctaveBandColors[index % kOctaveBandColors.size()];
    color.w = alpha;
    return color;
}
} // namespace fdn_sandbox::plot_ui
