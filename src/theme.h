#pragma once

#include <imgui.h>

#include <cstdint>

namespace fdn_sandbox::theme
{
enum class ColorRole : uint8_t
{
    ApplicationBackground,
    WindowBackground,
    PanelBackground,
    InsetBackground,
    OverlayBackground,
    TextPrimary,
    TextSecondary,
    Accent,
    AccentHovered,
    AccentActive,
    Border,
    Separator,
    StatusOk,
    StatusWarning,
    StatusError,
    MeterSafe,
    MeterWarning,
    MeterHot,
    PlotFdn,
    PlotReference,
    PlotOptimization,
    PlotAnalysis1,
    PlotAnalysis2,
    PlotAnalysis3,
    PlotAnalysis4,
    PlotAnalysis5,
    Count
};

void Apply(float dpi_scale);
const ImVec4& Color(ColorRole role);
} // namespace fdn_sandbox::theme
