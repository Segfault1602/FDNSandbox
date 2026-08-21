#pragma once

#include <imgui.h>

#include <cstdint>
#include <string_view>

namespace fdn_sandbox::theme
{
enum class ColorRole : uint8_t
{
    ApplicationBackground,
    WindowBackground,
    PanelBackground,
    InsetBackground,
    OverlayBackground,
    FieldBackground,
    FieldHovered,
    FieldActive,
    ButtonBackground,
    ButtonHovered,
    ButtonActive,
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

enum class FontRole : uint8_t
{
    Body,
    Heading,
    Description,
    Metadata,
    Count
};

void Apply(float dpi_scale);
void InitializeFonts();
const ImVec4& Color(ColorRole role);
void PushFont(FontRole role);
void PopFont();
void Text(FontRole font_role, ColorRole color_role, std::string_view text);
void TextWrapped(FontRole font_role, ColorRole color_role, std::string_view text);
bool BeginSection(const char* label, bool default_open = false);
void EndSection();
void DrawWordmark();
} // namespace fdn_sandbox::theme
