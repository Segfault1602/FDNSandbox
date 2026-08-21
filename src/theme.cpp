#include "theme.h"

#include <boost/dll/runtime_symbol_info.hpp>
#include <implot.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace fdn_sandbox::theme
{
namespace
{
using ColorArray = std::array<ImVec4, static_cast<size_t>(ColorRole::Count)>;
using FontArray = std::array<ImFont*, static_cast<size_t>(FontRole::Count)>;

const ColorArray kColors = {
    ImVec4(0.067f, 0.075f, 0.086f, 1.000f), // ApplicationBackground
    ImVec4(0.106f, 0.118f, 0.133f, 1.000f), // WindowBackground
    ImVec4(0.090f, 0.102f, 0.118f, 1.000f), // PanelBackground
    ImVec4(0.075f, 0.086f, 0.102f, 1.000f), // InsetBackground
    ImVec4(0.125f, 0.141f, 0.165f, 0.980f), // OverlayBackground
    ImVec4(0.133f, 0.165f, 0.192f, 1.000f), // FieldBackground
    ImVec4(0.165f, 0.216f, 0.251f, 1.000f), // FieldHovered
    ImVec4(0.192f, 0.286f, 0.333f, 1.000f), // FieldActive
    ImVec4(0.161f, 0.196f, 0.231f, 1.000f), // ButtonBackground
    ImVec4(0.200f, 0.263f, 0.306f, 1.000f), // ButtonHovered
    ImVec4(0.200f, 0.384f, 0.451f, 1.000f), // ButtonActive
    ImVec4(0.906f, 0.918f, 0.929f, 1.000f), // TextPrimary
    ImVec4(0.553f, 0.588f, 0.620f, 1.000f), // TextSecondary
    ImVec4(0.243f, 0.624f, 0.757f, 1.000f), // Accent
    ImVec4(0.345f, 0.706f, 0.824f, 1.000f), // AccentHovered
    ImVec4(0.176f, 0.471f, 0.576f, 1.000f), // AccentActive
    ImVec4(0.267f, 0.318f, 0.365f, 1.000f), // Border
    ImVec4(0.169f, 0.192f, 0.220f, 1.000f), // Separator
    ImVec4(0.349f, 0.663f, 0.416f, 1.000f), // StatusOk
    ImVec4(0.835f, 0.655f, 0.227f, 1.000f), // StatusWarning
    ImVec4(0.839f, 0.325f, 0.302f, 1.000f), // StatusError
    ImVec4(0.349f, 0.663f, 0.416f, 1.000f), // MeterSafe
    ImVec4(0.835f, 0.655f, 0.227f, 1.000f), // MeterWarning
    ImVec4(0.886f, 0.471f, 0.239f, 1.000f), // MeterHot
    ImVec4(0.290f, 0.565f, 0.761f, 1.000f), // PlotFdn
    ImVec4(0.890f, 0.427f, 0.490f, 1.000f), // PlotReference
    ImVec4(0.722f, 0.537f, 0.839f, 1.000f), // PlotOptimization
    ImVec4(0.388f, 0.773f, 0.651f, 1.000f), // PlotAnalysis1
    ImVec4(0.929f, 0.663f, 0.298f, 1.000f), // PlotAnalysis2
    ImVec4(0.510f, 0.671f, 0.886f, 1.000f), // PlotAnalysis3
    ImVec4(0.839f, 0.502f, 0.737f, 1.000f), // PlotAnalysis4
    ImVec4(0.655f, 0.773f, 0.349f, 1.000f), // PlotAnalysis5
};

FontArray fonts{};

constexpr std::array<float, static_cast<size_t>(FontRole::Count)> kFontSizes = {
    15.0f, // Body
    18.0f, // Heading
    14.0f, // Description
    12.5f, // Metadata
};

ImVec4 WithAlpha(const ImVec4& color, float alpha)
{
    return ImVec4(color.x, color.y, color.z, alpha);
}
} // namespace

const ImVec4& Color(ColorRole role)
{
    const size_t index = static_cast<size_t>(role);
    assert(index < kColors.size());
    return kColors[index];
}

void InitializeFonts()
{
    if (fonts[static_cast<size_t>(FontRole::Body)] != nullptr)
    {
        return;
    }

    const auto font_directory = boost::dll::program_location().parent_path() / "fonts";
    const std::string medium_path = (font_directory / "ClearSans-Medium.ttf").string();
    const std::string regular_path = (font_directory / "ClearSans-Regular.ttf").string();
    const std::string light_path = (font_directory / "ClearSans-Light.ttf").string();

    ImGuiIO& io = ImGui::GetIO();
    ImFont* medium = io.Fonts->AddFontFromFileTTF(medium_path.c_str());
    ImFont* regular = io.Fonts->AddFontFromFileTTF(regular_path.c_str());
    ImFont* light = io.Fonts->AddFontFromFileTTF(light_path.c_str());

    if (medium == nullptr || regular == nullptr || light == nullptr)
    {
        throw std::runtime_error("Failed to load ClearSans fonts from " + font_directory.string());
    }

    fonts[static_cast<size_t>(FontRole::Body)] = medium;
    fonts[static_cast<size_t>(FontRole::Heading)] = medium;
    fonts[static_cast<size_t>(FontRole::Description)] = regular;
    fonts[static_cast<size_t>(FontRole::Metadata)] = light;
    io.FontDefault = medium;
}

void PushFont(FontRole role)
{
    const size_t index = static_cast<size_t>(role);
    assert(index < fonts.size());
    assert(fonts[index] != nullptr);
    ImGui::PushFont(fonts[index], kFontSizes[index]);
}

void PopFont()
{
    ImGui::PopFont();
}

void Text(FontRole font_role, ColorRole color_role, std::string_view text)
{
    PushFont(font_role);
    ImGui::PushStyleColor(ImGuiCol_Text, Color(color_role));
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopStyleColor();
    PopFont();
}

void TextWrapped(FontRole font_role, ColorRole color_role, std::string_view text)
{
    PushFont(font_role);
    ImGui::PushStyleColor(ImGuiCol_Text, Color(color_role));
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    PopFont();
}

bool BeginSection(const char* label, bool default_open)
{
    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_FramePadding;
    if (default_open)
    {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }

    PushFont(FontRole::Heading);
    const bool open = ImGui::TreeNodeEx(label, flags);
    PopFont();
    return open;
}

void EndSection()
{
    ImGui::TreePop();
    ImGui::Spacing();
}

void DrawWordmark()
{
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const float height = ImGui::GetTextLineHeight();
    const float mark_width = height * 1.2f;
    const float center_y = start.y + height * 0.5f;
    const std::array points = {
        ImVec2(start.x, center_y),
        ImVec2(start.x + mark_width * 0.16f, center_y),
        ImVec2(start.x + mark_width * 0.30f, start.y),
        ImVec2(start.x + mark_width * 0.46f, start.y + height),
        ImVec2(start.x + mark_width * 0.63f, center_y),
        ImVec2(start.x + mark_width * 0.78f, center_y),
        ImVec2(start.x + mark_width * 0.89f, start.y + height * 0.25f),
        ImVec2(start.x + mark_width, center_y),
    };

    ImGui::GetWindowDrawList()->AddPolyline(points.data(), static_cast<int>(points.size()),
                                            ImGui::ColorConvertFloat4ToU32(Color(ColorRole::Accent)), 0, 1.8f);
    ImGui::Dummy(ImVec2(mark_width, height));
    ImGui::SameLine(0.0f, 5.0f);
    Text(FontRole::Body, ColorRole::TextPrimary, "FDN Sandbox");
    ImGui::SameLine(0.0f, 14.0f);
}

void Apply(float dpi_scale)
{
    assert(dpi_scale > 0.0f);

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    style.FontSizeBase = 15.0f;
    style.WindowPadding = ImVec2(10.0f, 10.0f);
    style.WindowRounding = 5.0f;
    style.WindowBorderSize = 1.0f;
    style.ChildRounding = 4.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 5.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = ImVec2(7.0f, 4.0f);
    style.FrameRounding = 4.0f;
    style.FrameBorderSize = 1.0f;
    style.ItemSpacing = ImVec2(8.0f, 5.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.CellPadding = ImVec2(7.0f, 4.0f);
    style.IndentSpacing = 20.0f;
    style.ScrollbarSize = 13.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabMinSize = 10.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;
    style.TabBorderSize = 0.0f;
    style.TabBarBorderSize = 1.0f;
    style.TabBarOverlineSize = 2.0f;
    style.SeparatorSize = 1.0f;
    style.DisabledAlpha = 0.65f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = Color(ColorRole::TextPrimary);
    colors[ImGuiCol_TextDisabled] = Color(ColorRole::TextSecondary);
    colors[ImGuiCol_WindowBg] = Color(ColorRole::WindowBackground);
    colors[ImGuiCol_ChildBg] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_PopupBg] = Color(ColorRole::OverlayBackground);
    colors[ImGuiCol_Border] = Color(ColorRole::Border);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_FrameBg] = Color(ColorRole::FieldBackground);
    colors[ImGuiCol_FrameBgHovered] = Color(ColorRole::FieldHovered);
    colors[ImGuiCol_FrameBgActive] = Color(ColorRole::FieldActive);
    colors[ImGuiCol_TitleBg] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_TitleBgActive] = Color(ColorRole::InsetBackground);
    colors[ImGuiCol_TitleBgCollapsed] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_MenuBarBg] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_ScrollbarBg] = Color(ColorRole::InsetBackground);
    colors[ImGuiCol_ScrollbarGrab] = Color(ColorRole::Border);
    colors[ImGuiCol_ScrollbarGrabHovered] = Color(ColorRole::Accent);
    colors[ImGuiCol_ScrollbarGrabActive] = Color(ColorRole::AccentActive);
    colors[ImGuiCol_CheckMark] = Color(ColorRole::Accent);
    colors[ImGuiCol_SliderGrab] = Color(ColorRole::Accent);
    colors[ImGuiCol_SliderGrabActive] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_Button] = Color(ColorRole::ButtonBackground);
    colors[ImGuiCol_ButtonHovered] = Color(ColorRole::ButtonHovered);
    colors[ImGuiCol_ButtonActive] = Color(ColorRole::ButtonActive);
    colors[ImGuiCol_Header] = WithAlpha(Color(ColorRole::Accent), 0.24f);
    colors[ImGuiCol_HeaderHovered] = WithAlpha(Color(ColorRole::Accent), 0.42f);
    colors[ImGuiCol_HeaderActive] = WithAlpha(Color(ColorRole::Accent), 0.58f);
    colors[ImGuiCol_Separator] = Color(ColorRole::Separator);
    colors[ImGuiCol_SeparatorHovered] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_SeparatorActive] = Color(ColorRole::AccentActive);
    colors[ImGuiCol_ResizeGrip] = WithAlpha(Color(ColorRole::Accent), 0.20f);
    colors[ImGuiCol_ResizeGripHovered] = WithAlpha(Color(ColorRole::Accent), 0.55f);
    colors[ImGuiCol_ResizeGripActive] = Color(ColorRole::AccentActive);
    colors[ImGuiCol_InputTextCursor] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_TabHovered] = WithAlpha(Color(ColorRole::Accent), 0.42f);
    colors[ImGuiCol_Tab] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_TabSelected] = Color(ColorRole::InsetBackground);
    colors[ImGuiCol_TabSelectedOverline] = Color(ColorRole::Accent);
    colors[ImGuiCol_TabDimmed] = Color(ColorRole::ApplicationBackground);
    colors[ImGuiCol_TabDimmedSelected] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_TabDimmedSelectedOverline] = Color(ColorRole::Separator);
    colors[ImGuiCol_DockingPreview] = WithAlpha(Color(ColorRole::Accent), 0.70f);
    colors[ImGuiCol_DockingEmptyBg] = Color(ColorRole::ApplicationBackground);
    colors[ImGuiCol_PlotLines] = Color(ColorRole::PlotFdn);
    colors[ImGuiCol_PlotLinesHovered] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_PlotHistogram] = Color(ColorRole::Accent);
    colors[ImGuiCol_PlotHistogramHovered] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_TableHeaderBg] = Color(ColorRole::PanelBackground);
    colors[ImGuiCol_TableBorderStrong] = Color(ColorRole::Border);
    colors[ImGuiCol_TableBorderLight] = Color(ColorRole::Separator);
    colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    colors[ImGuiCol_TableRowBgAlt] = WithAlpha(Color(ColorRole::TextPrimary), 0.025f);
    colors[ImGuiCol_TextLink] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_TextSelectedBg] = WithAlpha(Color(ColorRole::Accent), 0.35f);
    colors[ImGuiCol_TreeLines] = Color(ColorRole::Separator);
    colors[ImGuiCol_DragDropTarget] = Color(ColorRole::StatusWarning);
    colors[ImGuiCol_DragDropTargetBg] = WithAlpha(Color(ColorRole::StatusWarning), 0.18f);
    colors[ImGuiCol_UnsavedMarker] = Color(ColorRole::StatusWarning);
    colors[ImGuiCol_NavCursor] = Color(ColorRole::AccentHovered);
    colors[ImGuiCol_NavWindowingHighlight] = WithAlpha(Color(ColorRole::AccentHovered), 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.25f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.55f);

    style.ScaleAllSizes(dpi_scale);
    style.FontScaleDpi = dpi_scale;

    ImPlot::StyleColorsAuto();
    ImPlotStyle& plot_style = ImPlot::GetStyle();
    plot_style.PlotBorderSize = 1.0f;
    plot_style.MinorAlpha = 0.18f;
    plot_style.PlotPadding = ImVec2(10.0f, 10.0f);
    plot_style.LabelPadding = ImVec2(6.0f, 4.0f);
    plot_style.LegendPadding = ImVec2(8.0f, 8.0f);
    plot_style.LegendInnerPadding = ImVec2(6.0f, 5.0f);
    plot_style.Colors[ImPlotCol_FrameBg] = Color(ColorRole::InsetBackground);
    plot_style.Colors[ImPlotCol_PlotBg] = Color(ColorRole::InsetBackground);
    plot_style.Colors[ImPlotCol_PlotBorder] = Color(ColorRole::Border);
    plot_style.Colors[ImPlotCol_LegendBg] = Color(ColorRole::OverlayBackground);
    plot_style.Colors[ImPlotCol_LegendBorder] = Color(ColorRole::Border);
    plot_style.Colors[ImPlotCol_LegendText] = Color(ColorRole::TextPrimary);
    plot_style.Colors[ImPlotCol_TitleText] = Color(ColorRole::TextPrimary);
    plot_style.Colors[ImPlotCol_InlayText] = Color(ColorRole::TextSecondary);
    plot_style.Colors[ImPlotCol_AxisText] = Color(ColorRole::TextSecondary);
    plot_style.Colors[ImPlotCol_AxisGrid] = WithAlpha(Color(ColorRole::TextSecondary), 0.24f);
    plot_style.Colors[ImPlotCol_AxisTick] = WithAlpha(Color(ColorRole::TextSecondary), 0.52f);
    plot_style.Colors[ImPlotCol_AxisBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    plot_style.Colors[ImPlotCol_AxisBgHovered] = WithAlpha(Color(ColorRole::Accent), 0.16f);
    plot_style.Colors[ImPlotCol_AxisBgActive] = WithAlpha(Color(ColorRole::Accent), 0.28f);
    plot_style.Colors[ImPlotCol_Selection] = WithAlpha(Color(ColorRole::AccentHovered), 0.75f);
    plot_style.Colors[ImPlotCol_Crosshairs] = Color(ColorRole::TextSecondary);

    static ImPlotColormap fdn_colormap = -1;
    if (fdn_colormap == -1)
    {
        const std::array plot_colors = {
            Color(ColorRole::PlotFdn),       Color(ColorRole::PlotReference), Color(ColorRole::PlotOptimization),
            Color(ColorRole::PlotAnalysis1), Color(ColorRole::PlotAnalysis2), Color(ColorRole::PlotAnalysis3),
            Color(ColorRole::PlotAnalysis4), Color(ColorRole::PlotAnalysis5),
        };
        fdn_colormap = ImPlot::AddColormap("FDNSandbox", plot_colors.data(), static_cast<int>(plot_colors.size()));
    }
    plot_style.Colormap = fdn_colormap;
}
} // namespace fdn_sandbox::theme
