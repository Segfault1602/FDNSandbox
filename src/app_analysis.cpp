#include "app.h"

#include "fdn_analyzer.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft_utils.h>

#include "plot_ui.h"
#include "settings.h"
#include "theme.h"
#include "utils.h"
#include "widget.h"

#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <numbers>
#include <span>
#include <string>
#include <vector>

namespace
{
struct MelFormatterContext
{
    std::span<const float> mel_frequencies;
};

int MelFormatter(double value, char* buff, int size, void* data)
{
    const MelFormatterContext context = *reinterpret_cast<MelFormatterContext*>(data);
    auto mel_index = static_cast<size_t>(value);
    if (mel_index >= context.mel_frequencies.size())
    {
        mel_index = context.mel_frequencies.size() - 1; // Clamp to the last index if out of bounds
    }
    auto mel = static_cast<uint32_t>(context.mel_frequencies[mel_index]);
    const std::span<char> out_string(buff, size);
    const std::format_to_n_result result = std::format_to_n(out_string.begin(), size, "{}", mel);
    *result.out = '\0';
    return std::strlen(buff);
}

enum class SpectrumPlotMode : uint8_t
{
    Spectrum,
    Peaks,
    Both,
    Histogram,
};

struct SpectrumUiState
{
    double early_rir_duration = 0.5;
    SpectrumPlotMode mode = SpectrumPlotMode::Spectrum;
    SpectrumPlotMode previous_mode = SpectrumPlotMode::Spectrum;
    bool logarithmic_frequency = true;
    bool previous_logarithmic_frequency = true;
    bool show_rir = false;
};

bool DrawSpectrumModeButton(const char* label, SpectrumPlotMode candidate, SpectrumPlotMode& selected)
{
    if (!ImGui::RadioButton(label, selected == candidate))
    {
        return false;
    }
    selected = candidate;
    return true;
}

void DrawSpectrumControls(SpectrumUiState& state, bool has_rir)
{
    DrawSpectrumModeButton("Only Spectrum", SpectrumPlotMode::Spectrum, state.mode);
    ImGui::SameLine();
    DrawSpectrumModeButton("Only Peaks", SpectrumPlotMode::Peaks, state.mode);
    ImGui::SameLine();
    DrawSpectrumModeButton("Both", SpectrumPlotMode::Both, state.mode);
    ImGui::SameLine();
    DrawSpectrumModeButton("Histogram", SpectrumPlotMode::Histogram, state.mode);

    if (state.mode != SpectrumPlotMode::Histogram)
    {
        ImGui::SameLine();
        ImGui::Checkbox("Log Frequency", &state.logarithmic_frequency);
    }
    if (has_rir)
    {
        ImGui::Checkbox("Show RIR", &state.show_rir);
    }
    else
    {
        state.show_rir = false;
    }
}

void DrawSpectrumPicker(SpectrumUiState& state, fdn_analysis::FDNAnalyzer& fdn_analyzer,
                        fdn_analysis::IRAnalyzer& rir_analyzer)
{
    if (!ImPlot::BeginPlot("Impulse Response", ImVec2(), ImPlotFlags_NoLegend))
    {
        return;
    }
    if (state.show_rir)
    {
        DrawEarlyRIRPicker(rir_analyzer.GetImpulseResponse(), rir_analyzer.GetTimeData(), state.early_rir_duration);
    }
    else
    {
        DrawEarlyRIRPicker(fdn_analyzer.GetImpulseResponse(), fdn_analyzer.GetTimeData(), state.early_rir_duration);
    }
    ImPlot::EndPlot();
}

fdn_analysis::SpectrumData GetSpectrumData(const SpectrumUiState& state, fdn_analysis::FDNAnalyzer& fdn_analyzer,
                                           fdn_analysis::IRAnalyzer& rir_analyzer)
{
    return state.show_rir ? rir_analyzer.GetSpectrum(state.early_rir_duration)
                          : fdn_analyzer.GetSpectrum(state.early_rir_duration);
}

void DrawSpectrumSeries(const fdn_analysis::SpectrumData& data, SpectrumUiState& state, bool plot_mode_changed)
{
    const auto series_role =
        state.show_rir ? fdn_sandbox::plot_ui::SeriesRole::Reference : fdn_sandbox::plot_ui::SeriesRole::Fdn;
    const bool frequency_scale_changed = state.previous_logarithmic_frequency != state.logarithmic_frequency;
    fdn_sandbox::plot_ui::SetupFrequencyAxis(
        ImAxis_X1, state.logarithmic_frequency, data.frequency_bins.back(), ImPlotAxisFlags_None,
        plot_mode_changed || frequency_scale_changed ? ImPlotCond_Always : ImPlotCond_Once);
    state.previous_logarithmic_frequency = state.logarithmic_frequency;
    ImPlot::SetupAxis(ImAxis_Y1, "Magnitude (dB)");
    ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -60.0, 20.0, plot_mode_changed ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 40.0);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##0 dB", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##Noise Floor", -60.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator), 0.75f);

    if (state.mode == SpectrumPlotMode::Spectrum || state.mode == SpectrumPlotMode::Both)
    {
        const size_t count = std::min(data.frequency_bins.size(), data.spectrum.size());
        ImPlot::PlotLine("Spectrum", data.frequency_bins.data(), data.spectrum.data(), count,
                         fdn_sandbox::plot_ui::LineSpec(series_role, 1.7f));
    }
    if ((state.mode == SpectrumPlotMode::Peaks || state.mode == SpectrumPlotMode::Both) && !data.peaks.empty())
    {
        const size_t count = std::min(data.peaks_freqs.size(), data.peaks.size());
        const float marker_size = state.mode == SpectrumPlotMode::Peaks ? 2.5f : 3.0f;
        ImPlot::PlotScatter("Peaks", data.peaks_freqs.data(), data.peaks.data(), count,
                            fdn_sandbox::plot_ui::MarkerSpec(series_role, ImPlotMarker_Circle, marker_size, 1.0f));
    }
}

void DrawSpectrumHistogram(const fdn_analysis::SpectrumData& data, const SpectrumUiState& state, bool plot_mode_changed)
{
    ImPlot::SetupAxes("Magnitude (dB)", "Count");
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Linear);
    ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Linear);
    ImPlot::SetupAxisLimits(ImAxis_X1, -60.0, 20.0, plot_mode_changed ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, -120.0, 40.0);
    if (data.peaks.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No spectral peaks available");
        return;
    }

    ImPlotSpec spec = fdn_sandbox::plot_ui::LineSpec(state.show_rir ? fdn_sandbox::plot_ui::SeriesRole::Reference
                                                                    : fdn_sandbox::plot_ui::SeriesRole::Fdn);
    spec.FillColor = spec.LineColor;
    spec.FillAlpha = 0.65f;
    ImPlot::PlotHistogram("Histogram", data.peaks.data(), data.peaks.size(), ImPlotBin_Sqrt, 0.95f,
                          ImPlotRange(-60, 20), spec);
}

void DrawSpectrumPlot(SpectrumUiState& state, const fdn_analysis::SpectrumData& data)
{
    const bool plot_mode_changed = state.previous_mode != state.mode;
    if (plot_mode_changed && state.mode == SpectrumPlotMode::Histogram)
    {
        ImPlot::SetNextAxisToFit(ImAxis_Y1);
    }

    const std::string title = std::format("Spectrum ({} peaks)###Spectrum Plot", data.peaks.size());
    if (!ImPlot::BeginPlot(title.c_str(), ImVec2(), ImPlotFlags_NoLegend))
    {
        return;
    }
    if (data.spectrum.empty() || data.frequency_bins.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No spectrum data available");
    }
    else if (state.mode == SpectrumPlotMode::Histogram)
    {
        DrawSpectrumHistogram(data, state, plot_mode_changed);
    }
    else
    {
        DrawSpectrumSeries(data, state, plot_mode_changed);
    }
    state.previous_mode = state.mode;
    ImPlot::EndPlot();
}

void DrawSpectrumContents(SpectrumUiState& state, fdn_analysis::FDNAnalyzer& fdn_analyzer,
                          fdn_analysis::IRAnalyzer& rir_analyzer)
{
    static std::array row_ratios = {0.20f, 0.80f};
    if (!ImPlot::BeginSubplots("Spectrum Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, row_ratios.data()))
    {
        return;
    }
    DrawSpectrumPicker(state, fdn_analyzer, rir_analyzer);
    DrawSpectrumPlot(state, GetSpectrumData(state, fdn_analyzer, rir_analyzer));
    ImPlot::EndSubplots();
}
} // namespace

void FDNToolboxApp::DrawImpulseResponse()
{
    if (!ImGui::Begin("Impulse Response"))
    {
        ImGui::End();
        return;
    }

    constexpr float kMinDuration = 1.f;
    constexpr float kMaxDuration = 10.f;
    static float ir_duration = 1.f;
    if (ImGui::SliderScalar("IR Duration", ImGuiDataType_Float, &ir_duration, &kMinDuration, &kMaxDuration, "%.2f"))
    {
        Settings::Instance().SetIRDuration(ir_duration);
        fdn_analyzer_.SetImpulseResponseSize(ir_duration * Settings::Instance().SampleRate());
    }

    if (!rir_analyzer_.GetImpulseResponse().empty())
    {
        if (ImGui::Button("Match RIR Length"))
        {
            const float rir_duration = static_cast<float>(rir_analyzer_.GetImpulseResponse().size()) /
                                       static_cast<float>(Settings::Instance().SampleRate());
            ir_duration = rir_duration;
            Settings::Instance().SetIRDuration(ir_duration);
            fdn_analyzer_.SetImpulseResponseSize(ir_duration * Settings::Instance().SampleRate());
        }
    }

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    const auto imp_response = fdn_analyzer_.GetImpulseResponse();
    const auto time_data = fdn_analyzer_.GetTimeData();
    const auto rir_response = rir_analyzer_.GetImpulseResponse();
    const auto rir_time_data = rir_analyzer_.GetTimeData();
    const bool show_rir_overlay = show_rir && !rir_response.empty();
    const ImPlotFlags plot_flags = show_rir_overlay ? ImPlotFlags_None : ImPlotFlags_NoLegend;

    if (ImPlot::BeginPlot("Impulse Response", ImVec2(-1, -1), plot_flags))
    {
        ImPlot::SetupAxes("Time (s)", "Amplitude", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.25f, 1.25f);

        if (imp_response.empty() || time_data.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No impulse response available");
        }
        else
        {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, time_data.back(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, time_data.back());
            fdn_sandbox::plot_ui::DrawHorizontalGuide(
                "##Zero Amplitude", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary),
                0.75f);

            ImPlotSpec fdn_spec = fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn, 1.7f);
            if (fdn_analyzer_.IsClipping())
            {
                fdn_spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusError);
            }
            const size_t fdn_count = std::min(imp_response.size(), time_data.size());
            ImPlot::PlotLine("FDN Response", time_data.data(), imp_response.data(), fdn_count, fdn_spec);

            if (show_rir_overlay && !rir_time_data.empty())
            {
                const size_t rir_count = std::min(rir_response.size(), rir_time_data.size());
                ImPlot::PlotLine("Reference RIR", rir_time_data.data(), rir_response.data(), rir_count,
                                 fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.3f));
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::End(); // End the Impulse Response window
}
void FDNToolboxApp::DrawVisualization()
{
    DrawSpectrogram();
    DrawSpectrum();
    DrawCepstrum();
    DrawAutocorrelation();
    DrawFilterResponse();
    DrawEnergyDecayCurve();
    DrawEnergyDecayRelief();
    DrawT60s();
    DrawEchoDensity();
}
void FDNToolboxApp::DrawOptimizationLoss()
{
    if (!ImGui::Begin("Optimization Loss"))
    {
        ImGui::End();
        return;
    }

    optimization_gui_.PlotLossHistory();

    ImGui::End();
}
void FDNToolboxApp::DrawSpectrogram()
{
    if (!ImGui::Begin("Spectrogram"))
    {
        ImGui::End();
        return;
    }

    static float min_dB = -70.0f;
    static float max_dB = 0.0f;
    static std::vector<float> mels{};
    static MelFormatterContext ctx{.mel_frequencies = mels};

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    ImPlot::PushColormap(ImPlotColormap_Plasma);

    constexpr float kColorBarWidth = 100.0f;
    fdn_analysis::SpectrogramData spectrogram_data{};
    double tmax = 1.0;
    if (show_rir)
    {
        spectrogram_data = rir_analyzer_.GetSpectrogram(stft_options_, spectrogram_type_ == SpectrogramType::Mel);
        tmax = rir_analyzer_.GetImpulseResponseSize() / static_cast<double>(Settings::Instance().SampleRate());
    }
    else
    {
        spectrogram_data = fdn_analyzer_.GetSpectrogram(stft_options_, spectrogram_type_ == SpectrogramType::Mel);
        tmax = fdn_analyzer_.GetImpulseResponseSize() / static_cast<double>(Settings::Instance().SampleRate());
    }

    static bool previous_show_rir = show_rir;
    static SpectrogramType previous_spectrogram_type = spectrogram_type_;
    static uint32_t previous_bin_count = 0;
    static uint32_t previous_frame_count = 0;
    const bool dimensions_changed = show_rir != previous_show_rir || spectrogram_type_ != previous_spectrogram_type ||
                                    spectrogram_data.bin_count != previous_bin_count ||
                                    spectrogram_data.frame_count != previous_frame_count;

    if (ImPlot::BeginPlot("##Spectrogram", ImVec2(ImGui::GetCurrentWindow()->Size[0] - kColorBarWidth, -1)))
    {
        const double tmin = 0.0;
        double bin_count = Settings::Instance().SampleRate() / 2.0;
        const char* frequency_axis_label = "Frequency (Hz)";

        if (spectrogram_type_ == SpectrogramType::Mel && !spectrogram_data.data.empty())
        {
            mels = audio_utils::GetMelFrequencies(spectrogram_data.bin_count, 0.f,
                                                  Settings::Instance().SampleRate() / 2.f);
            ctx.mel_frequencies = mels;
            ImPlot::SetupAxisFormat(ImAxis_Y1, MelFormatter, &ctx);
            bin_count = spectrogram_data.bin_count;
            frequency_axis_label = "Mel Band Frequency (Hz)";
        }

        ImPlot::SetupAxes("Time (s)", frequency_axis_label, ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        const ImPlotCond view_condition = dimensions_changed ? ImPlotCond_Always : ImPlotCond_Once;
        ImPlot::SetupAxisLimits(ImAxis_X1, tmin, std::max(tmax, 0.001), view_condition);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(bin_count, 1.0), view_condition);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, tmin, std::max(tmax, 0.001));
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0.0, std::max(bin_count, 1.0));

        if (spectrogram_data.data.empty() || spectrogram_data.bin_count == 0 || spectrogram_data.frame_count == 0)
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No spectrogram data available");
        }
        else
        {
            ImPlotSpec spec{};
            spec.Flags = ImPlotHeatmapFlags_ColMajor;
            ImPlot::PlotHeatmap("##Spectrogram", spectrogram_data.data.data(), spectrogram_data.bin_count,
                                spectrogram_data.frame_count, min_dB, max_dB, nullptr, {tmin, 0}, {tmax, bin_count},
                                spec);
        }

        ImPlot::EndPlot();
    }

    ImGui::SameLine();
    ImGui::BeginGroup();
    fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Metadata, fdn_sandbox::theme::ColorRole::TextSecondary,
                             "dB");
    ImPlot::ColormapScale("##HeatScale", min_dB, max_dB, ImVec2(-1, -1));

    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
    {
        ImGui::OpenPopup("Range");
    }
    if (ImGui::BeginPopup("Range"))
    {
        ImGui::SliderFloat("Max", &max_dB, min_dB, 100);
        ImGui::SliderFloat("Min", &min_dB, -100, max_dB);
        ImGui::EndPopup();
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Right-click to adjust the displayed dB range");
    }
    ImGui::EndGroup();
    ImPlot::PopColormap();

    previous_show_rir = show_rir;
    previous_spectrogram_type = spectrogram_type_;
    previous_bin_count = spectrogram_data.bin_count;
    previous_frame_count = spectrogram_data.frame_count;

    ImGui::End(); // End the Spectrogram window
}
void FDNToolboxApp::DrawSpectrum()
{
    if (!ImGui::Begin("Spectrum"))
    {
        ImGui::End();
        return;
    }

    static SpectrumUiState state;
    DrawSpectrumControls(state, !rir_analyzer_.GetImpulseResponse().empty());
    DrawSpectrumContents(state, fdn_analyzer_, rir_analyzer_);
    ImGui::End();
}

namespace
{
enum class AutocorrelationType : uint8_t
{
    Time,
    Spectral,
};

struct AutocorrelationUiState
{
    double duration = 0.25;
    AutocorrelationType type = AutocorrelationType::Time;
    AutocorrelationType previous_type = AutocorrelationType::Time;
    size_t previous_lag_size = 0;
    bool show_rir = false;
    std::vector<float> lag_axis;
    std::vector<float> rir_lag_axis;
};

void DrawAutocorrelationControls(AutocorrelationUiState& state, bool has_rir)
{
    if (ImGui::RadioButton("Time", state.type == AutocorrelationType::Time))
    {
        state.type = AutocorrelationType::Time;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Spectral", state.type == AutocorrelationType::Spectral))
    {
        state.type = AutocorrelationType::Spectral;
    }

    if (has_rir)
    {
        ImGui::Checkbox("Show RIR", &state.show_rir);
    }
    else
    {
        state.show_rir = false;
    }
}

void DrawAutocorrelationPicker(AutocorrelationUiState& state, fdn_analysis::FDNAnalyzer& fdn_analyzer,
                               fdn_analysis::IRAnalyzer& rir_analyzer)
{
    if (!ImPlot::BeginPlot("Impulse Response", ImVec2(), ImPlotFlags_NoLegend))
    {
        return;
    }
    if (state.show_rir)
    {
        DrawEarlyRIRPicker(rir_analyzer.GetImpulseResponse(), rir_analyzer.GetTimeData(), state.duration);
    }
    else
    {
        DrawEarlyRIRPicker(fdn_analyzer.GetImpulseResponse(), fdn_analyzer.GetTimeData(), state.duration);
    }
    state.duration = std::clamp(state.duration, 0.1, 0.9);
    ImPlot::EndPlot();
}

void UpdateLagAxis(std::vector<float>& axis, size_t size, AutocorrelationType type)
{
    axis.resize(size);
    const float scale =
        type == AutocorrelationType::Time ? 1000.0f / static_cast<float>(Settings::Instance().SampleRate()) : 1.0f;
    for (size_t i = 0; i < axis.size(); ++i)
    {
        axis[i] = static_cast<float>(i) * scale;
    }
}

std::span<const float> SelectAutocorrelation(const fdn_analysis::AutocorrelationData& data, AutocorrelationType type)
{
    return type == AutocorrelationType::Time ? std::span<const float>(data.autocorrelation)
                                             : std::span<const float>(data.spectral_autocorrelation);
}

void DrawAutocorrelationPlot(AutocorrelationUiState& state, std::span<const float> fdn_data,
                             std::span<const float> rir_data)
{
    const ImPlotFlags flags = state.show_rir ? ImPlotFlags_None : ImPlotFlags_NoLegend;
    if (!ImPlot::BeginPlot("Autocorrelation", ImVec2(-1, -1), flags))
    {
        return;
    }

    UpdateLagAxis(state.lag_axis, fdn_data.size(), state.type);
    UpdateLagAxis(state.rir_lag_axis, rir_data.size(), state.type);
    ImPlot::SetupAxes(state.type == AutocorrelationType::Time ? "Lag (ms)" : "Frequency-bin Lag",
                      "Normalized Correlation");
    ImPlot::SetupAxisLimits(ImAxis_Y1, -1.1, 1.1, ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.5, 1.5);

    if (fdn_data.empty() || state.lag_axis.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No autocorrelation data available");
    }
    else
    {
        const bool range_changed =
            state.previous_type != state.type || state.previous_lag_size != state.lag_axis.size();
        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, state.lag_axis.back(),
                                range_changed ? ImPlotCond_Always : ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, state.lag_axis.back());
        fdn_sandbox::plot_ui::DrawVerticalGuide(
            "##Zero Lag", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
        ImPlot::PlotLine("FDN Autocorrelation", state.lag_axis.data(), fdn_data.data(), fdn_data.size(),
                         fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
        if (state.show_rir && !rir_data.empty())
        {
            ImPlot::PlotLine("RIR Autocorrelation", state.rir_lag_axis.data(), rir_data.data(), rir_data.size(),
                             fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.3f));
        }
        state.previous_type = state.type;
        state.previous_lag_size = state.lag_axis.size();
    }
    ImPlot::EndPlot();
}

void DrawAutocorrelationContents(AutocorrelationUiState& state, fdn_analysis::FDNAnalyzer& fdn_analyzer,
                                 fdn_analysis::IRAnalyzer& rir_analyzer)
{
    static std::array row_ratios = {0.2f, 0.8f};
    if (!ImPlot::BeginSubplots("Autocorrelation Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend,
                               row_ratios.data()))
    {
        return;
    }

    DrawAutocorrelationPicker(state, fdn_analyzer, rir_analyzer);
    const fdn_analysis::AutocorrelationData fdn_data = fdn_analyzer.GetAutocorrelation(state.duration);
    const fdn_analysis::AutocorrelationData rir_data =
        state.show_rir ? rir_analyzer.GetAutocorrelation(state.duration) : fdn_analysis::AutocorrelationData{};
    DrawAutocorrelationPlot(state, SelectAutocorrelation(fdn_data, state.type),
                            SelectAutocorrelation(rir_data, state.type));
    ImPlot::EndSubplots();
}
} // namespace

void FDNToolboxApp::DrawAutocorrelation()
{
    if (!ImGui::Begin("Autocorrelation"))
    {
        ImGui::End();
        return;
    }

    static AutocorrelationUiState state;
    DrawAutocorrelationControls(state, !rir_analyzer_.GetImpulseResponse().empty());
    DrawAutocorrelationContents(state, fdn_analyzer_, rir_analyzer_);
    ImGui::End();
}

namespace
{
float FindMinimumMagnitude(const fdn_analysis::FilterData& data)
{
    float minimum = 0.0f;
    for (const auto response : data.mag_responses)
    {
        if (!response.empty())
        {
            minimum = std::min(minimum, *std::ranges::min_element(response));
        }
    }
    return minimum;
}

void SetupFilterPhaseAxes()
{
    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
    ImPlot::SetupAxis(ImAxis_Y1, "Phase (rad)");
    ImPlot::SetupAxisLimits(ImAxis_Y1, -std::numbers::pi, std::numbers::pi, ImPlotCond_Once);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##0 rad", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##+pi", std::numbers::pi, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator), 0.6f);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##-pi", -std::numbers::pi, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator), 0.6f);
}

void DrawAttenuationMagnitude(const fdn_analysis::FilterData& data, float minimum_magnitude)
{
    if (!ImPlot::BeginPlot("Magnitude", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f), ImPlotFlags_None))
    {
        return;
    }
    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
    ImPlot::SetupAxis(ImAxis_Y1, "Magnitude (dB)");
    ImPlot::SetupAxisLimits(ImAxis_Y1, std::min(-60.0f, minimum_magnitude * 1.2f), 5.0, ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 20.0);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##0 dB", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
    if (data.mag_responses.empty() || data.frequency_bins.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No attenuation-filter response available");
    }
    for (size_t i = 0; i < data.mag_responses.size(); ++i)
    {
        const auto response = data.mag_responses[i];
        const std::string name = "Delay filter " + std::to_string(i + 1);
        ImPlotSpec spec{};
        spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(i);
        spec.LineWeight = 1.4f;
        ImPlot::PlotLine(name.c_str(), data.frequency_bins.data(), response.data(),
                         std::min(data.frequency_bins.size(), response.size()), spec);
    }
    ImPlot::EndPlot();
}

void DrawAttenuationPhase(const fdn_analysis::FilterData& data)
{
    if (!ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_None))
    {
        return;
    }
    ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
    SetupFilterPhaseAxes();
    if (data.phase_responses.empty() || data.frequency_bins.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No attenuation-filter phase available");
    }
    for (size_t i = 0; i < data.phase_responses.size(); ++i)
    {
        const auto response = data.phase_responses[i];
        const std::string name = "Delay filter " + std::to_string(i + 1);
        ImPlotSpec spec{};
        spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(i);
        spec.LineWeight = 1.4f;
        ImPlot::PlotLine(name.c_str(), data.frequency_bins.data(), response.data(),
                         std::min(data.frequency_bins.size(), response.size()), spec);
    }
    ImPlot::EndPlot();
}

void DrawAttenuationFilterTab(const fdn_analysis::FilterData& data, float minimum_magnitude)
{
    if (!ImGui::BeginTabItem("Attenuation filters"))
    {
        return;
    }
    if (ImPlot::BeginSubplots("Attenuation Filters Mag/Phase", 2, 1, ImVec2(-1, -1), ImPlotSubplotFlags_LinkAllX))
    {
        DrawAttenuationMagnitude(data, minimum_magnitude);
        DrawAttenuationPhase(data);
        ImPlot::EndSubplots();
    }
    ImGui::EndTabItem();
}

void DrawToneCorrectionMagnitude(const fdn_analysis::FilterData& data)
{
    if (!ImPlot::BeginPlot("Magnitude", ImVec2(-1, ImGui::GetCurrentWindow()->Size[1] * 0.5f), ImPlotFlags_NoLegend))
    {
        return;
    }
    fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
    ImPlot::SetupAxis(ImAxis_Y1, "Magnitude (dB)");
    ImPlot::SetupAxisLimits(ImAxis_Y1, -60.0, 10.0, ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 30.0);
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##0 dB", 0.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary), 0.75f);
    if (data.tc_mag_response.empty() || data.frequency_bins.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No tone-correction response available");
    }
    else
    {
        ImPlot::PlotLine("TC Filter", data.frequency_bins.data(), data.tc_mag_response.data(),
                         std::min(data.frequency_bins.size(), data.tc_mag_response.size()),
                         fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
    }
    ImPlot::EndPlot();
}

void DrawToneCorrectionPhase(const fdn_analysis::FilterData& data)
{
    if (!ImPlot::BeginPlot("Phase", ImVec2(-1, -1), ImPlotFlags_NoLegend))
    {
        return;
    }
    SetupFilterPhaseAxes();
    if (data.tc_phase_response.empty() || data.frequency_bins.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No tone-correction phase available");
    }
    else
    {
        ImPlot::PlotLine("TC Filter", data.frequency_bins.data(), data.tc_phase_response.data(),
                         std::min(data.frequency_bins.size(), data.tc_phase_response.size()),
                         fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
    }
    ImPlot::EndPlot();
}

void DrawToneCorrectionTab(const fdn_analysis::FilterData& data)
{
    if (!ImGui::BeginTabItem("Tone Correction Filter"))
    {
        return;
    }
    if (ImPlot::BeginSubplots("Tone Correction Filters Mag/Phase", 2, 1, ImVec2(-1, -1), ImPlotSubplotFlags_LinkAllX))
    {
        DrawToneCorrectionMagnitude(data);
        DrawToneCorrectionPhase(data);
        ImPlot::EndSubplots();
    }
    ImGui::EndTabItem();
}
} // namespace

void FDNToolboxApp::DrawFilterResponse()
{
    if (!ImGui::Begin("Filter Response"))
    {
        ImGui::End();
        return;
    }

    if (ImGui::BeginTabBar("Filter Responses"))
    {
        const fdn_analysis::FilterData data = fdn_analyzer_.GetFilterData();
        DrawAttenuationFilterTab(data, FindMinimumMagnitude(data));
        DrawToneCorrectionTab(data);
        ImGui::EndTabBar();
    }
    ImGui::End();
}

namespace
{
constexpr std::array<const char*, 9> kOctaveBandNames = {"63 Hz", "125 Hz", "250 Hz", "500 Hz", "1 kHz",
                                                         "2 kHz", "4 kHz",  "8 kHz",  "16 kHz"};

struct EnergyDecayUiState
{
    float decay_db_start = 5.0f;
    float decay_db_end = 25.0f;
    bool show_rir = false;
    std::array<bool, kOctaveBandNames.size()> octave_band_visibility{};
};

bool DrawEnergyDecayControls(EnergyDecayUiState& state, bool has_rir)
{
    ImGui::DragFloatRange2("T60 decay range", &state.decay_db_start, &state.decay_db_end, 1.0, 0, 60);
    state.decay_db_start = std::clamp(state.decay_db_start, 0.0f, state.decay_db_end - 1.0f);
    state.decay_db_end = std::clamp(state.decay_db_end, state.decay_db_start + 1.0f, 60.0f);

    if (has_rir)
    {
        ImGui::Checkbox("Show RIR", &state.show_rir);
    }
    else
    {
        state.show_rir = false;
    }

    for (size_t i = 0; i < state.octave_band_visibility.size(); ++i)
    {
        ImGui::Checkbox(kOctaveBandNames[i], &state.octave_band_visibility[i]);
        if (i + 1 < state.octave_band_visibility.size())
        {
            ImGui::SameLine();
        }
    }
    return std::ranges::any_of(state.octave_band_visibility, [](bool visible) { return visible; });
}

void DrawWidebandEnergyDecay(const fdn_analysis::EnergyDecayCurveData& fdn_data,
                             const fdn_analysis::EnergyDecayCurveData& rir_data, std::span<const float> fdn_time,
                             std::span<const float> rir_time, bool show_rir)
{
    const size_t fdn_count = std::min(fdn_time.size(), fdn_data.energy_decay_curve.size());
    ImPlot::PlotLine("FDN EDC", fdn_time.data(), fdn_data.energy_decay_curve.data(), fdn_count,
                     fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
    if (show_rir && !rir_data.energy_decay_curve.empty())
    {
        const size_t rir_count = std::min(rir_time.size(), rir_data.energy_decay_curve.size());
        ImPlot::PlotLine("Reference RIR EDC", rir_time.data(), rir_data.energy_decay_curve.data(), rir_count,
                         fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.3f));
    }
}

void DrawRirOctaveEnergyDecay(size_t band, const fdn_analysis::EnergyDecayCurveData& data, std::span<const float> time,
                              ImPlotSpec spec)
{
    if (data.edc_octaves[band].empty())
    {
        return;
    }
    const size_t point_count = std::min(time.size(), data.edc_octaves[band].size());
    if (point_count == 0)
    {
        return;
    }
    const size_t stride = std::max<size_t>(1, point_count / 2048);
    spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(band, 0.55f);
    spec.LineWeight = 1.0f;
    spec.Stride = static_cast<int>(stride * sizeof(float));
    const std::string label = "RIR " + std::string(kOctaveBandNames[band]);
    ImPlot::PlotLine(label.c_str(), time.data(), data.edc_octaves[band].data(),
                     static_cast<int>(((point_count - 1) / stride) + 1), spec);
}

void DrawOctaveEnergyDecay(const EnergyDecayUiState& state, const fdn_analysis::EnergyDecayCurveData& fdn_data,
                           const fdn_analysis::EnergyDecayCurveData& rir_data, std::span<const float> fdn_time,
                           std::span<const float> rir_time)
{
    for (size_t band = 0; band < kOctaveBandNames.size(); ++band)
    {
        if (!state.octave_band_visibility[band] || fdn_data.edc_octaves[band].empty())
        {
            continue;
        }
        const size_t point_count = std::min(fdn_time.size(), fdn_data.edc_octaves[band].size());
        const size_t stride = std::max<size_t>(1, point_count / 2048);
        ImPlotSpec spec{};
        spec.LineColor = fdn_sandbox::plot_ui::OctaveBandColor(band);
        spec.LineWeight = 1.5f;
        spec.Stride = static_cast<int>(stride * sizeof(float));
        ImPlot::PlotLine(kOctaveBandNames[band], fdn_time.data(), fdn_data.edc_octaves[band].data(),
                         static_cast<int>(((point_count - 1) / stride) + 1), spec);
        if (state.show_rir)
        {
            DrawRirOctaveEnergyDecay(band, rir_data, rir_time, spec);
        }
    }
}

void DrawEnergyDecayPlot(const EnergyDecayUiState& state, bool show_octave_bands,
                         const fdn_analysis::EnergyDecayCurveData& fdn_data,
                         const fdn_analysis::EnergyDecayCurveData& rir_data, const fdn_analysis::T60Data& t60_data,
                         std::span<const float> fdn_time, std::span<const float> rir_time)
{
    const std::string title = std::format("Energy Decay Curve (T60: {:.2f} s)", t60_data.overall_t60.t60);
    if (!ImPlot::BeginPlot(title.c_str(), ImVec2(-1, -1), ImPlotFlags_None))
    {
        return;
    }
    ImPlot::SetupLegend(ImPlotLocation_NorthEast,
                        show_octave_bands ? ImPlotLegendFlags_Outside : ImPlotLegendFlags_None);
    ImPlot::SetupAxes("Time (s)", "Level (dB)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -90.0f, 5.0f, ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -120.0, 10.0);

    if (fdn_data.energy_decay_curve.empty() || fdn_time.empty())
    {
        fdn_sandbox::plot_ui::DrawEmptyState("No energy-decay data available");
        ImPlot::EndPlot();
        return;
    }

    ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, fdn_time.back(), ImPlotCond_Once);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, fdn_time.back());
    fdn_sandbox::plot_ui::DrawHorizontalGuide(
        "##-60 dB", -60.0, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Separator), 0.75f);
    if (std::isfinite(t60_data.overall_t60.t60) && t60_data.overall_t60.t60 > 0.0f)
    {
        ImPlot::Annotation(t60_data.overall_t60.t60, -60.0,
                           fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent), ImVec2(8.0f, -8.0f), true,
                           "T60 %.2f s", t60_data.overall_t60.t60);
    }

    if (show_octave_bands)
    {
        DrawOctaveEnergyDecay(state, fdn_data, rir_data, fdn_time, rir_time);
    }
    else
    {
        DrawWidebandEnergyDecay(fdn_data, rir_data, fdn_time, rir_time, state.show_rir);
    }
    ImPlot::EndPlot();
}
} // namespace

void FDNToolboxApp::DrawEnergyDecayCurve()
{
    if (!ImGui::Begin("Energy Decay Curve"))
    {
        ImGui::End();
        return;
    }

    static EnergyDecayUiState state;
    const bool show_octave_bands = DrawEnergyDecayControls(state, !rir_analyzer_.GetImpulseResponse().empty());
    const fdn_analysis::EnergyDecayCurveData fdn_data = fdn_analyzer_.GetEnergyDecayCurveData();
    const fdn_analysis::EnergyDecayCurveData rir_data =
        state.show_rir ? rir_analyzer_.GetEnergyDecayCurveData() : fdn_analysis::EnergyDecayCurveData{};
    const fdn_analysis::T60Data t60_data = fdn_analyzer_.GetT60Data(-state.decay_db_start, -state.decay_db_end);
    const std::span<const float> rir_time = state.show_rir ? rir_analyzer_.GetTimeData() : std::span<const float>{};
    DrawEnergyDecayPlot(state, show_octave_bands, fdn_data, rir_data, t60_data, fdn_analyzer_.GetTimeData(), rir_time);
    ImGui::End();
}
void FDNToolboxApp::DrawEnergyDecayRelief()
{

    if (!ImGui::Begin("Energy Decay Relief"))
    {
        ImGui::End();
        return;
    }

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    fdn_analysis::EnergyDecayReliefData edr{};
    if (show_rir)
    {
        edr = rir_analyzer_.GetEnergyDecayReliefData();
    }
    else
    {
        edr = fdn_analyzer_.GetEnergyDecayReliefData();
    }

    if (edr.energy_decay_relief.empty() || edr.bin_count == 0 || edr.frame_count == 0)
    {
        fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Description,
                                 fdn_sandbox::theme::ColorRole::TextSecondary,
                                 "No energy-decay-relief data available.");
        ImGui::End();
        return;
    }

    fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Metadata, fdn_sandbox::theme::ColorRole::TextSecondary,
                             "Drag to rotate | Scroll to zoom");

    static std::vector<float> x_data;
    static std::vector<float> y_data;

    // constexpr uint32_t kDownsampleFactor = 1024;
    const uint32_t y_size = edr.bin_count;
    const uint32_t x_size = edr.frame_count;

    const size_t grid_size = x_size * y_size;
    if (edr.energy_decay_relief.size() < grid_size)
    {
        fdn_sandbox::theme::Text(fdn_sandbox::theme::FontRole::Description, fdn_sandbox::theme::ColorRole::StatusError,
                                 "Energy-decay-relief data dimensions are invalid.");
        ImGui::End();
        return;
    }

    if (x_data.size() != grid_size || y_data.size() != grid_size)
    {
        x_data.resize(grid_size);
        y_data.resize(grid_size);

        auto x_mdspan = utils::Span2D(x_data, y_size, x_size);
        auto y_mdspan = utils::Span2D(y_data, y_size, x_size);
        for (size_t i = 0; i < y_size; ++i)
        {
            for (size_t j = 0; j < x_size; ++j)
            {
                x_mdspan(i, j) = static_cast<float>(j * edr.hop_size) /
                                 static_cast<float>(Settings::Instance().SampleRate()); // Time in seconds
                y_mdspan(i, j) = static_cast<float>(i);                                 // Mel band index
            }
        }
    }

    static std::vector<float> z_data;
    z_data.resize(grid_size);
    auto z_mdspan = utils::Span2D(z_data, y_size, x_size);
    for (size_t i = 0; i < y_size; ++i)
    {
        for (size_t j = 0; j < x_size; ++j)
        {
            z_mdspan(i, j) = edr.energy_decay_relief[i + (j * y_size)];
            if (z_mdspan(i, j) < -80.0f)
            {
                z_mdspan(i, j) = -80.0f; // Clamp to -80 dB for better visualization
            }
        }
    }

    if (ImPlot3D::BeginPlot("Energy Decay Relief", ImVec2(-1, -1), ImPlot3DFlags_None))
    {
        ImPlot3D::PushColormap(ImPlotColormap_Viridis);
        ImPlot3D::PushStyleVar(ImPlot3DStyleVar_FillAlpha, 0.8f);
        ImPlot3D::SetupBoxScale(2.0, 1.0, 0.8);
        ImPlot3DSpec spec{};
        spec.Flags = ImPlot3DSurfaceFlags_None;
        spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary);
        ImPlot3D::SetupAxes("Time (s)", "Mel Band", "Level (dB)", ImPlot3DAxisFlags_AutoFit, ImPlot3DAxisFlags_AutoFit,
                            ImPlot3DAxisFlags_AutoFit);

        ImPlot3D::PlotSurface("EDR Surface", x_data.data(), y_data.data(), z_data.data(), x_size, y_size, 0.0, 0.0,
                              spec);

        ImPlot3D::PopStyleVar();
        ImPlot3D::PopColormap();
        ImPlot3D::EndPlot();
    }

    ImGui::End();
}
void FDNToolboxApp::DrawCepstrum()
{
    if (!ImGui::Begin("Cepstrum"))
    {
        ImGui::End();
        return;
    }
    static double early_rir_duration = 0.5; // 500 ms

    static std::array row_ratios = {0.2f, 0.8f};
    if (ImPlot::BeginSubplots("Cepstrum Subplot", 2, 1, ImVec2(-1, -1), ImPlotFlags_NoLegend, row_ratios.data()))
    {
        if (ImPlot::BeginPlot("Impulse Response Cepstrum#", ImVec2(), ImPlotFlags_NoLegend))
        {
            DrawEarlyRIRPicker(fdn_analyzer_.GetImpulseResponse(), fdn_analyzer_.GetTimeData(), early_rir_duration);
            early_rir_duration = std::max(early_rir_duration, 0.1);
            ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("Cepstrum", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            auto cepstrum_data = fdn_analyzer_.GetCepstrum(early_rir_duration);
            static std::vector<float> quefrency_ms;
            if (quefrency_ms.size() != cepstrum_data.cepstrum.size())
            {
                quefrency_ms.resize(cepstrum_data.cepstrum.size());
                for (size_t i = 0; i < quefrency_ms.size(); ++i)
                {
                    quefrency_ms[i] =
                        static_cast<float>(i) / static_cast<float>(Settings::Instance().SampleRate()) * 1000.0f;
                }
            }

            ImPlot::SetupAxes("Quefrency (ms)", "Cepstral Amplitude", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            if (cepstrum_data.cepstrum.empty() || quefrency_ms.empty())
            {
                fdn_sandbox::plot_ui::DrawEmptyState("No cepstrum data available");
            }
            else
            {
                ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, quefrency_ms.back(), ImPlotCond_Once);
                ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, quefrency_ms.back());
                ImPlot::PlotLine("Cepstrum", quefrency_ms.data(), cepstrum_data.cepstrum.data(),
                                 cepstrum_data.cepstrum.size(),
                                 fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn));
            }

            ImPlot::EndPlot();
        }
        ImPlot::EndSubplots();
    }

    ImGui::End();
}
void FDNToolboxApp::DrawEchoDensity()
{
    if (!ImGui::Begin("Echo Density"))
    {
        ImGui::End();
        return;
    }

    static int window_size_ms = 25;
    ImGui::SetNextItemWidth(200);
    ImGui::InputInt("Window Size (ms)", &window_size_ms, 1, 10);
    window_size_ms = std::clamp(window_size_ms, 5, 250); // Clamp to a reasonable range

    static int hop_size_ms = 10;
    ImGui::SetNextItemWidth(200);
    ImGui::InputInt("Hop Size (ms)", &hop_size_ms, 1, 10);
    hop_size_ms = std::clamp(hop_size_ms, 1, window_size_ms); // Clamp to a reasonable range

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    auto echo_density_data = fdn_analyzer_.GetEchoDensityData(window_size_ms, hop_size_ms);
    auto ir = fdn_analyzer_.GetImpulseResponse();
    auto time_data = fdn_analyzer_.GetTimeData();
    const ImPlotFlags plot_flags = show_rir ? ImPlotFlags_None : ImPlotFlags_NoLegend;

    if (ImPlot::BeginPlot("Echo Density", ImVec2(-1, -1), plot_flags))
    {
        ImPlot::SetupAxes("Time (s)", "Normalized Value", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.5f, ImPlotCond_Once);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 2.0f);

        if (ir.empty() || time_data.empty() || echo_density_data.echo_density.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No echo-density data available");
        }
        else
        {
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0f, time_data.back(), ImPlotCond_Once);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.0, time_data.back());
            ImPlotSpec ir_spec{};
            ir_spec.LineColor = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::TextSecondary);
            ir_spec.LineWeight = 0.8f;
            ir_spec.FillAlpha = 0.45f;
            ImPlot::PlotLine("FDN Impulse Response", time_data.data(), ir.data(), std::min(time_data.size(), ir.size()),
                             ir_spec);
            const size_t echo_density_count =
                std::min(echo_density_data.sparse_indices.size(), echo_density_data.echo_density.size());
            ImPlot::PlotLine("FDN Echo Density", echo_density_data.sparse_indices.data(),
                             echo_density_data.echo_density.data(), echo_density_count,
                             fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn, 1.8f));

            if (echo_density_data.mixing_time < time_data.back())
            {
                fdn_sandbox::plot_ui::DrawVerticalGuide(
                    "FDN Mixing Time", echo_density_data.mixing_time,
                    fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotFdn), 1.5f);
                ImPlot::Annotation(echo_density_data.mixing_time, 1.25,
                                   fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotFdn),
                                   ImVec2(6.0f, 0.0f), true, "FDN %.0f ms", echo_density_data.mixing_time * 1000.0f);
            }

            if (show_rir)
            {
                auto rir_echo_density_data = rir_analyzer_.GetEchoDensityData(window_size_ms, hop_size_ms);
                if (!rir_echo_density_data.echo_density.empty())
                {
                    const size_t rir_echo_density_count = std::min(rir_echo_density_data.sparse_indices.size(),
                                                                   rir_echo_density_data.echo_density.size());
                    ImPlot::PlotLine("RIR Echo Density", rir_echo_density_data.sparse_indices.data(),
                                     rir_echo_density_data.echo_density.data(), rir_echo_density_count,
                                     fdn_sandbox::plot_ui::LineSpec(fdn_sandbox::plot_ui::SeriesRole::Reference, 1.5f));

                    if (rir_echo_density_data.mixing_time < time_data.back())
                    {
                        fdn_sandbox::plot_ui::DrawVerticalGuide(
                            "RIR Mixing Time", rir_echo_density_data.mixing_time,
                            fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotReference), 1.3f);
                        ImPlot::Annotation(rir_echo_density_data.mixing_time, 1.05,
                                           fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::PlotReference),
                                           ImVec2(6.0f, 0.0f), true, "RIR %.0f ms",
                                           rir_echo_density_data.mixing_time * 1000.0f);
                    }
                }
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::End();
}
void FDNToolboxApp::DrawT60s()
{
    if (!ImGui::Begin("RT60s"))
    {
        ImGui::End();
        return;
    }

    const float db_start = -5;
    const float db_end = -50;

    static bool show_rir = false;
    ImGui::Checkbox("Show RIR", &show_rir);

    auto t60_data = fdn_analyzer_.GetT60Data(db_start, db_end);
    ImGui::Text("Wideband T60: %.2f s", t60_data.overall_t60.t60);
    const ImPlotFlags plot_flags = show_rir ? ImPlotFlags_None : ImPlotFlags_NoLegend;
    if (ImPlot::BeginPlot("RT60s", ImVec2(-1, -1), plot_flags))
    {
        fdn_sandbox::plot_ui::SetupFrequencyAxis(ImAxis_X1, true, Settings::Instance().SampleRate() / 2.0);
        ImPlot::SetupAxis(ImAxis_Y1, "RT60 (s)");
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 3.0f, ImPlotCond_Once);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0.0f, 50.f);
        ImPlot::SetupAxisZoomConstraints(ImAxis_Y1, 0.0f, 50.f);

        if (t60_data.t60_octaves.empty() || t60_data.octave_band_frequencies.empty())
        {
            fdn_sandbox::plot_ui::DrawEmptyState("No RT60 data available");
        }
        else
        {
            const size_t fdn_t60_count = std::min(t60_data.octave_band_frequencies.size(), t60_data.t60_octaves.size());
            ImPlot::PlotLine(
                "FDN RT60", t60_data.octave_band_frequencies.data(), t60_data.t60_octaves.data(), fdn_t60_count,
                fdn_sandbox::plot_ui::MarkerSpec(fdn_sandbox::plot_ui::SeriesRole::Fdn, ImPlotMarker_Circle, 5.0f));

            if (show_rir)
            {
                auto rir_t60_data = rir_analyzer_.GetT60Data(db_start, db_end);
                if (!rir_t60_data.t60_octaves.empty())
                {
                    const size_t rir_t60_count =
                        std::min(rir_t60_data.octave_band_frequencies.size(), rir_t60_data.t60_octaves.size());
                    ImPlot::PlotLine("RIR RT60", rir_t60_data.octave_band_frequencies.data(),
                                     rir_t60_data.t60_octaves.data(), rir_t60_count,
                                     fdn_sandbox::plot_ui::MarkerSpec(fdn_sandbox::plot_ui::SeriesRole::Reference,
                                                                      ImPlotMarker_Square, 5.0f));
                }
            }
        }

        ImPlot::EndPlot();
    }
    ImGui::End();
}
