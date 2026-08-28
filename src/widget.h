#pragma once

#include <cstdint>
#include <span>

#include <sffdn/sffdn.h>

#include <array>
#include <vector>

void DrawInputOutputGainsPlot(const sfFDN::FDNConfig& config, sfFDN::FDN* fdn);
void DrawDelaysPlot(const sfFDN::FDNConfig& config, uint32_t max_delay);
void DrawFeedbackMatrixPlot(const sfFDN::FDNConfig& config, sfFDN::FDN* fdn, std::vector<float>& feedback_matrix,
                            uint64_t sample_index = 0);
bool DrawGainsWidget(std::span<float> gains, float& min_gain, float& max_gain, float& set_all_value);

struct ThreeBandDesignerData
{
    std::span<float> t60s;
    std::span<float> frequencies;
};

/// Persistent plotting and drag-control state for the three-band designer.
struct ThreeBandDesignerState
{
    std::array<double, 2> frequencies_draggable = {300.0, 8000.0};
    std::array<double, 3> t60s_draggable = {1.5, 1.0, 0.5};
    std::array<float, 2> t60_range = {0.01f, 5.0f};
    std::vector<float> filter_frequencies;
    std::vector<float> response;
};

/// Persistent plotting and drag-control state for the ten-band filter designer.
struct FilterDesignerState
{
    std::vector<float> frequencies;
    std::vector<float> gains_plot;
    std::vector<float> t60s_plot;
    std::vector<float> frequencies_plot;
    std::vector<float> filter_frequencies;
    std::vector<float> response;
    std::vector<double> frequency_ticks;
    bool show_filter_response = false;
};

bool Draw3BandDesigner(ThreeBandDesignerData data, bool& show_3band_designer, ThreeBandDesignerState& state);
bool DrawFilterDesigner(std::span<float> t60s, bool& show_delay_filter_designer, FilterDesignerState& state);

// bool DrawInputGainsWidget(sfFDN::FDNConfig& config);
// bool DrawOutputGainsWidget(sfFDN::FDNConfig& config);
// bool DrawDelayLengthsWidget(sfFDN::FDNConfig& config, int& min_delay, int& max_delay, uint32_t random_seed);

// bool DrawScalarMatrixWidget(sfFDN::FDNConfig& config, uint32_t random_seed);
// bool DrawDelayFilterWidget(sfFDN::FDNConfig& config);
// bool DrawToneCorrectionFilterDesigner(sfFDN::FDNConfig& config);

bool DrawEarlyRIRPicker(std::span<const float> impulse_response, std::span<const float> time_data, double& ir_duration);

// bool DrawExtraDelayWidget(sfFDN::FDNConfig& config, bool force_update);
// bool DrawInputVelvetNoiseDecorrelator(sfFDN::VelvetNoiseDecorrelatorConfig& config, bool force_update);
// bool DrawInputVelvetNoiseDecorrelatorMultiChannel(sfFDN::VelvetNoiseDecorrelatorConfig& config,
//                                                   uint32_t& selected_sequence, bool force_update);
// bool DrawInputSeriesSchroederAllpassWidget(sfFDN::SchroederAllpassConfig& config, bool force_update);
// bool DrawExtraSchroederAllpassWidget(sfFDN::SchroederAllpassConfig& config, uint32_t channel_count,
//                                      bool force_update);
// bool DrawDiffuserWidget(sfFDN::FDNConfig& config, bool force_update);