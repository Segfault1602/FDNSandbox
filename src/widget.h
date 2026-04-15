#pragma once

#include <cstdint>
#include <span>

#include "app.h"

#include <sffdn/sffdn.h>

void DrawInputOutputGainsPlot(const sfFDN::FDNConfig2& config, sfFDN::FDN* fdn);
void DrawDelaysPlot(const sfFDN::FDNConfig2& config, uint32_t max_delay);
void DrawFeedbackMatrixPlot(const sfFDN::FDNConfig2& config, sfFDN::FDN* fdn);

// bool DrawInputGainsWidget(sfFDN::FDNConfig2& config);
// bool DrawOutputGainsWidget(sfFDN::FDNConfig2& config);
// bool DrawDelayLengthsWidget(sfFDN::FDNConfig2& config, int& min_delay, int& max_delay, uint32_t random_seed);

// bool DrawScalarMatrixWidget(sfFDN::FDNConfig2& config, uint32_t random_seed);
// bool DrawDelayFilterWidget(sfFDN::FDNConfig2& config);
// bool DrawToneCorrectionFilterDesigner(sfFDN::FDNConfig2& config);

bool DrawEarlyRIRPicker(std::span<const float> impulse_response, std::span<const float> time_data, double& ir_duration);

// bool DrawExtraDelayWidget(sfFDN::FDNConfig2& config, bool force_update);
// bool DrawInputVelvetNoiseDecorrelator(sfFDN::VelvetNoiseDecorrelatorConfig& config, bool force_update);
// bool DrawInputVelvetNoiseDecorrelatorMultiChannel(sfFDN::VelvetNoiseDecorrelatorConfig& config,
//                                                   uint32_t& selected_sequence, bool force_update);
// bool DrawInputSeriesSchroederAllpassWidget(sfFDN::SchroederAllpassConfig& config, bool force_update);
// bool DrawExtraSchroederAllpassWidget(sfFDN::SchroederAllpassConfig& config, uint32_t channel_count, bool
// force_update); bool DrawTimeVaryingDelayWidget(sfFDN::TimeVaryingDelayConfig& config, uint32_t channel_count, bool
// force_update); bool DrawDiffuserWidget(sfFDN::FDNConfig2& config, bool force_update);