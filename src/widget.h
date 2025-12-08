#pragma once

#include <cstdint>
#include <span>

#include "app.h"

#include <sffdn/sffdn.h>

void DrawInputOutputGainsPlot(const sfFDN::FDNConfig& config, sfFDN::FDN* fdn);
void DrawDelaysPlot(const sfFDN::FDNConfig& config, uint32_t max_delay);
void DrawFeedbackMatrixPlot(const sfFDN::FDNConfig& config);

bool DrawInputGainsWidget(sfFDN::FDNConfig& config);
bool DrawOutputGainsWidget(sfFDN::FDNConfig& config);
bool DrawDelayLengthsWidget(sfFDN::FDNConfig& config, int& min_delay, int& max_delay, uint32_t random_seed);

bool DrawScalarMatrixWidget(sfFDN::FDNConfig& config, uint32_t random_seed);
bool DrawDelayFilterWidget(sfFDN::FDNConfig& config);
bool DrawToneCorrectionFilterDesigner(sfFDN::FDNConfig& config);

bool DrawEarlyRIRPicker(std::span<const float> impulse_response, std::span<const float> time_data, double& ir_duration);

bool DrawExtraDelayWidget(sfFDN::FDNConfig& config, bool force_update);
bool DrawInputVelvetNoiseDecorrelator(sfFDN::VelvetNoiseDecorrelatorConfig& config, bool force_update);
bool DrawInputVelvetNoiseDecorrelatorMultiChannel(sfFDN::VelvetNoiseDecorrelatorConfig& config,
                                                  uint32_t& selected_sequence, bool force_update);
bool DrawInputSeriesSchroederAllpassWidget(sfFDN::SchroederAllpassConfig& config, bool force_update);
bool DrawExtraSchroederAllpassWidget(sfFDN::SchroederAllpassConfig& config, uint32_t channel_count, bool force_update);
bool DrawTimeVaryingDelayWidget(sfFDN::TimeVaryingDelayConfig& config, uint32_t channel_count, bool force_update);
bool DrawDiffuserWidget(sfFDN::FDNConfig& config, bool force_update);