#pragma once

#include <cstdint>
#include <span>

#include "app.h"
#include "fdn_config.h"

#include <sffdn/sffdn.h>

void DrawInputOutputGainsPlot(const FDNConfig& config);
void DrawDelaysPlot(const FDNConfig& config, uint32_t max_delay);
void DrawFeedbackMatrixPlot(const FDNConfig& config);

bool DrawInputGainsWidget(FDNConfig& config);
bool DrawOutputGainsWidget(FDNConfig& config);
bool DrawDelayLengthsWidget(FDNConfig& config, int& min_delay, int& max_delay, uint32_t random_seed);

bool DrawScalarMatrixWidget(FDNConfig& config, uint32_t random_seed);
bool DrawDelayFilterWidget(FDNConfig& config);
bool DrawToneCorrectionFilterDesigner(FDNConfig& config);

bool DrawEarlyRIRPicker(std::span<const float> impulse_response, std::span<const float> time_data, double& ir_duration);

bool DrawExtraDelayWidget(FDNConfig& config, bool force_update);
bool DrawExtraSchroederAllpassWidget(FDNConfig& config, bool force_update);