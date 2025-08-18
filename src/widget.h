#pragma once

#include <cstdint>
#include <span>

#include "app.h"
#include <sffdn/sffdn.h>

void DrawInputOutputGainsPlot(const sfFDN::FDN* fdn);
void DrawDelaysPlot(const sfFDN::FDN* fdn, uint32_t max_delay);
void DrawFeedbackMatrixPlot(const sfFDN::FDN* fdn);

bool DrawInputGainsWidget(sfFDN::FDN* fdn, std::span<float> gains);
bool DrawOutputGainsWidget(sfFDN::FDN* fdn, std::span<float> gains);
bool DrawDelayLengthsWidget(size_t N, std::span<uint32_t> delays, int& min_delay, int& max_delay, uint32_t random_seed,
                            bool refresh);

bool DrawFeedbackMatrixWidget(FDNConfig& fdn_config, uint32_t random_seed, bool refresh);
bool DrawScalarMatrixWidget(FDNConfig& fdn_config, uint32_t random_seed, bool refresh);
bool DrawDelayFilterWidget(FDNConfig& fdn_config);

bool DrawEarlyRIRPicker(std::span<const float> impulse_response, std::span<const float> time_data, double& ir_duration);
