#pragma once

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace fdn_optimization
{

struct LossFunction
{
    std::function<float(std::span<const float>)> func;
    float weight;
    std::string name;
};

float RMS(std::span<const float> signal);

float SpectralFlatnessLoss(std::span<const float> signal);

float RMSLoss(std::span<const float> signal, float target_rms);

float PowerEnvelopeLoss(std::span<const float> signal, uint32_t window_size, uint32_t hop_size, uint32_t sample_rate);

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate, float target_mixing_time);

float SparsityLoss(std::span<const float> signal);

float EDCLoss(std::span<const float> signal,
              const std::array<std::vector<float>, audio_utils::analysis::kNumOctaveBands>& target_relief);

float EDRLoss(std::span<const float> signal, const audio_utils::analysis::EnergyDecayReliefResult& target_edr,
              const audio_utils::analysis::EnergyDecayReliefOptions& options);

} // namespace fdn_optimization