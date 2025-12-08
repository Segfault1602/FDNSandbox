#pragma once

#include <audio_utils/fft.h>

#include <cstdint>
#include <functional>
#include <span>

namespace fdn_optimization
{

struct LossFunction
{
    std::function<float(std::span<const float>)> func;
    float weight;
    std::string name;
};

float SpectralFlatnessLoss(std::span<const float> signal);

float RMSLoss(std::span<const float> signal, float target_rms);

float PowerEnvelopeLoss(std::span<const float> signal, uint32_t window_size, uint32_t hop_size, uint32_t sample_rate);

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate, float target_mixing_time);

float SparsityLoss(std::span<const float> signal);

} // namespace fdn_optimization