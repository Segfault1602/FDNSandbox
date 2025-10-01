#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "sffdn/sffdn.h"

enum class DelayFilterType : uint8_t
{
    Proportional = 0,
    OnePole = 1,
    TwoFilter = 2,
};

using matrix_variant_t = std::variant<sfFDN::CascadedFeedbackMatrixInfo, std::vector<float>>;

struct FDNConfig
{
    uint32_t N; // Number of channels
    bool transposed;
    std::vector<float> input_gains;      // Input gains for each channel
    std::vector<float> output_gains;     // Output gains for each channel
    std::vector<uint32_t> delays;        // Delay lengths in samples for each channel
    matrix_variant_t matrix_info;        // Info for feedback matrix
    std::vector<float> attenuation_t60s; // T60 values for attenuation filters
    std::vector<float> tc_gains;         // Tone correction gains for each band
    std::vector<float> tc_frequencies;   // Center frequencies for tone correction bands

    // Extras!
    bool use_extra_delays;
    std::vector<uint32_t> input_stage_delays;
    std::vector<uint32_t> schroeder_allpass_delays;
    std::vector<float> schroeder_allpass_gains;

    static FDNConfig LoadFromFile(const std::string& filename);
    static void SaveToFile(const std::string& filename, const FDNConfig& config);
};

void to_json(nlohmann::json& j, const FDNConfig& p);
void from_json(const nlohmann::json& j, FDNConfig& p);

std::unique_ptr<sfFDN::FDN> CreateFDNFromConfig(const FDNConfig& config, uint32_t samplerate);