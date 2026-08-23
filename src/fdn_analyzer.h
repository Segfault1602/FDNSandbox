#pragma once

#include <bitset>
#include <cstdint>
#include <memory>
#include <vector>

#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include <audio_utils/audio_analysis.h>
#include <sffdn/sffdn.h>

#include "ir_analyzer.h"

namespace fdn_analysis
{

// Result spans borrow analyzer-owned storage and must be reacquired after recomputation.
struct FilterData
{
    std::vector<std::span<const float>> mag_responses;
    std::vector<std::span<const float>> phase_responses;

    std::span<const float> tc_mag_response;
    std::span<const float> tc_phase_response;

    std::span<const float> frequency_bins;
};

enum class FDNAnalysisType : uint8_t
{
    ImpulseResponse,
    FilterResponse
};
inline constexpr size_t kFDNAnalysisTypeCount = 2;

class FDNAnalyzer
{
  public:
    FDNAnalyzer(uint32_t samplerate, quill::Logger* logger);

    void SetFDN(std::unique_ptr<sfFDN::FDN> fdn);

    void SetImpulseResponseSize(uint32_t size_samples);
    [[nodiscard]] uint32_t GetImpulseResponseSize() const;

    std::span<const float> GetImpulseResponse();
    [[nodiscard]] bool IsClipping() const;
    std::span<const float> GetTimeData();

    SpectrogramData GetSpectrogram(audio_utils::analysis::STFTOptions stft_options, bool mel_scale = false);

    SpectrumData GetSpectrum(float early_rir_time);

    CepstrumData GetCepstrum(float early_rir_time);

    AutocorrelationData GetAutocorrelation(float early_rir_time);

    FilterData GetFilterData();

    EnergyDecayCurveData GetEnergyDecayCurveData();

    EnergyDecayReliefData GetEnergyDecayReliefData();

    T60Data GetT60Data(float decay_db_start, float decay_db_end);

    EchoDensityData GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms);

    void Invalidate(IRAnalysisType type)
    {
        ir_analyzer_.Invalidate(type);
    }

  private:
    quill::Logger* logger_;
    std::unique_ptr<sfFDN::FDN> fdn_;
    uint32_t samplerate_;

    IRAnalyzer ir_analyzer_;

    uint32_t impulse_response_size_samples_{0};
    bool is_clipping_{false};

    std::vector<std::vector<float>> filter_mag_responses_;
    std::vector<std::vector<float>> filter_phase_responses_;
    std::vector<float> tc_filter_mag_response_;
    std::vector<float> tc_filter_phase_response_;
    std::vector<float> filter_freq_bins_;

    std::bitset<kFDNAnalysisTypeCount> analysis_flags_ = 0;
};
} // namespace fdn_analysis