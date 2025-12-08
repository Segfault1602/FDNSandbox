#pragma once

#include <bitset>
#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <vector>

#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include <audio_utils/audio_analysis.h>
#include <sffdn/sffdn.h>

#include "analysis.h"

namespace fdn_analysis
{

struct SpectrogramData
{
    std::span<const float> data;
    uint32_t bin_count;
    uint32_t frame_count;
};

struct SpectrumData
{
    std::span<const float> spectrum;
    std::span<const float> frequency_bins;
    std::span<const float> peaks;
    std::span<const float> peaks_freqs;
};

struct CepstrumData
{
    std::span<const float> cepstrum;
};

struct AutocorrelationData
{
    std::span<const float> autocorrelation;
    std::span<const float> spectral_autocorrelation;
};

struct EnergyDecayCurveData
{
    std::span<const float> energy_decay_curve;
    std::array<std::span<const float>, 10> edc_octaves;
};

struct T60Data
{
    fdn_analysis::EstimateT60Results overall_t60;
    std::span<const float> t60_octaves;
    std::span<const float> octave_band_frequencies;
};

struct EchoDensityData
{
    std::span<const float> echo_density;
    std::span<const float> sparse_indices;
    float mixing_time;
};

enum class AnalysisType : uint8_t
{
    ImpulseResponse = 0,
    Spectrogram = 1,
    Spectrum = 2,
    Autocorrelation = 3,
    EnergyDecayCurve = 4,
    Cepstrum = 5,
    EchoDensity = 6,
    T60s = 7,
    AnalysisTypeCount
};

class IRAnalyzer
{
  public:
    IRAnalyzer(uint32_t samplerate, quill::Logger* logger);

    void SetImpulseResponse(std::vector<float>&& ir);
    uint32_t GetImpulseResponseSize() const;

    std::span<const float> GetImpulseResponse();

    bool IsClipping();
    std::span<const float> GetTimeData();

    SpectrogramData GetSpectrogram(audio_utils::analysis::SpectrogramInfo spec_info, bool mel_scale = false);

    SpectrumData GetSpectrum(float early_rir_time);

    CepstrumData GetCepstrum(float early_rir_time);

    AutocorrelationData GetAutocorrelation(float early_rir_time);

    EnergyDecayCurveData GetEnergyDecayCurveData();

    T60Data GetT60Data(float decay_db_start, float decay_db_end);

    EchoDensityData GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms);

    void RequestAnalysis(AnalysisType type)
    {
        analysis_flags_.set(static_cast<size_t>(type));
    }

  private:
    uint32_t samplerate_;
    quill::Logger* logger_;

    std::vector<float> impulse_response_;
    std::vector<float> time_data_;
    bool is_clipping_;

    std::vector<float> spectrogram_data_;
    uint32_t spectrogram_bin_count_;
    uint32_t spectrogram_frame_count_;

    std::vector<float> spectrum_data_;
    std::vector<float> frequency_bins_;
    std::vector<float> spectrum_peaks_;
    std::vector<float> peaks_freqs_;
    float spectrum_early_rir_time_;

    std::vector<float> cepstrum_data_;
    float cepstrum_early_rir_time_;

    std::vector<float> autocorrelation_data_;
    std::vector<float> spectral_autocorrelation_data_;
    float autocorrelation_early_rir_time_;

    std::vector<float> energy_decay_curve_;
    std::array<std::vector<float>, 10> edc_octaves_;
    std::vector<float> octave_band_frequencies_;

    fdn_analysis::EstimateT60Results overall_t60_;
    std::vector<float> t60_octaves_;

    std::vector<float> echo_density_;
    std::vector<float> echo_density_indices_;
    uint32_t echo_density_window_size_ms_;
    uint32_t echo_density_hop_size_ms_;
    float mixing_time_;

    std::bitset<static_cast<size_t>(AnalysisType::AnalysisTypeCount)> analysis_flags_ = 0;
};
} // namespace fdn_analysis