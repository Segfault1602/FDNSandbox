#pragma once

#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include <audio_utils/audio_analysis.h>
#include <sffdn/sffdn.h>

#include <bitset>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace fdn_analysis
{

// Result spans borrow analyzer-owned storage. Reacquire them after source replacement,
// invalidation, or any getter that may recompute the same analysis.
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
    std::array<std::span<const float>, 9> edc_octaves;
};

struct EnergyDecayReliefData
{
    std::span<const float> energy_decay_relief;
    uint32_t bin_count;
    uint32_t frame_count;
    uint32_t hop_size;
};

struct T60Data
{
    audio_utils::analysis::EstimateT60Results overall_t60;
    std::span<const float> t60_octaves;
    std::span<const float> octave_band_frequencies;
};

struct EchoDensityData
{
    std::span<const float> echo_density;
    std::span<const float> sparse_indices;
    float mixing_time;
};

enum class IRAnalysisType : uint8_t
{
    Spectrogram,
    Spectrum,
    Autocorrelation,
    EnergyDecayCurve,
    EnergyDecayRelief,
    Cepstrum,
    EchoDensity,
    T60s
};
inline constexpr size_t kIRAnalysisTypeCount = 8;

class IRAnalyzer
{
  public:
    IRAnalyzer(uint32_t samplerate, quill::Logger* logger);

    void SetImpulseResponse(std::vector<float>&& ir);
    [[nodiscard]] uint32_t GetImpulseResponseSize() const;

    std::span<const float> GetImpulseResponse();

    [[nodiscard]] bool IsClipping() const;
    std::span<const float> GetTimeData();

    SpectrogramData GetSpectrogram(audio_utils::analysis::STFTOptions stft_options, bool mel_scale = false);

    SpectrumData GetSpectrum(float early_rir_time);

    CepstrumData GetCepstrum(float early_rir_time);

    AutocorrelationData GetAutocorrelation(float early_rir_time);

    EnergyDecayCurveData GetEnergyDecayCurveData();

    EnergyDecayReliefData GetEnergyDecayReliefData();

    T60Data GetT60Data(float decay_db_start, float decay_db_end);

    EchoDensityData GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms);

    void Invalidate(IRAnalysisType type);

  private:
    struct T60CacheKey
    {
        float decay_db_start;
        float decay_db_end;
    };

    uint32_t samplerate_;
    quill::Logger* logger_;

    std::vector<float> impulse_response_;
    std::vector<float> time_data_;
    bool is_clipping_{false};

    std::vector<float> spectrogram_data_;
    uint32_t spectrogram_bin_count_{0};
    uint32_t spectrogram_frame_count_{0};
    std::optional<audio_utils::analysis::STFTOptions> spectrogram_options_;
    bool spectrogram_mel_scale_{false};

    std::vector<float> spectrum_data_;
    std::vector<float> frequency_bins_;
    std::vector<float> spectrum_peaks_;
    std::vector<float> peaks_freqs_;
    float spectrum_early_rir_time_{0.5f};

    std::vector<float> cepstrum_data_;
    float cepstrum_early_rir_time_{0.5f};

    std::vector<float> autocorrelation_data_;
    std::vector<float> spectral_autocorrelation_data_;
    float autocorrelation_early_rir_time_{0.5f};

    std::vector<float> energy_decay_curve_;
    std::array<std::vector<float>, 9> edc_octaves_;
    std::vector<float> octave_band_frequencies_;

    std::vector<float> edr_data_;
    uint32_t edr_bin_count_{0};
    uint32_t edr_frame_count_{0};
    uint32_t edr_hop_size_{512};

    audio_utils::analysis::EstimateT60Results overall_t60_;
    std::vector<float> t60_octaves_;
    std::optional<T60CacheKey> t60_cache_key_;

    std::vector<float> echo_density_;
    std::vector<float> echo_density_indices_;
    uint32_t echo_density_window_size_ms_{25};
    uint32_t echo_density_hop_size_ms_{10};
    float mixing_time_{0.0f};

    std::bitset<kIRAnalysisTypeCount> analysis_flags_ = 0;
};
} // namespace fdn_analysis