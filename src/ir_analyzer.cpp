#include "ir_analyzer.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <algorithm>
#include <chrono>

namespace
{
constexpr size_t kSpectrumNFFT = 48000;

struct EarlyIRSampleCountInput
{
    std::chrono::duration<float> duration;
    uint32_t sample_rate;
    size_t impulse_size;
};

size_t ClampEarlyIRSampleCount(const EarlyIRSampleCountInput& input)
{
    const float sample_count = std::max(0.0f, input.duration.count() * static_cast<float>(input.sample_rate));
    return std::min(static_cast<size_t>(sample_count), input.impulse_size);
}

bool SameStftOptions(const audio_utils::analysis::STFTOptions& lhs,
                     const audio_utils::analysis::STFTOptions& rhs)
{
    return lhs.fft_size == rhs.fft_size && lhs.overlap == rhs.overlap && lhs.window_size == rhs.window_size &&
           lhs.window_type == rhs.window_type && lhs.samplerate == rhs.samplerate;
}

struct Stopwatch
{
    Stopwatch()
        : start_time_(std::chrono::high_resolution_clock::now())
    {
    }

    [[nodiscard]] double ElapsedMs() const
    {
        const auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    }

  private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace

namespace fdn_analysis
{
IRAnalyzer::IRAnalyzer(uint32_t samplerate, quill::Logger* logger)
    : samplerate_(samplerate)
    , logger_(logger)
    , overall_t60_(0.0f)
{
}

void IRAnalyzer::SetImpulseResponse(std::vector<float>&& ir)
{
    impulse_response_ = std::move(ir);

    time_data_.resize(impulse_response_.size());
    for (size_t i = 0; i < impulse_response_.size(); ++i)
    {
        time_data_[i] =
            static_cast<float>(i) / static_cast<float>(samplerate_); // Convert sample index to time in seconds
    }

    spectrogram_data_.clear();
    spectrogram_bin_count_ = 0;
    spectrogram_frame_count_ = 0;
    spectrogram_options_.reset();
    spectrum_data_.clear();
    frequency_bins_.clear();
    spectrum_peaks_.clear();
    peaks_freqs_.clear();
    cepstrum_data_.clear();
    autocorrelation_data_.clear();
    spectral_autocorrelation_data_.clear();
    energy_decay_curve_.clear();
    for (auto& octave : edc_octaves_)
    {
        octave.clear();
    }
    octave_band_frequencies_.clear();
    edr_data_.clear();
    edr_bin_count_ = 0;
    edr_frame_count_ = 0;
    overall_t60_ = {};
    t60_octaves_.clear();
    t60_cache_key_.reset();
    echo_density_.clear();
    echo_density_indices_.clear();
    mixing_time_ = 0.0f;
    analysis_flags_.set();
}

void IRAnalyzer::Invalidate(IRAnalysisType type)
{
    analysis_flags_.set(static_cast<size_t>(type));
    if (type == IRAnalysisType::EnergyDecayCurve)
    {
        analysis_flags_.set(static_cast<size_t>(IRAnalysisType::T60s));
    }
}

[[nodiscard]] uint32_t IRAnalyzer::GetImpulseResponseSize() const
{
    return impulse_response_.size();
}

std::span<const float> IRAnalyzer::GetImpulseResponse()
{
    return impulse_response_;
}

std::span<const float> IRAnalyzer::GetTimeData()
{
    return time_data_;
}

[[nodiscard]] bool IRAnalyzer::IsClipping() const
{
    return is_clipping_;
}

SpectrogramData IRAnalyzer::GetSpectrogram(audio_utils::analysis::STFTOptions stft_options, bool mel_scale)
{
    const bool options_changed =
        !spectrogram_options_ || !SameStftOptions(*spectrogram_options_, stft_options) ||
        spectrogram_mel_scale_ != mel_scale;
    if ((analysis_flags_.test(static_cast<size_t>(IRAnalysisType::Spectrogram)) || options_changed) &&
        !GetImpulseResponse().empty())
    {
        const Stopwatch stopwatch;
        constexpr size_t kNMels = 128;
        audio_utils::analysis::STFTResult result;
        if (mel_scale)
        {
            result = audio_utils::analysis::MelSpectrogram(GetImpulseResponse(), stft_options, kNMels, true);
        }
        else
        {
            result = audio_utils::analysis::STFT(GetImpulseResponse(), stft_options, true);
        }
        spectrogram_data_ = std::move(result.data);

        float max_val = *std::ranges::max_element(spectrogram_data_);
        max_val = (max_val < 1e-6f) ? 1.f : max_val; // Prevent log of zero

        for (auto& v : spectrogram_data_)
        {
            v /= max_val;                      // Normalize to [0, 1]
            v = 20.f * std::log10f(v + 1e-6f); // Convert to dB
        }

        float min_val = *std::ranges::min_element(spectrogram_data_);
        max_val = *std::ranges::max_element(spectrogram_data_);
        LOG_INFO(logger_, "Spectrogram min: {:.2f} dB, max: {:.2f} dB", min_val, max_val);

        spectrogram_bin_count_ = result.num_bins;
        spectrogram_frame_count_ = result.num_frames;
        spectrogram_options_ = stft_options;
        spectrogram_mel_scale_ = mel_scale;

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::Spectrogram));
        LOG_INFO(logger_, "Computing spectrogram took {} ms", stopwatch.ElapsedMs());
    }

    return SpectrogramData{
        .data = spectrogram_data_, .bin_count = spectrogram_bin_count_, .frame_count = spectrogram_frame_count_};
}

SpectrumData IRAnalyzer::GetSpectrum(float early_rir_time)
{
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::Spectrum)) ||
        early_rir_time != spectrum_early_rir_time_)
    {
        const Stopwatch stopwatch;
        spectrum_early_rir_time_ = early_rir_time;

        const size_t early_rir_sample_count =
            ClampEarlyIRSampleCount({.duration = std::chrono::duration<float>{early_rir_time},
                                     .sample_rate = samplerate_,
                                     .impulse_size = impulse_response_.size()});
        const auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        auto nfft = std::max(kSpectrumNFFT, early_rir.size());

        nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
        audio_utils::FFT fft(nfft);

        spectrum_data_.resize((nfft / 2) + 1, 0.f);
        constexpr audio_utils::ForwardFFTOptions options{
            .output_type = audio_utils::FFTOutputType::Magnitude,
            .to_db = true,
        };
        fft.ForwardMag(early_rir, spectrum_data_, options);

        // Generate frequency bins
        const auto kNumFrequencyBins = static_cast<size_t>(spectrum_data_.size());
        frequency_bins_.resize(kNumFrequencyBins);
        const auto sample_rate = static_cast<float>(samplerate_);
        for (size_t i = 0; i < kNumFrequencyBins; ++i)
        {
            frequency_bins_[i] = (static_cast<float>(i) * sample_rate) / static_cast<float>(nfft);
        }

        spectrum_peaks_.clear();
        peaks_freqs_.clear();
        // Reserve space for peaks, 25% is probably an overly optimistic estimate
        const auto peak_reserve = static_cast<size_t>(0.25 * static_cast<double>(kNumFrequencyBins));
        spectrum_peaks_.reserve(peak_reserve);
        peaks_freqs_.reserve(peak_reserve);

        for (size_t i = 1; i < kNumFrequencyBins - 1; ++i)
        {
            if (spectrum_data_[i] > spectrum_data_[i - 1] && spectrum_data_[i] > spectrum_data_[i + 1])
            {
                spectrum_peaks_.push_back(spectrum_data_.at(i));
                peaks_freqs_.push_back(frequency_bins_.at(i));
            }
        }

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::Spectrum));
        LOG_INFO(logger_, "Computing spectrum took {} ms", stopwatch.ElapsedMs());

        // Compute Spectral Flatness for fun
        std::vector<float> temp_spectrum;
        temp_spectrum.resize(spectrum_data_.size());
        fft.ForwardMag(early_rir, temp_spectrum, {.output_type = audio_utils::FFTOutputType::Power, .to_db = false});
        float flatness = audio_utils::analysis::SpectralFlatness(temp_spectrum);
        LOG_INFO(logger_, "Spectral Flatness: {:.4f}", flatness);
    }

    assert(spectrum_data_.size() == frequency_bins_.size());
    assert(spectrum_peaks_.size() == peaks_freqs_.size());

    return SpectrumData{
        .spectrum = spectrum_data_,
        .frequency_bins = frequency_bins_,
        .peaks = spectrum_peaks_,
        .peaks_freqs = peaks_freqs_,
    };
}

CepstrumData IRAnalyzer::GetCepstrum(float early_rir_time)
{
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::Cepstrum)) ||
        early_rir_time != cepstrum_early_rir_time_)
    {
        const Stopwatch stopwatch;
        cepstrum_early_rir_time_ = early_rir_time;

        const size_t early_rir_sample_count =
            ClampEarlyIRSampleCount({.duration = std::chrono::duration<float>{early_rir_time},
                                     .sample_rate = samplerate_,
                                     .impulse_size = impulse_response_.size()});
        const auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        auto nfft = std::max(kSpectrumNFFT, early_rir.size());

        nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
        audio_utils::FFT fft(nfft);
        cepstrum_data_.resize(nfft, 0.f);
        fft.RealCepstrum(early_rir, cepstrum_data_);

        // Only keep the first half of the cepstrum (real cepstrum is symmetric)
        cepstrum_data_.resize(nfft / 2);

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::Cepstrum));
        LOG_INFO(logger_, "Computing cepstrum took {} ms", stopwatch.ElapsedMs());
    }

    return CepstrumData{.cepstrum = cepstrum_data_};
}

AutocorrelationData IRAnalyzer::GetAutocorrelation(float early_rir_time)
{
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::Autocorrelation)) ||
        early_rir_time != autocorrelation_early_rir_time_)
    {
        const Stopwatch stopwatch;
        autocorrelation_early_rir_time_ = early_rir_time;

        const size_t early_rir_sample_count =
            ClampEarlyIRSampleCount({.duration = std::chrono::duration<float>{early_rir_time},
                                     .sample_rate = samplerate_,
                                     .impulse_size = impulse_response_.size()});
        const auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);

        autocorrelation_data_ = audio_utils::analysis::Autocorrelation(early_rir, true);

        auto nfft = std::max(kSpectrumNFFT, early_rir.size());
        nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
        audio_utils::FFT fft(nfft);

        std::vector<float> spectrum_data((nfft / 2) + 1, 0.f);
        fft.ForwardMag(early_rir, spectrum_data, {.output_type = audio_utils::FFTOutputType::Power, .to_db = true});

        spectral_autocorrelation_data_ = audio_utils::analysis::Autocorrelation(spectrum_data);

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::Autocorrelation));
        LOG_INFO(logger_, "Computing autocorrelation took {} ms", stopwatch.ElapsedMs());
    }

    return AutocorrelationData{
        .autocorrelation = autocorrelation_data_,
        .spectral_autocorrelation = spectral_autocorrelation_data_,
    };
}

EnergyDecayCurveData IRAnalyzer::GetEnergyDecayCurveData()
{
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::EnergyDecayCurve)))
    {
        const Stopwatch stopwatch;
        energy_decay_curve_ = audio_utils::analysis::EnergyDecayCurve(GetImpulseResponse(), true);
        edc_octaves_ = audio_utils::analysis::EnergyDecayCurve_FilterBank(GetImpulseResponse(), true, samplerate_);
        auto octave_band_frequencies = audio_utils::analysis::GetOctaveBandFrequencies();
        octave_band_frequencies_.resize(octave_band_frequencies.size());
        for (size_t i = 0; i < octave_band_frequencies.size(); ++i)
        {
            octave_band_frequencies_[i] = octave_band_frequencies[i];
        }

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::EnergyDecayCurve));
        LOG_INFO(logger_, "Analyzing energy decay curve took {} ms", stopwatch.ElapsedMs());
    }

    std::array<std::span<const float>, audio_utils::analysis::kNumOctaveBands> edc_octaves;
    for (size_t i = 0; i < edc_octaves.size(); ++i)
    {
        edc_octaves[i] = edc_octaves_[i];
    }

    return EnergyDecayCurveData{
        .energy_decay_curve = energy_decay_curve_,
        .edc_octaves = edc_octaves,
    };
}

EnergyDecayReliefData IRAnalyzer::GetEnergyDecayReliefData()
{
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::EnergyDecayRelief)))
    {
        const Stopwatch stopwatch;

        const audio_utils::analysis::EnergyDecayReliefOptions options{};
        auto edr_data = audio_utils::analysis::EnergyDecayRelief(GetImpulseResponse(), options);
        edr_data_ = std::move(edr_data.data);
        edr_bin_count_ = edr_data.num_bins;
        edr_frame_count_ = edr_data.num_frames;
        edr_hop_size_ = options.hop_size;

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::EnergyDecayRelief));
        LOG_INFO(logger_, "Analyzing energy decay relief took {} ms", stopwatch.ElapsedMs());
    }

    return EnergyDecayReliefData{
        .energy_decay_relief = std::span<const float>(edr_data_),
        .bin_count = edr_bin_count_,
        .frame_count = edr_frame_count_,
        .hop_size = edr_hop_size_,
    };
}

T60Data IRAnalyzer::GetT60Data(float decay_db_start, float decay_db_end)
{
    if (decay_db_start <= decay_db_end)
    {
        throw std::invalid_argument("decay_db_start must be greater than decay_db_end");
    }

    if (GetImpulseResponse().empty())
    {
        return {};
    }

    const bool bounds_changed = !t60_cache_key_ || t60_cache_key_->decay_db_start != decay_db_start ||
                                t60_cache_key_->decay_db_end != decay_db_end;
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::T60s)) || bounds_changed)
    {
        const Stopwatch stopwatch;
        auto edc_data = GetEnergyDecayCurveData();
        auto time_data = GetTimeData();

        audio_utils::analysis::EstimateT60Options options;
        options.decay_start_db = decay_db_start;
        options.decay_end_db = decay_db_end;
        options.use_linear_regression = false;

        overall_t60_ = audio_utils::analysis::EstimateT60(edc_data.energy_decay_curve, time_data, options);

        t60_octaves_.clear();
        t60_octaves_.reserve(edc_data.edc_octaves.size());
        for (const auto& octave : edc_data.edc_octaves)
        {
            auto t60_result = audio_utils::analysis::EstimateT60(octave, time_data, options);
            t60_octaves_.push_back(t60_result.t60);
        }

        t60_cache_key_ = T60CacheKey{.decay_db_start = decay_db_start, .decay_db_end = decay_db_end};
        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::T60s));
        LOG_INFO(logger_, "Analyzing T60 took {} ms", stopwatch.ElapsedMs());
    }

    return T60Data{
        .overall_t60 = overall_t60_, .t60_octaves = t60_octaves_, .octave_band_frequencies = octave_band_frequencies_};
}

EchoDensityData IRAnalyzer::GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms)
{
    if (analysis_flags_.test(static_cast<size_t>(IRAnalysisType::EchoDensity)) ||
        (echo_density_window_size_ms_ != window_size_ms || echo_density_hop_size_ms_ != hop_size_ms))
    {
        const Stopwatch stopwatch;
        echo_density_window_size_ms_ = window_size_ms;
        echo_density_hop_size_ms_ = hop_size_ms;

        audio_utils::analysis::EchoDensityOptions options;
        options.window_size = (window_size_ms * samplerate_) / 1000;
        options.hop_size = (hop_size_ms * samplerate_) / 1000;
        options.sample_rate = samplerate_;

        auto echo_density_result = audio_utils::analysis::EchoDensity(GetImpulseResponse(), options);

        echo_density_ = std::move(echo_density_result.echo_densities);
        echo_density_indices_.assign(echo_density_result.sparse_indices.begin(),
                                     echo_density_result.sparse_indices.end());

        mixing_time_ = echo_density_result.mixing_time;
        // Convert indices to time in seconds
        for (auto& idx : echo_density_indices_)
        {
            idx /= static_cast<float>(samplerate_);
        }

        analysis_flags_.reset(static_cast<size_t>(IRAnalysisType::EchoDensity));
        LOG_INFO(logger_, "Analyzing echo density took {} ms", stopwatch.ElapsedMs());
    }

    return EchoDensityData{
        .echo_density = echo_density_, .sparse_indices = echo_density_indices_, .mixing_time = mixing_time_};
}
} // namespace fdn_analysis