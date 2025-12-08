#include "ir_analyzer.h"

#include "analysis.h"
#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <algorithm>
#include <chrono>
#include <numbers>

namespace
{
constexpr size_t kSpectrumNFFT = 48000;

struct Stopwatch
{
    Stopwatch()
        : start_time_(std::chrono::high_resolution_clock::now())
    {
    }

    double ElapsedMs() const
    {
        auto end_time = std::chrono::high_resolution_clock::now();
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
    , is_clipping_(false)
    , spectrogram_bin_count_(0)
    , spectrogram_frame_count_(0)
    , spectrum_early_rir_time_(0.5f)
    , cepstrum_early_rir_time_(0.5f)
    , autocorrelation_early_rir_time_(0.5f)
    , overall_t60_(0.0f)
    , echo_density_window_size_ms_(25)
    , echo_density_hop_size_ms_(10)
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

    analysis_flags_.set();
}

uint32_t IRAnalyzer::GetImpulseResponseSize() const
{
    return impulse_response_.size();
}

std::span<const float> IRAnalyzer::GetImpulseResponse()
{
    analysis_flags_.reset(static_cast<size_t>(AnalysisType::ImpulseResponse));
    return impulse_response_;
}

std::span<const float> IRAnalyzer::GetTimeData()
{
    return time_data_;
}

bool IRAnalyzer::IsClipping()
{
    return is_clipping_;
}

SpectrogramData IRAnalyzer::GetSpectrogram(audio_utils::analysis::SpectrogramInfo spec_info, bool mel_scale)
{
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::Spectrogram)))
    {
        Stopwatch stopwatch;
        constexpr size_t kNMels = 128;
        audio_utils::analysis::SpectrogramResult result;
        if (mel_scale)
        {
            result = audio_utils::analysis::MelSpectrogram(GetImpulseResponse(), spec_info, kNMels, true);
        }
        else
        {
            result = audio_utils::analysis::STFT(GetImpulseResponse(), spec_info, true);
        }
        spectrogram_data_ = std::move(result.data);

        float max_val = *std::ranges::max_element(spectrogram_data_);
        max_val = (max_val < 1e-6f) ? 1.f : max_val; // Prevent log of zero

        for (auto& v : spectrogram_data_)
        {
            v /= max_val; // Normalize to [0, 1]

            v = 20.f * std::log10f(v + 1e-6f); // Convert to dB
        }

        float min_val = *std::ranges::min_element(spectrogram_data_);
        max_val = *std::ranges::max_element(spectrogram_data_);
        LOG_INFO(logger_, "Spectrogram min: {:.2f} dB, max: {:.2f} dB", min_val, max_val);

        spectrogram_bin_count_ = result.num_bins;
        spectrogram_frame_count_ = result.num_frames;

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::Spectrogram));
        LOG_INFO(logger_, "Computing spectrogram took {} ms", stopwatch.ElapsedMs());
    }

    return SpectrogramData{
        .data = spectrogram_data_, .bin_count = spectrogram_bin_count_, .frame_count = spectrogram_frame_count_};
}

SpectrumData IRAnalyzer::GetSpectrum(float early_rir_time)
{
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::Spectrum)) || early_rir_time != spectrum_early_rir_time_)
    {
        Stopwatch stopwatch;
        spectrum_early_rir_time_ = early_rir_time;

        uint32_t early_rir_sample_count = static_cast<uint32_t>(early_rir_time * samplerate_);
        early_rir_sample_count = std::min(early_rir_sample_count, static_cast<uint32_t>(impulse_response_.size()));
        auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        uint32_t nfft = std::max(kSpectrumNFFT, early_rir.size());

        nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
        audio_utils::FFT fft(nfft);

        spectrum_data_.resize((nfft / 2) + 1, 0.f);
        fft.ForwardAbs(early_rir, spectrum_data_, true, true);

        // Generate frequency bins
        const uint32_t kNumFrequencyBins = spectrum_data_.size();
        frequency_bins_.resize(kNumFrequencyBins);
        for (size_t i = 0; i < kNumFrequencyBins; ++i)
        {
            frequency_bins_[i] = static_cast<float>(i) * samplerate_ / nfft;
        }

        spectrum_peaks_.clear();
        peaks_freqs_.clear();
        // Reserve space for peaks, 25% is probably an overly optimistic estimate
        spectrum_peaks_.reserve(0.25 * kNumFrequencyBins);
        peaks_freqs_.reserve(0.25 * kNumFrequencyBins);

        for (size_t i = 1; i < kNumFrequencyBins - 1; ++i)
        {
            if (spectrum_data_[i] > spectrum_data_[i - 1] && spectrum_data_[i] > spectrum_data_[i + 1])
            {
                spectrum_peaks_.push_back(spectrum_data_.at(i));
                peaks_freqs_.push_back(frequency_bins_.at(i));
            }
        }

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::Spectrum));
        LOG_INFO(logger_, "Computing spectrum took {} ms", stopwatch.ElapsedMs());

        // Compute Spectral Flatness for fun
        std::vector<float> temp_spectrum;
        temp_spectrum.resize(spectrum_data_.size());
        fft.ForwardAbs(early_rir, temp_spectrum, false, false);
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
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::Cepstrum)) || early_rir_time != cepstrum_early_rir_time_)
    {
        Stopwatch stopwatch;
        cepstrum_early_rir_time_ = early_rir_time;

        uint32_t early_rir_sample_count = static_cast<uint32_t>(early_rir_time * samplerate_);
        early_rir_sample_count = std::min(early_rir_sample_count, static_cast<uint32_t>(impulse_response_.size()));
        auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        uint32_t nfft = std::max(kSpectrumNFFT, early_rir.size());

        nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
        audio_utils::FFT fft(nfft);
        cepstrum_data_.resize(nfft, 0.f);
        fft.RealCepstrum(early_rir, cepstrum_data_);

        // Only keep the first half of the cepstrum (real cepstrum is symmetric)
        cepstrum_data_.resize(nfft / 2);

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::Cepstrum));
        LOG_INFO(logger_, "Computing cepstrum took {} ms", stopwatch.ElapsedMs());
    }

    return CepstrumData{.cepstrum = cepstrum_data_};
}

AutocorrelationData IRAnalyzer::GetAutocorrelation(float early_rir_time)
{
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::Autocorrelation)) ||
        early_rir_time != autocorrelation_early_rir_time_)
    {
        Stopwatch stopwatch;
        autocorrelation_early_rir_time_ = early_rir_time;

        uint32_t early_rir_sample_count = static_cast<uint32_t>(early_rir_time * samplerate_);
        early_rir_sample_count = std::min(early_rir_sample_count, static_cast<uint32_t>(impulse_response_.size()));
        auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);

        autocorrelation_data_ = audio_utils::analysis::Autocorrelation(early_rir, true);

        uint32_t nfft = std::max(kSpectrumNFFT, early_rir.size());
        nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
        audio_utils::FFT fft(nfft);

        std::vector<float> spectrum_data((nfft / 2) + 1, 0.f);
        fft.ForwardAbs(early_rir, spectrum_data, true, true);

        spectral_autocorrelation_data_ = audio_utils::analysis::Autocorrelation(spectrum_data);

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::Autocorrelation));
        LOG_INFO(logger_, "Computing autocorrelation took {} ms", stopwatch.ElapsedMs());
    }

    return AutocorrelationData{
        .autocorrelation = autocorrelation_data_,
        .spectral_autocorrelation = spectral_autocorrelation_data_,
    };
}

EnergyDecayCurveData IRAnalyzer::GetEnergyDecayCurveData()
{
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::EnergyDecayCurve)))
    {
        Stopwatch stopwatch;
        energy_decay_curve_ = fdn_analysis::EnergyDecayCurve(GetImpulseResponse(), true);
        edc_octaves_ = fdn_analysis::EnergyDecayRelief(GetImpulseResponse(), true);
        auto octave_band_frequencies = fdn_analysis::GetOctaveBandFrequencies();
        octave_band_frequencies_.resize(octave_band_frequencies.size());
        for (size_t i = 0; i < octave_band_frequencies.size(); ++i)
        {
            octave_band_frequencies_[i] = octave_band_frequencies[i];
        }

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::EnergyDecayCurve));
        LOG_INFO(logger_, "Analyzing energy decay curve took {} ms", stopwatch.ElapsedMs());
    }

    std::array<std::span<const float>, 10> edc_octaves;
    for (size_t i = 0; i < edc_octaves.size(); ++i)
    {
        edc_octaves[i] = edc_octaves_[i];
    }

    return EnergyDecayCurveData{
        .energy_decay_curve = energy_decay_curve_,
        .edc_octaves = edc_octaves,
    };
}

T60Data IRAnalyzer::GetT60Data(float decay_db_start, float decay_db_end)
{
    if (decay_db_start <= decay_db_end)
    {
        throw std::invalid_argument("decay_db_start must be greater than decay_db_end");
    }

    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::T60s)))
    {
        Stopwatch stopwatch;
        auto edc_data = GetEnergyDecayCurveData();
        auto time_data = GetTimeData();

        overall_t60_ = fdn_analysis::EstimateT60(edc_data.energy_decay_curve, time_data, decay_db_start, decay_db_end);

        t60_octaves_.clear();
        t60_octaves_.reserve(edc_data.edc_octaves.size());
        for (const auto& octave : edc_data.edc_octaves)
        {
            auto t60_result = fdn_analysis::EstimateT60(octave, time_data, decay_db_start, decay_db_end);
            t60_octaves_.push_back(t60_result.t60);
        }

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::T60s));
        LOG_INFO(logger_, "Analyzing T60 took {} ms", stopwatch.ElapsedMs());
    }

    return T60Data{
        .overall_t60 = overall_t60_, .t60_octaves = t60_octaves_, .octave_band_frequencies = octave_band_frequencies_};
}

EchoDensityData IRAnalyzer::GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms)
{
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::EchoDensity)) ||
        (echo_density_window_size_ms_ != window_size_ms || echo_density_hop_size_ms_ != hop_size_ms))
    {
        Stopwatch stopwatch;
        echo_density_window_size_ms_ = window_size_ms;
        echo_density_hop_size_ms_ = hop_size_ms;

        uint32_t window_size = (window_size_ms * samplerate_) / 1000;
        uint32_t hop_size = (hop_size_ms * samplerate_) / 1000;
        auto echo_density_result = fdn_analysis::EchoDensity(GetImpulseResponse(), window_size, samplerate_, hop_size);

        echo_density_ = std::move(echo_density_result.echo_densities);
        echo_density_indices_.assign(echo_density_result.sparse_indices.begin(),
                                     echo_density_result.sparse_indices.end());

        mixing_time_ = echo_density_result.mixing_time;
        // Convert indices to time in seconds
        for (auto& idx : echo_density_indices_)
        {
            idx /= static_cast<float>(samplerate_);
        }

        analysis_flags_.reset(static_cast<size_t>(AnalysisType::EchoDensity));
        LOG_INFO(logger_, "Analyzing echo density took {} ms", stopwatch.ElapsedMs());
    }

    return EchoDensityData{
        .echo_density = echo_density_, .sparse_indices = echo_density_indices_, .mixing_time = mixing_time_};
}
} // namespace fdn_analysis