#include "fdn_analyzer.h"
#include "sffdn/audio_buffer.h"

#include "analysis.h"
#include <audio_utils/audio_analysis.h>

#include <algorithm>
#include <numbers>
#include <sys/types.h>

namespace
{
#ifndef NDEBUG
constexpr int kFFTSize = 1024; // Size of FFT for spectrogram
constexpr int kSpectrogramWindowSize = 1024;
constexpr int kOverlap = 300; // Overlap size for spectrogram
#else
constexpr int kFFTSize = 1 << 14; // Size of FFT for spectrogram
constexpr int kSpectrogramWindowSize = 512;
constexpr int kOverlap = 400; // Overlap size for spectrogram
#endif
constexpr audio_utils::FFTWindowType kFFTWindowType = audio_utils::FFTWindowType::Hann;

constexpr size_t kSpectrumNFFT = 48000;
constexpr uint32_t kFilterNFFT = 8192; // Number of FFT points for filter response

void GetFilterResponse(std::span<const float> input, std::vector<float>& mag_response,
                       std::vector<float>& phase_response)
{
    std::vector<std::complex<float>> filter_spectrum = audio_utils::FFT(input, kFilterNFFT);
    mag_response.resize(filter_spectrum.size());
    phase_response.resize(filter_spectrum.size());

    // Compute the phase response
    for (size_t j = 0; j < phase_response.size(); ++j)
    {
        phase_response[j] = 180.f / std::numbers::pi * std::arg(filter_spectrum[j]);
    }

    // Compute the magnitude response
    for (size_t j = 0; j < mag_response.size(); ++j)
    {
        // Convert to dB scale
        mag_response[j] = 20.0f * std::log10(std::abs(filter_spectrum[j]));
        if (std::isinf(mag_response[j]) || std::isnan(mag_response[j]))
        {
            mag_response[j] = -50.0f; // Set to a low value if the log is invalid
        }
    }
}

} // namespace

namespace fdn_analysis
{
FDNAnalyzer::FDNAnalyzer(uint32_t samplerate)
    : samplerate_(samplerate)
    , impulse_response_size_samples_(0)
    , spectrogram_bin_count_(0)
    , spectrogram_frame_count_(0)
    , spectrum_early_rir_time_(0.5f)
{
    // Initialize the impulse response size to a default value
    SetImpulseResponseSize(samplerate);
}

void FDNAnalyzer::SetFDN(std::unique_ptr<sfFDN::FDN> fdn)
{
    fdn_ = std::move(fdn);

    analysis_flags_.set();
}

void FDNAnalyzer::SetImpulseResponseSize(uint32_t size_samples)
{
    impulse_response_size_samples_ = size_samples;
    impulse_response_.resize(size_samples, 0.0f);
    analysis_flags_.set();
}

uint32_t FDNAnalyzer::GetImpulseResponseSize() const
{
    return impulse_response_size_samples_;
}

std::span<const float> FDNAnalyzer::GetImpulseResponse()
{
    if (analysis_flags_.test(AnalysisType::ImpulseResponse))
    {
        std::vector<float> ir(impulse_response_size_samples_, 0.0f);
        ir[0] = 1.f;

        std::vector<float> output(impulse_response_size_samples_, 0.0f);

        sfFDN::AudioBuffer input_buffer(ir);
        sfFDN::AudioBuffer output_buffer(output);

        fdn_->Clear();
        fdn_->Process(input_buffer, output_buffer);

        impulse_response_ = std::move(output);
        is_clipping_ = std::any_of(impulse_response_.begin(), impulse_response_.end(),
                                   [](float sample) { return sample < -1.0f || sample > 1.0f; });
        analysis_flags_.reset(AnalysisType::ImpulseResponse);
    }
    return impulse_response_;
}

std::span<const float> FDNAnalyzer::GetTimeData()
{
    // Call GetImpulseResponse to make sure we have the latest impulse response
    auto impulse_response = GetImpulseResponse();

    time_data_.resize(impulse_response.size());
    for (size_t i = 0; i < impulse_response.size(); ++i)
    {
        time_data_[i] =
            static_cast<float>(i) / static_cast<float>(samplerate_); // Convert sample index to time in seconds
    }
    return time_data_;
}

bool FDNAnalyzer::IsClipping()
{
    return is_clipping_;
}

SpectrogramData FDNAnalyzer::GetSpectrogram()
{
    if (analysis_flags_.test(AnalysisType::Spectrogram))
    {
        audio_utils::analysis::SpectrogramInfo spec_info{.fft_size = kFFTSize,
                                                         .overlap = kOverlap,
                                                         .samplerate = static_cast<int>(samplerate_),
                                                         .window_size = kSpectrogramWindowSize,
                                                         .window_type = kFFTWindowType};
        constexpr size_t kNMels = 512;
        spectrogram_data_ = audio_utils::analysis::MelSpectrogram(GetImpulseResponse(), spec_info, kNMels);
        spectrogram_bin_count_ = spec_info.num_freqs;
        spectrogram_frame_count_ = spec_info.num_frames;

        analysis_flags_.reset(AnalysisType::Spectrogram);
    }

    return SpectrogramData{
        .data = spectrogram_data_, .bin_count = spectrogram_bin_count_, .frame_count = spectrogram_frame_count_};
}

SpectrumData FDNAnalyzer::GetSpectrum(float early_rir_time)
{
    if (analysis_flags_.test(AnalysisType::Spectrum) || early_rir_time != spectrum_early_rir_time_)
    {
        spectrum_early_rir_time_ = early_rir_time;

        uint32_t early_rir_sample_count = static_cast<uint32_t>(early_rir_time * samplerate_);
        early_rir_sample_count = std::min(early_rir_sample_count, static_cast<uint32_t>(impulse_response_.size()));
        auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        const uint32_t nfft = std::max(kSpectrumNFFT, early_rir.size());
        spectrum_data_ = audio_utils::AbsFFT(early_rir, nfft, true, true);

        // Generate frequency bins
        const uint32_t kNumFrequencyBins = spectrum_data_.size();
        frequency_bins_.resize(kNumFrequencyBins);
        for (size_t i = 0; i < kNumFrequencyBins; ++i)
        {
            frequency_bins_[i] = static_cast<float>(i) * samplerate_ / kSpectrumNFFT;
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

        analysis_flags_.reset(AnalysisType::Spectrum);
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

CepstrumData FDNAnalyzer::GetCepstrum(float early_rir_time)
{
    if (analysis_flags_.test(AnalysisType::Cepstrum) || early_rir_time != cepstrum_early_rir_time_)
    {
        cepstrum_early_rir_time_ = early_rir_time;

        uint32_t early_rir_sample_count = static_cast<uint32_t>(early_rir_time * samplerate_);
        early_rir_sample_count = std::min(early_rir_sample_count, static_cast<uint32_t>(impulse_response_.size()));
        auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        const uint32_t nfft = std::max(kSpectrumNFFT, early_rir.size());
        cepstrum_data_ = audio_utils::AbsCepstrum(early_rir, nfft);

        analysis_flags_.reset(AnalysisType::Cepstrum);
    }

    return CepstrumData{.cepstrum = cepstrum_data_};
}

AutocorrelationData FDNAnalyzer::GetAutocorrelation(float early_rir_time)
{
    if (analysis_flags_.test(AnalysisType::Autocorrelation) || early_rir_time != autocorrelation_early_rir_time_)
    {
        autocorrelation_early_rir_time_ = early_rir_time;

        uint32_t early_rir_sample_count = static_cast<uint32_t>(early_rir_time * samplerate_);
        early_rir_sample_count = std::min(early_rir_sample_count, static_cast<uint32_t>(impulse_response_.size()));
        auto early_rir = GetImpulseResponse().subspan(0, early_rir_sample_count);
        const uint32_t nfft = std::max(kSpectrumNFFT, early_rir.size());

        autocorrelation_data_ = audio_utils::analysis::Autocorrelation(early_rir, nfft);
        spectral_autocorrelation_data_ = audio_utils::analysis::Autocorrelation(audio_utils::AbsFFT(early_rir, nfft));

        analysis_flags_.reset(AnalysisType::Autocorrelation);
    }

    return AutocorrelationData{
        .autocorrelation = autocorrelation_data_,
        .spectral_autocorrelation = spectral_autocorrelation_data_,
    };
}

FilterData FDNAnalyzer::GetFilterData()
{
    assert(fdn_ != nullptr);
    if (analysis_flags_.test(AnalysisType::FilterResponse))
    {
        filter_freq_bins_.resize(kFilterNFFT / 2 + 1);
        for (size_t i = 0; i < filter_freq_bins_.size(); ++i)
        {
            filter_freq_bins_[i] = static_cast<float>(i) * samplerate_ / kFilterNFFT;
        }

        filter_mag_responses_.clear();
        filter_phase_responses_.clear();

        if (fdn_->GetFilterBank())
        {
            auto filter_bank = fdn_->GetFilterBank();
            filter_bank->Clear();

            const uint32_t kFilterCount = filter_bank->InputChannelCount();

            std::vector<float> impulse(kFilterNFFT * kFilterCount, 0.0f);
            // Create an impulse for each filter
            for (uint32_t i = 0; i < kFilterCount; ++i)
            {
                impulse[i * kFilterNFFT] = 1.0f;
            }

            std::vector<float> output(kFilterNFFT * kFilterCount, 0.0f);

            sfFDN::AudioBuffer input_buffer(kFilterNFFT, kFilterCount, impulse);
            sfFDN::AudioBuffer output_buffer(kFilterNFFT, kFilterCount, output);
            filter_bank->Process(input_buffer, output_buffer);

            filter_mag_responses_.resize(kFilterCount);
            filter_phase_responses_.resize(kFilterCount);

            for (uint32_t i = 0; i < kFilterCount; ++i)
            {
                auto channel_span = output_buffer.GetChannelSpan(i);
                GetFilterResponse(channel_span, filter_mag_responses_[i], filter_phase_responses_[i]);
            }
        }

        if (fdn_->GetTCFilter())
        {
            auto* tc_filter = fdn_->GetTCFilter();
            tc_filter->Clear();

            constexpr uint32_t kFilterNFFT = 8192; // Number of FFT points for TC filter response
            std::vector<float> impulse(kFilterNFFT, 0.0f);
            impulse[0] = 1.0f; // Create an impulse

            std::vector<float> output(kFilterNFFT, 0.0f);

            sfFDN::AudioBuffer input_buffer(kFilterNFFT, 1, impulse);
            sfFDN::AudioBuffer output_buffer(kFilterNFFT, 1, output);
            tc_filter->Process(input_buffer, output_buffer);
            GetFilterResponse(output_buffer.GetChannelSpan(0), tc_filter_mag_response_, tc_filter_phase_response_);
        }

        analysis_flags_.reset(AnalysisType::FilterResponse);
    }

    FilterData filter_data;
    for (const auto& response : filter_mag_responses_)
    {
        filter_data.mag_responses.emplace_back(response);
    }

    for (const auto& response : filter_phase_responses_)
    {
        filter_data.phase_responses.emplace_back(response);
    }

    filter_data.tc_mag_response = tc_filter_mag_response_;
    filter_data.tc_phase_response = tc_filter_phase_response_;

    filter_data.frequency_bins = filter_freq_bins_;

    return filter_data;
}

EnergyDecayCurveData FDNAnalyzer::GetEnergyDecayCurveData()
{
    if (analysis_flags_.test(AnalysisType::EnergyDecayCurve))
    {
        energy_decay_curve_ = fdn_analysis::EnergyDecayCurve(GetImpulseResponse(), true);
        edc_octaves_ = fdn_analysis::EnergyDecayRelief(GetImpulseResponse(), true);
        auto octave_band_frequencies = fdn_analysis::GetOctaveBandFrequencies();
        octave_band_frequencies_.resize(octave_band_frequencies.size());
        for (size_t i = 0; i < octave_band_frequencies.size(); ++i)
        {
            octave_band_frequencies_[i] = octave_band_frequencies[i];
        }

        analysis_flags_.reset(AnalysisType::EnergyDecayCurve);
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

T60Data FDNAnalyzer::GetT60Data(float decay_db_start, float decay_db_end)
{
    if (decay_db_start <= decay_db_end)
    {
        throw std::invalid_argument("decay_db_start must be greater than decay_db_end");
    }

    if (analysis_flags_.test(AnalysisType::T60s) ||
        (decay_db_start != decay_db_start_ || decay_db_end != decay_db_end_))
    {
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

        analysis_flags_.reset(AnalysisType::T60s);
    }

    return T60Data{
        .overall_t60 = overall_t60_, .t60_octaves = t60_octaves_, .octave_band_frequencies = octave_band_frequencies_};
}

EchoDensityData FDNAnalyzer::GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms)
{
    if (analysis_flags_.test(AnalysisType::EchoDensity) ||
        (echo_density_window_size_ms_ != window_size_ms || echo_density_hop_size_ms_ != hop_size_ms))
    {
        echo_density_window_size_ms_ = window_size_ms;
        echo_density_hop_size_ms_ = hop_size_ms;

        uint32_t window_size = (window_size_ms * samplerate_) / 1000;
        uint32_t hop_size = (hop_size_ms * samplerate_) / 1000;
        auto echo_density_result = fdn_analysis::EchoDensity(GetImpulseResponse(), window_size, samplerate_, hop_size);

        echo_density_ = std::move(echo_density_result.echo_densities);
        echo_density_indices_.assign(echo_density_result.sparse_indices.begin(),
                                     echo_density_result.sparse_indices.end());

        analysis_flags_.reset(AnalysisType::EchoDensity);
    }

    return EchoDensityData{.echo_density = echo_density_, .sparse_indices = echo_density_indices_};
}
} // namespace fdn_analysis