#include "fdn_analyzer.h"
#include "sffdn/audio_buffer.h"

#include "analysis.h"
#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <algorithm>
#include <chrono>
#include <numbers>

namespace
{
constexpr size_t kSpectrumNFFT = 48000;
constexpr uint32_t kFilterNFFT = 8192; // Number of FFT points for filter response

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

void GetFilterResponse(std::span<const float> input, std::vector<float>& mag_response,
                       std::vector<float>& phase_response)
{
    uint32_t nfft = audio_utils::FFT::NextSupportedFFTSize(kFilterNFFT);
    audio_utils::FFT fft(nfft);
    std::vector<std::complex<float>> filter_spectrum((nfft / 2) + 1, 0.f);

    fft.Forward(input, filter_spectrum);

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
FDNAnalyzer::FDNAnalyzer(uint32_t samplerate, quill::Logger* logger)
    : logger_(logger)
    , samplerate_(samplerate)
    , ir_analyzer_(samplerate, logger)
{
    // Initialize the impulse response size to a default value
    SetImpulseResponseSize(samplerate);
}

void FDNAnalyzer::SetFDN(std::unique_ptr<sfFDN::FDN> fdn)
{
    fdn_ = std::move(fdn);
    analysis_flags_.set();
    GetImpulseResponse(); // Force computation of the impulse response
}

void FDNAnalyzer::SetImpulseResponseSize(uint32_t size_samples)
{
    impulse_response_size_samples_ = size_samples;
    analysis_flags_.set();
}

uint32_t FDNAnalyzer::GetImpulseResponseSize() const
{
    return impulse_response_size_samples_;
}

std::span<const float> FDNAnalyzer::GetImpulseResponse()
{
    if (analysis_flags_.test(static_cast<size_t>(AnalysisType::ImpulseResponse)))
    {
        Stopwatch stopwatch;

        std::vector<float> ir(impulse_response_size_samples_, 0.0f);
        ir[0] = 1.f;

        std::vector<float> impulse_response;
        impulse_response.resize(impulse_response_size_samples_, 0.0f);
        sfFDN::AudioBuffer input_buffer(ir);
        sfFDN::AudioBuffer output_buffer(impulse_response);

        fdn_->Clear();
        fdn_->SetDirectGain(0.0f);
        fdn_->Process(input_buffer, output_buffer);

        is_clipping_ =
            std::ranges::any_of(impulse_response, [](float sample) { return sample < -1.0f || sample > 1.0f; });
        analysis_flags_.reset(static_cast<size_t>(AnalysisType::ImpulseResponse));

        auto end = std::chrono::high_resolution_clock::now();
        LOG_INFO(logger_, "Rendering new impulse response took {} ms", stopwatch.ElapsedMs());

        ir_analyzer_.SetImpulseResponse(std::move(impulse_response));
    }

    return ir_analyzer_.GetImpulseResponse();
}

std::span<const float> FDNAnalyzer::GetTimeData()
{
    // Call GetImpulseResponse to make sure we have the latest impulse response
    GetImpulseResponse();

    return ir_analyzer_.GetTimeData();
}

bool FDNAnalyzer::IsClipping()
{
    return is_clipping_;
}

SpectrogramData FDNAnalyzer::GetSpectrogram(audio_utils::analysis::SpectrogramInfo spec_info, bool mel_scale)
{
    return ir_analyzer_.GetSpectrogram(spec_info, mel_scale);
}

SpectrumData FDNAnalyzer::GetSpectrum(float early_rir_time)
{
    return ir_analyzer_.GetSpectrum(early_rir_time);
}

CepstrumData FDNAnalyzer::GetCepstrum(float early_rir_time)
{
    return ir_analyzer_.GetCepstrum(early_rir_time);
}

AutocorrelationData FDNAnalyzer::GetAutocorrelation(float early_rir_time)
{
    return ir_analyzer_.GetAutocorrelation(early_rir_time);
}

FilterData FDNAnalyzer::GetFilterData()
{
    assert(fdn_ != nullptr);
    if (analysis_flags_.test(static_cast<size_t>(FDNAnalysisType::FilterResponse)))
    {
        Stopwatch stopwatch;
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
        else
        {
            std::ranges::fill(tc_filter_mag_response_, 0.0f);
            std::ranges::fill(tc_filter_phase_response_, 0.0f);
        }

        analysis_flags_.reset(static_cast<size_t>(FDNAnalysisType::FilterResponse));
        LOG_INFO(logger_, "Analyzing filter response took {} ms", stopwatch.ElapsedMs());
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
    return ir_analyzer_.GetEnergyDecayCurveData();
}

T60Data FDNAnalyzer::GetT60Data(float decay_db_start, float decay_db_end)
{
    return ir_analyzer_.GetT60Data(decay_db_start, decay_db_end);
}

EchoDensityData FDNAnalyzer::GetEchoDensityData(uint32_t window_size_ms, uint32_t hop_size_ms)
{
    return ir_analyzer_.GetEchoDensityData(window_size_ms, hop_size_ms);
}
} // namespace fdn_analysis