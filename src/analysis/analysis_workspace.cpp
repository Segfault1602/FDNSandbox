#include "analysis/analysis_workspace.h"

#include <utility>

namespace
{

audio_utils::analysis::STFTOptions MakeDefaultStftOptions(std::uint32_t sample_rate)
{
    return {
#ifndef NDEBUG
        .fft_size = 1024,
        .overlap = 300,
        .window_size = 1024,
#else
        .fft_size = 2048,
        .overlap = 400,
        .window_size = 512,
#endif
        .window_type = audio_utils::FFTWindowType::Hann,
        .samplerate = sample_rate,
    };
}

bool SameStftOptions(const audio_utils::analysis::STFTOptions& lhs, const audio_utils::analysis::STFTOptions& rhs)
{
    return lhs.fft_size == rhs.fft_size && lhs.overlap == rhs.overlap && lhs.window_size == rhs.window_size &&
           lhs.window_type == rhs.window_type && lhs.samplerate == rhs.samplerate;
}

} // namespace

namespace fdn_sandbox::analysis
{

AnalysisWorkspace::AnalysisWorkspace(std::uint32_t sample_rate, quill::Logger* logger)
    : generated_(sample_rate, logger)
    , reference_(sample_rate, logger)
    , spectrogram_settings_{
          .stft = MakeDefaultStftOptions(sample_rate),
          .scale = SpectrogramScale::Stft,
      }
{
}

void AnalysisWorkspace::InstallGeneratedFdn(std::unique_ptr<sfFDN::FDN> fdn)
{
    generated_.SetFDN(std::move(fdn));
}

void AnalysisWorkspace::SetReferenceIr(std::vector<float> impulse_response, ReferenceIrMetadata metadata)
{
    if (impulse_response.empty())
    {
        ClearReferenceIr();
        return;
    }
    reference_.SetImpulseResponse(std::move(impulse_response));
    reference_metadata_ = std::move(metadata);
}

void AnalysisWorkspace::ClearReferenceIr()
{
    reference_.SetImpulseResponse({});
    reference_metadata_.reset();
}

void AnalysisWorkspace::SetSpectrogramSettings(SpectrogramSettings settings)
{
    if (SameStftOptions(spectrogram_settings_.stft, settings.stft) && spectrogram_settings_.scale == settings.scale)
    {
        return;
    }

    spectrogram_settings_ = settings;
    generated_.Invalidate(fdn_analysis::IRAnalysisType::Spectrogram);
    reference_.Invalidate(fdn_analysis::IRAnalysisType::Spectrogram);
}

const SpectrogramSettings& AnalysisWorkspace::GetSpectrogramSettings() const noexcept
{
    return spectrogram_settings_;
}

fdn_analysis::FDNAnalyzer& AnalysisWorkspace::Generated() noexcept
{
    return generated_;
}

fdn_analysis::IRAnalyzer& AnalysisWorkspace::Reference() noexcept
{
    return reference_;
}

const fdn_analysis::FDNAnalyzer& AnalysisWorkspace::Generated() const noexcept
{
    return generated_;
}

const fdn_analysis::IRAnalyzer& AnalysisWorkspace::Reference() const noexcept
{
    return reference_;
}

bool AnalysisWorkspace::HasReference() const noexcept
{
    return reference_metadata_.has_value() && reference_.GetImpulseResponseSize() != 0;
}

const std::optional<ReferenceIrMetadata>& AnalysisWorkspace::ReferenceMetadata() const noexcept
{
    return reference_metadata_;
}

} // namespace fdn_sandbox::analysis
