#pragma once

#include "fdn_analyzer.h"
#include "ir_analyzer.h"

#include <audio_utils/audio_analysis.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fdn_sandbox::analysis
{

enum class SpectrogramScale : std::uint8_t
{
    Stft,
    Mel
};

struct SpectrogramSettings
{
    audio_utils::analysis::STFTOptions stft;
    SpectrogramScale scale = SpectrogramScale::Stft;
};

struct ReferenceIrMetadata
{
    std::string filename;
    std::uint32_t analysis_sample_rate;
};

class AnalysisWorkspace final
{
  public:
    AnalysisWorkspace(std::uint32_t sample_rate, quill::Logger* logger);

    void InstallGeneratedFdn(std::unique_ptr<sfFDN::FDN> fdn);
    void SetReferenceIr(std::vector<float> impulse_response, ReferenceIrMetadata metadata);
    void ClearReferenceIr();

    void SetSpectrogramSettings(SpectrogramSettings settings);
    const SpectrogramSettings& GetSpectrogramSettings() const noexcept;

    fdn_analysis::FDNAnalyzer& Generated() noexcept;
    fdn_analysis::IRAnalyzer& Reference() noexcept;
    const fdn_analysis::FDNAnalyzer& Generated() const noexcept;
    const fdn_analysis::IRAnalyzer& Reference() const noexcept;

    bool HasReference() const noexcept;
    const std::optional<ReferenceIrMetadata>& ReferenceMetadata() const noexcept;

  private:
    fdn_analysis::FDNAnalyzer generated_;
    fdn_analysis::IRAnalyzer reference_;
    SpectrogramSettings spectrogram_settings_;
    std::optional<ReferenceIrMetadata> reference_metadata_;
};

} // namespace fdn_sandbox::analysis
