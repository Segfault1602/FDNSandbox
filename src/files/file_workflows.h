#pragma once

#include "audio/audio_clip.h"
#include "files/file_error.h"

#include <sffdn/sffdn.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace fdn_sandbox::files
{

/// A decoded mono RIR and the convolver prepared from the same samples.
struct LoadedRir
{
    std::filesystem::path path;
    std::uint32_t effective_sample_rate;
    std::vector<float> mono_samples;
    std::unique_ptr<sfFDN::PartitionedConvolver> convolver;
};

/**
 * @brief Performs headless, non-real-time audio file operations.
 *
 * Decodes clips and RIRs, resamples clips, prepares convolvers, and writes
 * impulse responses without depending on ImGui or application notifications.
 */
class FileWorkflows final
{
  public:
    [[nodiscard]] std::expected<LoadedRir, FileError> LoadRir(const std::filesystem::path& path,
                                                              std::size_t convolution_block_size) const;

    [[nodiscard]] std::expected<audio::AudioClip, FileError> LoadClip(const std::filesystem::path& path,
                                                                      std::uint32_t engine_sample_rate) const;

    [[nodiscard]] std::expected<void, FileError> SaveImpulseResponse(const std::filesystem::path& path,
                                                                     std::span<const float> samples,
                                                                     std::uint32_t sample_rate) const;
};

} // namespace fdn_sandbox::files
