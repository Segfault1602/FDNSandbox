#pragma once

#include "audio/audio_clip.h"

#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace fdn_sandbox::audio
{

std::expected<std::unique_ptr<AudioClip>, std::string> LoadAudioClip(const std::filesystem::path& path,
                                                                    std::uint32_t target_sample_rate);

} // namespace fdn_sandbox::audio
