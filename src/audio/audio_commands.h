#pragma once

#include "audio/audio_clip.h"

#include <sffdn/sffdn.h>

#include <cstdint>
#include <memory>
#include <variant>

namespace fdn_sandbox::audio
{

struct InstallFdn
{
    std::uint64_t generation = 0;
    std::unique_ptr<sfFDN::FDN> value;
};

struct InstallConvolver
{
    std::unique_ptr<sfFDN::PartitionedConvolver> value;
};

struct InstallClip
{
    std::unique_ptr<AudioClip> value;
};

struct PlayClip
{
    bool loop = false;
};

struct StopClip
{
};

struct TriggerImpulse
{
};

using AudioCommand = std::variant<InstallFdn, InstallConvolver, InstallClip, PlayClip, StopClip, TriggerImpulse>;

} // namespace fdn_sandbox::audio
