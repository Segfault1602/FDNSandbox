#pragma once

#include <cstdint>
#include <vector>

namespace fdn_sandbox::audio
{

struct AudioClip
{
    std::vector<float> samples;
    std::uint32_t sample_rate = 0;
};

} // namespace fdn_sandbox::audio
