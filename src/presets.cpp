#include "presets.h"

#include <algorithm>
#include <memory>

#include <sffdn/sffdn.h>

#include "settings.h"

namespace presets
{

std::unique_ptr<sfFDN::FDN> CreateDefaultFDN()
{
    return CreateFDNFromConfig(kDefaultFDNConfig, Settings::Instance().SampleRate());
}

} // namespace presets