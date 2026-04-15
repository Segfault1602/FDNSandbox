#pragma once

#include "app.h"

#include <sffdn/sffdn.h>

namespace presets
{

sfFDN::FDNConfig2 GetDefaultFDNConfig();

std::unique_ptr<sfFDN::FDN> CreateDefaultFDN();

} // namespace presets