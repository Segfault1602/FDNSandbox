#pragma once

#include "app.h"

#include <sffdn/sffdn.h>

namespace presets
{

sfFDN::FDNConfig GetDefaultFDNConfig();

std::unique_ptr<sfFDN::FDN> CreateDefaultFDN();

} // namespace presets