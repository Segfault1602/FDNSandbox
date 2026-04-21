#include "presets.h"

#include <algorithm>
#include <memory>

#include <sffdn/sffdn.h>

#include "settings.h"

namespace presets
{

sfFDN::FDNConfig GetDefaultFDNConfig()
{
    sfFDN::FDNConfig config{};
    config.fdn_size = 8;
    config.transposed = false;
    config.direct_gain = 0.0f;
    config.block_size = Settings::Instance().BlockSize();
    config.sample_rate = Settings::Instance().SampleRate();
    config.delay_bank_config = sfFDN::DelayBankOptions{.delays = {809, 877, 937, 1049, 1151, 1249, 1373, 1499},
                                                       .block_size = Settings::Instance().BlockSize(),
                                                       .interpolation_type = sfFDN::DelayInterpolationType::None};

    config.input_block_config.parallel_gains_config.gains = std::vector<float>(config.fdn_size, 0.5f);
    config.output_block_config.parallel_gains_config.gains = std::vector<float>(config.fdn_size, 0.5f);

    config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
        .matrix_size = config.fdn_size,
        .type = sfFDN::ScalarMatrixType::Hadamard,
    };

    sfFDN::AttenuationFilterBankOptions attenuation_filter_bank_options{};
    for (auto delay : config.delay_bank_config.delays)
    {
        sfFDN::HomogenousFilterOptions attenuation_options{
            .t60 = 1.f, .delay = delay, .sample_rate = static_cast<float>(config.sample_rate)};
        attenuation_filter_bank_options.filter_configs.push_back(attenuation_options);
    }
    config.loop_filter_configs.push_back(attenuation_filter_bank_options);

    return config;
}

std::unique_ptr<sfFDN::FDN> CreateDefaultFDN()
{
    // return CreateFDNFromConfig(GetDefaultFDNConfig(), Settings::Instance().SampleRate());
    return nullptr;
}

} // namespace presets