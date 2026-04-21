#pragma once

#include <sffdn/sffdn.h>

struct FDNWidgetVisitor
{
    const sfFDN::FDNConfig& fdn_config;

    bool operator()(sfFDN::ScalarFeedbackMatrixOptions& config);
    bool operator()(sfFDN::CascadedFeedbackMatrixOptions& config);
    bool operator()(sfFDN::ModulationOptions& config);
    bool operator()(sfFDN::ParallelGainsOptions& config);
    bool operator()(sfFDN::DelayOptions& config);
    bool operator()(sfFDN::DelayBankOptions& config);
    bool operator()(sfFDN::DelayBankTimeVaryingOptions& config);
    bool operator()(sfFDN::SchroederAllpassSectionOptions& config);
    bool operator()(sfFDN::MultichannelSchroederAllpassSectionOptions& config);
    bool operator()(sfFDN::HomogenousFilterOptions& config);
    bool operator()(sfFDN::TwoBandFilterOptions& config);
    bool operator()(sfFDN::ThreeBandFilterOptions& config);
    bool operator()(sfFDN::TenBandFilterOptions& config);
    bool operator()(sfFDN::GraphicEQOptions& config);

    bool operator()(sfFDN::AllpassFilterOptions& config);
    bool operator()(sfFDN::CascadedBiquadsOptions& config);
    bool operator()(sfFDN::FirOptions& config);
    bool operator()(sfFDN::AttenuationFilterBankOptions& config);
};

bool DrawFDNOptions(sfFDN::DelayBankOptions& config, const sfFDN::FDNConfig& fdn_config);
bool DrawFDNOptions(sfFDN::ParallelGainsOptions& config, const sfFDN::FDNConfig& fdn_config);

bool DrawFDNOptions(sfFDN::attenuation_filter_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config);
bool DrawFDNOptions(sfFDN::single_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config);
bool DrawFDNOptions(sfFDN::multi_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config);
bool DrawFDNOptions(sfFDN::feedback_matrix_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config);

bool DrawSingleChannelProcessorList(std::vector<sfFDN::single_channel_processor_variant_t>& processors,
                                    sfFDN::FDNConfig& fdn_config);
std::optional<sfFDN::single_channel_processor_variant_t> DrawAddSingleChannelProcessorPopup();

bool DrawMultiChannelProcessorList(std::vector<sfFDN::multi_channel_processor_variant_t>& processors,
                                   sfFDN::FDNConfig& fdn_config, bool is_loop_filter = false);

bool DrawVelvetNoiseDecorrelatorConfig(sfFDN::FirOptions& config, const sfFDN::FDNConfig& fdn_config);