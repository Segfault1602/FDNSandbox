#pragma once

#include <sffdn/sffdn.h>

struct FDNWidgetVisitor
{
    const sfFDN::FDNConfig& fdn_config;

    bool operator()(sfFDN::ScalarFeedbackMatrixOptions& config) const;
    bool operator()(sfFDN::CascadedFeedbackMatrixOptions& config) const;
    bool operator()(sfFDN::ModulationOptions& config) const;
    bool operator()(sfFDN::ParallelGainsOptions& config) const;
    bool operator()(sfFDN::DelayOptions& config) const;
    bool operator()(sfFDN::DelayBankOptions& config) const;
    bool operator()(sfFDN::DelayBankTimeVaryingOptions& config) const;
    bool operator()(sfFDN::SchroederAllpassSectionOptions& config) const;
    bool operator()(sfFDN::MultichannelSchroederAllpassSectionOptions& config) const;
    bool operator()(sfFDN::HomogenousFilterOptions& config) const;
    bool operator()(sfFDN::TwoBandFilterOptions& config) const;
    bool operator()(sfFDN::ThreeBandFilterOptions& config) const;
    bool operator()(sfFDN::TenBandFilterOptions& config) const;
    bool operator()(sfFDN::GraphicEQOptions& config) const;

    bool operator()(sfFDN::AllpassFilterOptions& config) const;
    bool operator()(sfFDN::CascadedBiquadsOptions& config) const;
    bool operator()(sfFDN::FirOptions& config) const;
    bool operator()(sfFDN::MultichannelFirOptions& config) const;
    bool operator()(sfFDN::AttenuationFilterBankOptions& config) const;
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
                                   sfFDN::FDNConfig& fdn_config);
std::optional<sfFDN::multi_channel_processor_variant_t> DrawAddMultiChannelProcessorPopup(
    const sfFDN::FDNConfig& fdn_config);

bool DrawVelvetNoiseDecorrelatorConfig(sfFDN::FirOptions& config, const sfFDN::FDNConfig& fdn_config);