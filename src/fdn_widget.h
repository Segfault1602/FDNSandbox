#pragma once

#include "widget.h"

#include <imgui.h>

#include <cassert>
#include <imfilebrowser.h>

#include <sffdn/sffdn.h>

#include <cstddef>
#include <map>

struct FDNDelayRange
{
    int minimum = 256;
    int maximum = 4000;
};

struct FDNDelayWidgetState
{
    std::map<ImGuiID, FDNDelayRange> ranges;
    bool make_prime = false;
};

/// Per-configurator state used by FDN option widgets and embedded file pickers.
struct FDNWidgetState
{
    FDNDelayWidgetState delay_widget;
    float minimum_gain = -1.0f;
    float maximum_gain = 1.0f;
    float set_all_gain = 0.0f;
    int schroeder_delay_minimum = 1;
    int schroeder_delay_maximum = 1000;
    int multichannel_schroeder_delay_minimum = 1;
    int multichannel_schroeder_delay_maximum = 1500;
    // Identifies which filter editor owns the (single, shared) designer window. Zero means closed.
    // A plain bool would make every channel in an independent filter bank open the same window in
    // the same frame, since the designers use fixed window names.
    ImGuiID three_band_designer_owner = 0;
    ImGuiID ten_band_designer_owner = 0;
    ThreeBandDesignerState three_band_designer;
    FilterDesignerState ten_band_designer;
    std::map<std::ptrdiff_t, int> fir_filter_types;
    ImGui::FileBrowser fir_file_dialog;
    std::size_t multichannel_fir_file = 0;
    std::uint32_t multichannel_fir_channel = 0;
    std::size_t velvet_file = 0;
    std::uint32_t velvet_channel = 0;
    std::vector<float> feedback_matrix;
};

/// Dispatches an FDN option variant to its matching ImGui editor.
struct FDNWidgetVisitor
{
    const sfFDN::FDNConfig& fdn_config;
    FDNWidgetState& state;

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

bool DrawFDNOptions(sfFDN::DelayBankOptions& config, const sfFDN::FDNConfig& fdn_config, FDNWidgetState& state);
bool DrawFDNOptions(sfFDN::ParallelGainsOptions& config, const sfFDN::FDNConfig& fdn_config, FDNWidgetState& state);

bool DrawFDNOptions(sfFDN::attenuation_filter_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state);
bool DrawFDNOptions(sfFDN::single_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state);
bool DrawFDNOptions(sfFDN::multi_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state);
bool DrawFDNOptions(sfFDN::feedback_matrix_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state);

bool DrawSingleChannelProcessorList(std::vector<sfFDN::single_channel_processor_variant_t>& processors,
                                    sfFDN::FDNConfig& fdn_config, FDNWidgetState& state);
std::optional<sfFDN::single_channel_processor_variant_t> DrawAddSingleChannelProcessorPopup();

bool DrawMultiChannelProcessorList(std::vector<sfFDN::multi_channel_processor_variant_t>& processors,
                                   sfFDN::FDNConfig& fdn_config, FDNWidgetState& state);
std::optional<sfFDN::multi_channel_processor_variant_t> DrawAddMultiChannelProcessorPopup(
    const sfFDN::FDNConfig& fdn_config);

bool DrawVelvetNoiseDecorrelatorConfig(sfFDN::FirOptions& config, const sfFDN::FDNConfig& fdn_config,
                                       FDNWidgetState& state);