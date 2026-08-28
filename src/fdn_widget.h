#pragma once

#include "utils.h"
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

/// Persistent plotting buffers for the graphic EQ editor.
struct GraphicEqEditorState
{
    std::vector<float> plot_frequencies;
    std::vector<float> response_db;
    std::vector<double> frequency_ticks;
    float plot_sample_rate = 0.0f;
    bool show_response = true;
};

/// Caches the rotation block count of a time-varying feedback matrix.
///
/// RealSchur mode derives its blocks from a basis built at construction, so learning the count costs a Schur
/// decomposition plus a Householder QR. That is far too expensive to repeat every frame, hence the cache.
struct TimeVaryingMatrixBlockCache
{
    uint32_t matrix_size = 0;
    sfFDN::TimeVaryingMatrixMode mode = sfFDN::TimeVaryingMatrixMode::Hadamard;
    uint32_t rng_seed = 0;
    uint32_t block_count = 0;
    bool valid = false;

    uint32_t Get(const sfFDN::TimeVaryingFeedbackMatrixOptions& config)
    {
        if (!valid || matrix_size != config.matrix_size || mode != config.mode || rng_seed != config.rng_seed)
        {
            matrix_size = config.matrix_size;
            mode = config.mode;
            rng_seed = config.rng_seed;
            block_count = utils::GetTimeVaryingRotationBlockCount(config);
            valid = true;
        }
        return block_count;
    }
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
    GraphicEqEditorState graphic_eq_editor;
    std::map<std::ptrdiff_t, int> fir_filter_types;
    ImGui::FileBrowser fir_file_dialog;
    std::size_t multichannel_fir_file = 0;
    std::uint32_t multichannel_fir_channel = 0;
    std::size_t velvet_file = 0;
    std::uint32_t velvet_channel = 0;
    std::vector<float> feedback_matrix;
    TimeVaryingMatrixBlockCache time_varying_matrix_blocks;
    /// Wall-clock-driven sample counter used to animate the time-varying feedback matrix heatmap.
    ///
    /// Advanced by `sample_rate * frame_delta` rather than one sample per frame, so a 1 Hz LFO visibly completes one
    /// cycle per second. A double holds sample counts exactly past any plausible session length.
    double feedback_matrix_animation_samples = 0.0;
};

/// Dispatches an FDN option variant to its matching ImGui editor.
struct FDNWidgetVisitor
{
    const sfFDN::FDNConfig& fdn_config;
    FDNWidgetState& state;

    bool operator()(sfFDN::ScalarFeedbackMatrixOptions& config) const;
    bool operator()(sfFDN::CascadedFeedbackMatrixOptions& config) const;
    bool operator()(sfFDN::TimeVaryingFeedbackMatrixOptions& config) const;
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
std::optional<sfFDN::single_channel_processor_variant_t> DrawAddSingleChannelProcessorPopup(
    const sfFDN::FDNConfig& fdn_config);

bool DrawMultiChannelProcessorList(std::vector<sfFDN::multi_channel_processor_variant_t>& processors,
                                   sfFDN::FDNConfig& fdn_config, FDNWidgetState& state);
std::optional<sfFDN::multi_channel_processor_variant_t> DrawAddMultiChannelProcessorPopup(
    const sfFDN::FDNConfig& fdn_config);

bool DrawVelvetNoiseDecorrelatorConfig(sfFDN::FirOptions& config, const sfFDN::FDNConfig& fdn_config,
                                       FDNWidgetState& state);