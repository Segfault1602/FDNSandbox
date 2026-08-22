#include "app.h"

#include "fdn_widget.h"
#include "theme.h"
#include "utils.h"
#include "widget.h"

#include <sffdn/sffdn.h>

#include <imgui.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
void FillAttenuationFilterBank(sfFDN::AttenuationFilterBankOptions& filter_bank,
                               const sfFDN::attenuation_filter_variant_t& filter, uint32_t fdn_size)
{
    filter_bank.filter_configs.assign(fdn_size, filter);
}

enum class AttenuationFilterType
{
    Homogenous,
    TwoBand,
    ThreeBand,
    TenBand,
};

AttenuationFilterType GetAttenuationFilterType(const sfFDN::attenuation_filter_variant_t& filter)
{
    return std::visit(
        [](const auto& options) {
            using Filter = std::decay_t<decltype(options)>;
            if constexpr (std::is_same_v<Filter, sfFDN::HomogenousFilterOptions>)
            {
                return AttenuationFilterType::Homogenous;
            }
            else if constexpr (std::is_same_v<Filter, sfFDN::TwoBandFilterOptions>)
            {
                return AttenuationFilterType::TwoBand;
            }
            else if constexpr (std::is_same_v<Filter, sfFDN::ThreeBandFilterOptions>)
            {
                return AttenuationFilterType::ThreeBand;
            }
            else
            {
                return AttenuationFilterType::TenBand;
            }
        },
        filter);
}

sfFDN::attenuation_filter_variant_t CreateAttenuationFilter(AttenuationFilterType type)
{
    switch (type)
    {
    case AttenuationFilterType::Homogenous:
        return sfFDN::HomogenousFilterOptions{};
    case AttenuationFilterType::TwoBand:
        return sfFDN::TwoBandFilterOptions{};
    case AttenuationFilterType::ThreeBand:
        return sfFDN::ThreeBandFilterOptions{};
    case AttenuationFilterType::TenBand:
        return sfFDN::TenBandFilterOptions{};
    }
    std::unreachable();
}

bool DrawAttenuationFilterType(sfFDN::attenuation_filter_variant_t& filter)
{
    constexpr std::array<const char*, 4> kFilterTypes = {"Homogenous", "2 Bands", "3 Bands", "10 Bands"};
    int selected_type = static_cast<int>(GetAttenuationFilterType(filter));
    const int previous_type = selected_type;
    if (!ImGui::Combo("Filter Type", &selected_type, kFilterTypes.data(), kFilterTypes.size()) ||
        selected_type == previous_type)
    {
        return false;
    }
    filter = CreateAttenuationFilter(static_cast<AttenuationFilterType>(selected_type));
    return true;
}

sfFDN::feedback_matrix_variant_t ConvertFeedbackMatrix(const sfFDN::feedback_matrix_variant_t& matrix, bool cascaded)
{
    return std::visit(
        [cascaded](const auto& options) -> sfFDN::feedback_matrix_variant_t {
            if (cascaded)
            {
                sfFDN::CascadedFeedbackMatrixOptions converted;
                converted.matrix_size = options.matrix_size;
                converted.type = options.type;
                return converted;
            }
            sfFDN::ScalarFeedbackMatrixOptions converted;
            converted.matrix_size = options.matrix_size;
            converted.type = options.type;
            return converted;
        },
        matrix);
}
} // namespace

bool FDNToolboxApp::DrawFDNConfigurator()
{
    static uint32_t random_seed = 0;
    static const int max_delay = 6000;

    if (!ImGui::Begin("FDN Configurator"))
    {
        ImGui::End();
        return false;
    }

    bool config_changed = false;
    bool fdn_size_changed = false;
    config_changed |= DrawStructureSection(random_seed, fdn_size_changed);
    DrawVisualOverviewSection(fdn_size_changed, max_delay);
    config_changed |= DrawInputStageSection();
    config_changed |= DrawDelayNetworkSection();
    config_changed |= DrawFeedbackMatrixSection();
    config_changed |= DrawLoopFiltersSection();
    config_changed |= DrawOutputStageSection();
    config_changed |= DrawToneCorrectionSection();

    ImGui::End();
    return config_changed;
}

bool FDNToolboxApp::DrawStructureSection(uint32_t& random_seed, bool& fdn_size_changed)
{
    if (!fdn_sandbox::theme::BeginSection("Structure", true))
    {
        return false;
    }

    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Configure the network size, deterministic seed, and matrix orientation.");
    bool config_changed = false;
    static bool random_seed_checkbox = false;
    ImGui::Checkbox("Random Seed", &random_seed_checkbox);
    if (random_seed_checkbox)
    {
        ImGui::SameLine();
        config_changed |= ImGui::InputScalar("Seed", ImGuiDataType_U32, &random_seed, nullptr, nullptr, "%u");
    }
    else
    {
        random_seed = 0;
    }

    constexpr uint32_t kNMin = 4;
    constexpr uint32_t kNMax = 32;
    uint32_t fdn_size = fdn_config_.fdn_size;
    fdn_size_changed =
        ImGui::SliderScalar("N", ImGuiDataType_U32, &fdn_size, &kNMin, &kNMax, nullptr, ImGuiSliderFlags_AlwaysClamp);
    if (fdn_size_changed)
    {
        utils::ResizeFDNConfig(fdn_config_, fdn_size);
    }
    config_changed |= fdn_size_changed;
    fdn_config_.fdn_size = fdn_size;

    static bool transpose = false;
    if (ImGui::Checkbox("Transpose", &transpose))
    {
        fdn_config_.transposed = transpose;
        config_changed = true;
    }
    fdn_sandbox::theme::EndSection();
    return config_changed;
}

void FDNToolboxApp::DrawVisualOverviewSection(bool fdn_size_changed, int max_delay)
{
    if (!fdn_sandbox::theme::BeginSection("Visual Overview", true))
    {
        return;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Inspect gain distribution, delay lengths, and the active feedback matrix.");
    if (!fdn_size_changed)
    {
        DrawInputOutputGainsPlot(fdn_config_, gui_fdn_.get());
        DrawDelaysPlot(fdn_config_, max_delay);
        DrawFeedbackMatrixPlot(fdn_config_, gui_fdn_.get());
    }
    fdn_sandbox::theme::EndSection();
}

bool FDNToolboxApp::DrawInputStageSection()
{
    if (!fdn_sandbox::theme::BeginSection("Input Stage", true))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Set input gains and processors applied before the feedback loop.");
    bool changed = DrawFDNOptions(fdn_config_.input_block_config.parallel_gains_config, fdn_config_);
    changed |= DrawSingleChannelProcessorList(fdn_config_.input_block_config.single_channel_processors, fdn_config_);
    changed |= DrawMultiChannelProcessorList(fdn_config_.input_block_config.multichannel_processors, fdn_config_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FDNToolboxApp::DrawDelayNetworkSection()
{
    if (!fdn_sandbox::theme::BeginSection("Delay Network", true))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Configure delay lengths and interpolation for each feedback path.");
    const bool changed = DrawFDNOptions(fdn_config_.delay_bank_config, fdn_config_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FDNToolboxApp::DrawFeedbackMatrixSection()
{
    if (!fdn_sandbox::theme::BeginSection("Feedback Matrix"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Choose the matrix topology and whether it is implemented as cascaded stages.");

    bool changed = false;
    bool is_cascaded = std::holds_alternative<sfFDN::CascadedFeedbackMatrixOptions>(fdn_config_.feedback_matrix_config);
    if (ImGui::Checkbox("Cascaded", &is_cascaded))
    {
        changed = true;
        fdn_config_.feedback_matrix_config = ConvertFeedbackMatrix(fdn_config_.feedback_matrix_config, is_cascaded);
    }
    changed |= DrawFDNOptions(fdn_config_.feedback_matrix_config, fdn_config_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FDNToolboxApp::DrawLoopFiltersSection()
{
    if (!fdn_sandbox::theme::BeginSection("Loop Filters"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Shape decay independently or apply a shared attenuation profile.");

    assert(fdn_config_.attenuation_filter_bank_config.has_value());
    bool filter_bank_changed = utils::NormalizeAttenuationFilterBank(fdn_config_);
    auto filter_bank = fdn_config_.attenuation_filter_bank_config.value();
    auto shared_filter = filter_bank.filter_configs.front();

    static bool independent_t60s = false;
    const bool independence_changed = ImGui::Checkbox("Independent T60s", &independent_t60s);
    const bool filter_type_changed = DrawAttenuationFilterType(shared_filter);

    if (!independent_t60s)
    {
        const bool shared_filter_changed = DrawFDNOptions(shared_filter, fdn_config_);
        if (shared_filter_changed || filter_type_changed || independence_changed)
        {
            FillAttenuationFilterBank(filter_bank, shared_filter, fdn_config_.fdn_size);
            filter_bank_changed = true;
        }
    }
    else
    {
        if (filter_type_changed)
        {
            FillAttenuationFilterBank(filter_bank, shared_filter, fdn_config_.fdn_size);
            filter_bank_changed = true;
        }
        for (uint32_t i = 0; i < fdn_config_.fdn_size; ++i)
        {
            ImGui::PushID(i);
            filter_bank_changed |= DrawFDNOptions(filter_bank.filter_configs[i], fdn_config_);
            ImGui::PopID();
        }
    }

    if (filter_bank_changed)
    {
        fdn_config_.attenuation_filter_bank_config = filter_bank;
    }
    const bool processors_changed = DrawMultiChannelProcessorList(fdn_config_.loop_filter_configs, fdn_config_);
    fdn_sandbox::theme::EndSection();
    return filter_bank_changed || processors_changed;
}

bool FDNToolboxApp::DrawOutputStageSection()
{
    if (!fdn_sandbox::theme::BeginSection("Output Stage"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Set output gains and processors applied after the feedback loop.");
    bool changed = DrawFDNOptions(fdn_config_.output_block_config.parallel_gains_config, fdn_config_);
    changed |= DrawSingleChannelProcessorList(fdn_config_.output_block_config.single_channel_processors, fdn_config_);
    changed |= DrawMultiChannelProcessorList(fdn_config_.output_block_config.multichannel_processors, fdn_config_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FDNToolboxApp::DrawToneCorrectionSection()
{
    if (!fdn_sandbox::theme::BeginSection("Tone Correction"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Apply final single-channel tone-correction processing.");
    const bool changed = DrawSingleChannelProcessorList(fdn_config_.tone_correction_filters, fdn_config_);
    fdn_sandbox::theme::EndSection();
    return changed;
}
