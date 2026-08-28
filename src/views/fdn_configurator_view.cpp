#include "views/fdn_configurator_view.h"

#include "fdn_widget.h"
#include "session/fdn_session.h"
#include "sffdn/types.h"
#include "theme.h"
#include "utils.h"
#include "views/window_ids.h"
#include "widget.h"

#include <imgui.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <format>
#include <type_traits>
#include <utility>
#include <variant>

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

enum class FeedbackMatrixKind : int
{
    Scalar = 0,
    Cascaded = 1,
    TimeVarying = 2,
};

FeedbackMatrixKind GetFeedbackMatrixKind(const sfFDN::feedback_matrix_variant_t& matrix)
{
    if (std::holds_alternative<sfFDN::CascadedFeedbackMatrixOptions>(matrix))
    {
        return FeedbackMatrixKind::Cascaded;
    }
    if (std::holds_alternative<sfFDN::TimeVaryingFeedbackMatrixOptions>(matrix))
    {
        return FeedbackMatrixKind::TimeVarying;
    }
    return FeedbackMatrixKind::Scalar;
}

sfFDN::feedback_matrix_variant_t ConvertFeedbackMatrix(const sfFDN::feedback_matrix_variant_t& matrix,
                                                       FeedbackMatrixKind kind, float sample_rate)
{
    // The scalar-family types share matrix_size and type; the time-varying options share neither a scalar type nor
    // any modulation state with them, so conversion in either direction rebuilds from defaults.
    const auto matrix_size = std::visit([](const auto& options) { return options.matrix_size; }, matrix);
    const auto scalar_type = std::visit(
        [matrix_size](const auto& options) {
            using T = std::decay_t<decltype(options)>;
            if constexpr (std::is_same_v<T, sfFDN::TimeVaryingFeedbackMatrixOptions>)
            {
                return utils::IsPowerOfTwo(matrix_size) ? sfFDN::ScalarMatrixType::Hadamard
                                                        : sfFDN::ScalarMatrixType::Random;
            }
            else
            {
                return options.type;
            }
        },
        matrix);

    switch (kind)
    {
    case FeedbackMatrixKind::Cascaded:
    {
        sfFDN::CascadedFeedbackMatrixOptions converted;
        converted.matrix_size = matrix_size;
        converted.type = scalar_type;
        return converted;
    }
    case FeedbackMatrixKind::TimeVarying:
        return utils::MakeTimeVaryingFeedbackMatrix(matrix_size, sample_rate);
    case FeedbackMatrixKind::Scalar:
    default:
    {
        sfFDN::ScalarFeedbackMatrixOptions converted;
        converted.matrix_size = matrix_size;
        converted.type = scalar_type;
        return converted;
    }
    }
}
} // namespace

namespace fdn_sandbox::views
{

bool FdnConfiguratorView::Draw(session::FdnSession& session)
{
    if (!initialized_)
    {
        transpose_ = session.Draft().transposed;
        initialized_ = true;
    }
    if (!ImGui::Begin(ids::kFdnConfigurator))
    {
        ImGui::End();
        return false;
    }

    bool config_changed = false;
    const StructureSectionResult structure_result = DrawStructureSection(session);
    config_changed |= structure_result.config_changed;
    DrawVisualOverviewSection(session, structure_result.fdn_size_changed);
    config_changed |= DrawInputStageSection(session);
    config_changed |= DrawDelayNetworkSection(session);
    config_changed |= DrawFeedbackMatrixSection(session);
    config_changed |= DrawLoopFiltersSection(session);
    config_changed |= DrawOutputStageSection(session);
    config_changed |= DrawToneCorrectionSection(session);

    ImGui::End();
    return config_changed;
}

FdnConfiguratorView::StructureSectionResult FdnConfiguratorView::DrawStructureSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Structure", true))
    {
        return {};
    }

    StructureSectionResult result;
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Configure the network size, deterministic seed, and matrix orientation.");
    ImGui::Checkbox("Random Seed", &random_seed_enabled_);
    if (random_seed_enabled_)
    {
        ImGui::SameLine();
        result.config_changed |= ImGui::InputScalar("Seed", ImGuiDataType_U32, &random_seed_, nullptr, nullptr, "%u");
    }
    else
    {
        random_seed_ = 0;
    }

    constexpr uint32_t kNMin = 4;
    constexpr uint32_t kNMax = 32;
    auto& fdn_config = session.Draft();
    uint32_t fdn_size = fdn_config.fdn_size;
    result.fdn_size_changed =
        ImGui::SliderScalar("N", ImGuiDataType_U32, &fdn_size, &kNMin, &kNMax, nullptr, ImGuiSliderFlags_AlwaysClamp);
    if (result.fdn_size_changed)
    {
        utils::ResizeFDNConfig(fdn_config, fdn_size);
    }
    result.config_changed |= result.fdn_size_changed;
    fdn_config.fdn_size = fdn_size;

    if (ImGui::Checkbox("Transpose", &transpose_))
    {
        fdn_config.transposed = transpose_;
        result.config_changed = true;
    }
    fdn_sandbox::theme::EndSection();
    return result;
}

void FdnConfiguratorView::DrawVisualOverviewSection(session::FdnSession& session, bool fdn_size_changed)
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
        constexpr int kMaxDelay = 6000;
        DrawInputOutputGainsPlot(session.Draft(), session.Preview());
        DrawDelaysPlot(session.Draft(), kMaxDelay);

        // Drive the time-varying matrix animation from wall-clock time so the heatmap modulates at the rate the user
        // configured. The preview FDN is UI-owned and never processed, so nothing else advances its notion of time.
        const float sample_rate = session.Draft().sample_rate;
        const float frame_delta = ImGui::GetIO().DeltaTime;
        if (sample_rate > 0.0f && std::isfinite(frame_delta) && frame_delta > 0.0f)
        {
            widget_state_.feedback_matrix_animation_samples +=
                static_cast<double>(sample_rate) * static_cast<double>(frame_delta);
        }

        DrawFeedbackMatrixPlot(session.Draft(), session.Preview(), widget_state_.feedback_matrix,
                               static_cast<uint64_t>(widget_state_.feedback_matrix_animation_samples));
    }
    fdn_sandbox::theme::EndSection();
}

bool FdnConfiguratorView::DrawInputStageSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Input Stage", true))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Set input gains and processors applied before the feedback loop.");
    auto& fdn_config = session.Draft();
    bool changed = DrawFDNOptions(fdn_config.input_block_config.parallel_gains_config, fdn_config, widget_state_);
    changed |= DrawSingleChannelProcessorList(fdn_config.input_block_config.single_channel_processors, fdn_config,
                                              widget_state_);
    changed |=
        DrawMultiChannelProcessorList(fdn_config.input_block_config.multichannel_processors, fdn_config, widget_state_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FdnConfiguratorView::DrawDelayNetworkSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Delay Network", true))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Configure delay lengths and interpolation for each feedback path.");
    auto& fdn_config = session.Draft();
    const bool changed = DrawFDNOptions(fdn_config.delay_bank_config, fdn_config, widget_state_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FdnConfiguratorView::DrawFeedbackMatrixSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Feedback Matrix"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Choose the matrix topology and whether it is implemented as cascaded stages.");

    bool changed = false;
    auto& fdn_config = session.Draft();
    constexpr std::array<const char*, 3> kMatrixKindNames = {"Scalar", "Cascaded", "Time-Varying"};
    const auto current_kind = GetFeedbackMatrixKind(fdn_config.feedback_matrix_config);
    // A time-varying matrix is orthogonal by construction from 2x2 rotations, which an odd order cannot supply.
    const bool time_varying_supported = fdn_config.fdn_size >= 2 && (fdn_config.fdn_size % 2) == 0;

    // Labelled "Implementation" rather than "Matrix Type": the scalar and cascaded editors below already draw a
    // "Matrix Type" combo for the ScalarMatrixType, and ImGui derives widget IDs from labels, so reusing the label
    // makes the two collide. "Structure" is taken as well, by the section tree node above.
    if (ImGui::BeginCombo("Implementation", kMatrixKindNames.at(static_cast<size_t>(current_kind))))
    {
        for (size_t index = 0; index < kMatrixKindNames.size(); ++index)
        {
            const auto kind = static_cast<FeedbackMatrixKind>(index);
            const bool selectable = kind != FeedbackMatrixKind::TimeVarying || time_varying_supported;
            if (ImGui::Selectable(kMatrixKindNames.at(index), kind == current_kind,
                                  selectable ? ImGuiSelectableFlags_None : ImGuiSelectableFlags_Disabled) &&
                kind != current_kind)
            {
                fdn_config.feedback_matrix_config =
                    ConvertFeedbackMatrix(fdn_config.feedback_matrix_config, kind, fdn_config.sample_rate);
                changed = true;
            }
            if (!selectable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            {
                ImGui::SetTooltip("A time-varying matrix requires an even FDN size.");
            }
        }
        ImGui::EndCombo();
    }

    changed |= DrawFDNOptions(fdn_config.feedback_matrix_config, fdn_config, widget_state_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FdnConfiguratorView::DrawLoopFiltersSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Loop Filters"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Shape decay independently or apply a shared attenuation profile.");

    auto& fdn_config = session.Draft();
    assert(fdn_config.attenuation_filter_bank_config.has_value());
    bool filter_bank_changed = utils::NormalizeAttenuationFilterBank(fdn_config);
    auto filter_bank = fdn_config.attenuation_filter_bank_config.value();
    auto shared_filter = filter_bank.filter_configs.front();

    // Shared mode can only render the first channel, and any edit made in it overwrites every
    // channel. Optimizing per-channel attenuation (3-band RIR matching in particular) produces a
    // heterogeneous bank, so force independent mode to keep the view honest and stop the next
    // interaction from silently flattening the result.
    if (!utils::IsAttenuationFilterBankHomogeneous(filter_bank))
    {
        independent_t60s_ = true;
    }

    const bool independence_changed = ImGui::Checkbox("Independent T60s", &independent_t60s_);
    const bool filter_type_changed = DrawAttenuationFilterType(shared_filter);

    if (!independent_t60s_)
    {
        const bool shared_filter_changed = DrawFDNOptions(shared_filter, fdn_config, widget_state_);
        if (shared_filter_changed || filter_type_changed || independence_changed)
        {
            FillAttenuationFilterBank(filter_bank, shared_filter, fdn_config.fdn_size);
            filter_bank_changed = true;
        }
    }
    else
    {
        if (filter_type_changed)
        {
            FillAttenuationFilterBank(filter_bank, shared_filter, fdn_config.fdn_size);
            filter_bank_changed = true;
        }
        for (uint32_t i = 0; i < fdn_config.fdn_size; ++i)
        {
            ImGui::PushID(static_cast<int>(i));
            ImGui::SeparatorText(std::format("Channel {}", i).c_str());
            filter_bank_changed |= DrawFDNOptions(filter_bank.filter_configs[i], fdn_config, widget_state_);
            ImGui::PopID();
        }
    }

    if (filter_bank_changed)
    {
        fdn_config.attenuation_filter_bank_config = filter_bank;
    }
    const bool processors_changed =
        DrawMultiChannelProcessorList(fdn_config.loop_filter_configs, fdn_config, widget_state_);
    fdn_sandbox::theme::EndSection();
    return filter_bank_changed || processors_changed;
}

bool FdnConfiguratorView::DrawOutputStageSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Output Stage"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Set output gains and processors applied after the feedback loop.");
    auto& fdn_config = session.Draft();
    bool changed = DrawFDNOptions(fdn_config.output_block_config.parallel_gains_config, fdn_config, widget_state_);
    changed |= DrawSingleChannelProcessorList(fdn_config.output_block_config.single_channel_processors, fdn_config,
                                              widget_state_);
    changed |= DrawMultiChannelProcessorList(fdn_config.output_block_config.multichannel_processors, fdn_config,
                                             widget_state_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

bool FdnConfiguratorView::DrawToneCorrectionSection(session::FdnSession& session)
{
    if (!fdn_sandbox::theme::BeginSection("Tone Correction"))
    {
        return false;
    }
    fdn_sandbox::theme::TextWrapped(fdn_sandbox::theme::FontRole::Description,
                                    fdn_sandbox::theme::ColorRole::TextSecondary,
                                    "Apply final single-channel tone-correction processing.");
    auto& fdn_config = session.Draft();
    const bool changed = DrawSingleChannelProcessorList(fdn_config.tone_correction_filters, fdn_config, widget_state_);
    fdn_sandbox::theme::EndSection();
    return changed;
}

} // namespace fdn_sandbox::views
