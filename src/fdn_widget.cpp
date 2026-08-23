#include "fdn_widget.h"

#include <imgui.h>
#include <implot.h>

#include <imfilebrowser.h>

#include <sffdn/sffdn.h>

#include "theme.h"
#include "utils.h"
#include "widget.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

namespace
{
bool DrawScalarMatrixTypeComboBox(int& selected_matrix_type, uint32_t fdn_size)
{
    bool config_changed = false;
    const std::string combo_preview_value =
        utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type));
    if (ImGui::BeginCombo("Matrix Type", combo_preview_value.c_str()))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::ScalarMatrixType::Count); i++)
        {
            const bool is_selected = (selected_matrix_type == i);
            ImGuiSelectableFlags flags = ImGuiSelectableFlags_None;

            if (!utils::IsPowerOfTwo(fdn_size) &&
                (static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard ||
                 static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::VariableDiffusion))
            {
                flags |= ImGuiSelectableFlags_Disabled;
            }
            if (ImGui::Selectable(utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(i)).c_str(), is_selected,
                                  flags))
            {
                selected_matrix_type = i;
                config_changed = true;
            }

            if (!utils::IsPowerOfTwo(fdn_size) &&
                (static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard ||
                 static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::VariableDiffusion))
            {
                ImGui::SameLine();
                ImGui::TextColored(fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::StatusWarning),
                                   " only supported for N that is a power of 2.");
            }
        }
        ImGui::EndCombo();
    }

    return config_changed;
}

std::optional<std::vector<float>> ReadMatrixFromClipboard(uint32_t matrix_size)
{
    const char* clipboard_text = ImGui::GetClipboardText();
    if (clipboard_text == nullptr)
    {
        return std::nullopt;
    }

    const size_t matrix_count = static_cast<size_t>(matrix_size) * static_cast<size_t>(matrix_size);
    std::vector<float> matrix(matrix_count, 0.0f);
    std::istringstream clipboard_stream{std::string{clipboard_text}};
    std::string line;
    for (size_t row = 0; row < matrix_size && std::getline(clipboard_stream, line); ++row)
    {
        std::istringstream line_stream{line};
        for (size_t column = 0; column < matrix_size; ++column)
        {
            float value = 0.0f;
            if (!(line_stream >> value))
            {
                break;
            }
            const size_t index = (row * static_cast<size_t>(matrix_size)) + column;
            matrix[index] = value;
        }
    }

    return matrix;
}

FDNDelayRange& GetDelayRange(FDNWidgetState& state, const sfFDN::DelayBankOptions& config)
{
    auto& ranges = state.delay_widget.ranges;
    if (ranges.size() > 100)
    {
        ranges.clear();
    }

    return ranges.try_emplace(&config).first->second;
}

bool EnsureDelayCount(sfFDN::DelayBankOptions& config, uint32_t fdn_size)
{
    if (config.delays.size() == fdn_size)
    {
        return false;
    }

    config.delays.resize(fdn_size, 512.f);
    return true;
}

bool DrawDelayInterpolationType(sfFDN::DelayBankOptions& config)
{
    int interpolation_type = static_cast<int>(config.interpolation_type);
    const std::string preview_value = utils::GetDelayInterpolationTypeName(interpolation_type);
    if (!ImGui::BeginCombo("Interpolation Type", preview_value.c_str()))
    {
        return false;
    }

    bool config_changed = false;
    for (int index = 0; index < static_cast<int>(sfFDN::DelayInterpolationType::Count); ++index)
    {
        const bool is_selected = interpolation_type == index;
        if (ImGui::Selectable(utils::GetDelayInterpolationTypeName(index).c_str(), is_selected))
        {
            interpolation_type = index;
            config.interpolation_type = static_cast<sfFDN::DelayInterpolationType>(interpolation_type);
            config_changed = true;
        }
    }
    ImGui::EndCombo();
    return config_changed;
}

void DrawDelayRange(FDNDelayRange& range, uint32_t block_size)
{
    const int block_size_i = static_cast<int>(block_size);
    ImGui::DragIntRange2("Delay Range", &range.minimum, &range.maximum, 1, block_size_i, 0, "%d samples", "%d samples",
                         ImGuiSliderFlags_AlwaysClamp);
    range.minimum = std::max(range.minimum, 0);
    range.maximum = std::max(range.maximum, range.minimum + 1);
}

bool DrawDelayToolbar(bool& make_prime)
{
    if (ImGui::Button("Presets"))
    {
        ImGui::OpenPopup("Delay Presets");
    }

    ImGui::SameLine();
    const bool config_changed = ImGui::Checkbox("Make Prime", &make_prime);

    ImGui::SameLine();
    if (ImGui::Button("Ordering"))
    {
        ImGui::OpenPopup("Delay Ordering");
    }

    return config_changed;
}

bool DrawDelayPresetsPopup(sfFDN::DelayBankOptions& config, const sfFDN::FDNConfig& fdn_config,
                           const FDNDelayRange& range)
{
    if (!ImGui::BeginPopup("Delay Presets"))
    {
        return false;
    }

    bool config_changed = false;
    for (int index = 0; index < static_cast<int>(sfFDN::DelayLengthType::Count); ++index)
    {
        if (ImGui::Selectable(utils::GetDelayLengthTypeName(index).c_str()))
        {
            config.delays =
                sfFDN::GetDelayLengths(fdn_config.fdn_size, static_cast<float>(range.minimum),
                                       static_cast<float>(range.maximum), static_cast<sfFDN::DelayLengthType>(index));
            config_changed = true;
        }
    }
    ImGui::EndPopup();
    return config_changed;
}

bool DrawDelayOrderingPopup(std::span<float> delays)
{
    if (!ImGui::BeginPopup("Delay Ordering"))
    {
        return false;
    }

    bool config_changed = false;
    if (ImGui::Selectable("Randomize"))
    {
        std::random_device random_device;
        std::mt19937 engine(random_device());
        std::ranges::shuffle(delays, engine);
        config_changed = true;
    }
    if (ImGui::Selectable("Sort Ascending"))
    {
        std::ranges::sort(delays);
        config_changed = true;
    }
    if (ImGui::Selectable("Sort Descending"))
    {
        std::ranges::sort(delays, std::greater<>());
        config_changed = true;
    }
    ImGui::EndPopup();
    return config_changed;
}

bool DrawDelaySliders(sfFDN::DelayBankOptions& config, const FDNDelayRange& range)
{
    bool config_changed = false;
    for (const size_t index : std::views::iota(size_t{0}, config.delays.size()))
    {
        float& delay = config.delays[index];
        const std::string label = "Delay " + std::to_string(index);
        if (config.interpolation_type == sfFDN::DelayInterpolationType::None)
        {
            int delay_samples = static_cast<int>(delay);
            config_changed |= ImGui::SliderInt(label.c_str(), &delay_samples, range.minimum, range.maximum);
            delay = static_cast<float>(delay_samples);
            continue;
        }

        config_changed |= ImGui::SliderFloat(label.c_str(), &delay, static_cast<float>(range.minimum),
                                             static_cast<float>(range.maximum));
    }
    return config_changed;
}

void MakeDelaysPrime(std::span<float> delays)
{
    std::ranges::transform(delays, delays.begin(), [](float delay) {
        return static_cast<float>(utils::GetClosestPrime(static_cast<uint32_t>(delay)));
    });
}
} // namespace

bool FDNWidgetVisitor::operator()(sfFDN::ScalarFeedbackMatrixOptions& config) const
{
    bool config_changed = false;
    if (config.matrix_size != fdn_config.fdn_size)
    {
        config.matrix_size = fdn_config.fdn_size;
        config_changed = true;
    }

    int selected_matrix_type = static_cast<int>(config.type);
    config_changed |= DrawScalarMatrixTypeComboBox(selected_matrix_type, fdn_config.fdn_size);

    if (config_changed)
    {
        config.type = static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type);
        config.custom_matrix = sfFDN::GenerateMatrix(config.matrix_size, config.type, config.rng_seed, config.arg);
    }

    if (ImGui::Button("Read from clipboard"))
    {
        if (auto matrix = ReadMatrixFromClipboard(config.matrix_size))
        {
            config.custom_matrix = std::move(*matrix);
            config_changed = true;
        }
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::CascadedFeedbackMatrixOptions& config) const
{
    bool config_changed = false;
    if (config.matrix_size != fdn_config.fdn_size)
    {
        config.matrix_size = fdn_config.fdn_size;
        config_changed = true;
    }

    constexpr uint32_t kMaxStages = 10;
    constexpr uint32_t kMinStages = 0;
    constexpr uint32_t kStep = 1;
    config_changed |= ImGui::InputScalar("Stage Count", ImGuiDataType_U32, &config.stage_count, &kStep, nullptr, "%u",
                                         ImGuiInputTextFlags_None);
    config.stage_count = std::clamp(config.stage_count, kMinStages, kMaxStages);

    constexpr float kMinSparsity = 1.f;
    constexpr float kMaxSparsity = 10.f;
    config_changed |= ImGui::SliderScalar("Sparsity", ImGuiDataType_Float, &config.sparsity, &kMinSparsity,
                                          &kMaxSparsity, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    config.sparsity = std::clamp(config.sparsity, kMinSparsity, kMaxSparsity);

    int selected_matrix_type = static_cast<int>(config.type);
    config_changed |= DrawScalarMatrixTypeComboBox(selected_matrix_type, fdn_config.fdn_size);

    if (config_changed)
    {
        config.type = static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type);
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::ModulationOptions& config) const
{
    bool config_changed = false;

    float frequency_hz = config.frequency * fdn_config.sample_rate;

    std::array<float, 3> values = {frequency_hz, config.amplitude, config.initial_phase};

    config_changed |= ImGui::DragFloat3("Frequency (Hz) / Amplitude / Initial Phase", values.data(), 0.01f, 0.0f, 200.f,
                                        "%.2f", ImGuiSliderFlags_AlwaysClamp);

    // Check if field is being actively edited
    if (ImGui::IsItemActive() || ImGui::IsItemHovered())
    {
        config_changed = false;
        // ImGui::SetTooltip("Frequency: %.2f Hz\nAmplitude: %.2f\nInitial Phase: %.2f", values[0], values[1],
        // values[2]);
    }

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        config_changed = true;
    }

    // config_changed |= ImGui::DragScalar("Frequency", ImGuiDataType_Float, &frequency_hz, nullptr, nullptr,
    // "%.2f", 0);

    // ImGui::SameLine();
    // config_changed |=
    //     ImGui::InputScalar("Amplitude", ImGuiDataType_Float, &config.amplitude, nullptr, nullptr, "%.2f", 0);

    // ImGui::SameLine();
    // config_changed |= ImGui::DragScalar("Initial Phase", ImGuiDataType_Float, &config.initial_phase, 0.01f,
    //                                     &kMinInitialPhase, &kMaxInitialPhase, "%.2f",
    //                                     ImGuiSliderFlags_AlwaysClamp);

    constexpr float kMinInitialPhase = 0.f;
    constexpr float kMaxInitialPhase = 1.f;
    frequency_hz = std::clamp(values[0], 0.00f, 20.f);
    config.frequency = frequency_hz / fdn_config.sample_rate;
    config.amplitude = std::clamp(values[1], 0.00f, 200.0f);
    config.initial_phase = std::clamp(values[2], kMinInitialPhase, kMaxInitialPhase);
    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::ParallelGainsOptions& config) const
{
    bool config_changed = false;
    if (config.gains.size() != fdn_config.fdn_size)
    {
        config.gains.resize(fdn_config.fdn_size, 1.f / static_cast<float>(fdn_config.fdn_size));
        config_changed = true;
    }

    config_changed |= DrawGainsWidget(config.gains, state.minimum_gain, state.maximum_gain, state.set_all_gain);

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::DelayOptions& config) const
{
    bool config_changed = false;

    int delay_interpolation_type = static_cast<int>(config.interp_type);
    const std::string combo_preview_value = utils::GetDelayInterpolationTypeName(delay_interpolation_type);
    if (ImGui::BeginCombo("Interpolation Type", combo_preview_value.c_str()))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::DelayInterpolationType::Count); i++)
        {
            const bool is_selected = (delay_interpolation_type == i);
            if (ImGui::Selectable(utils::GetDelayInterpolationTypeName(i).c_str(), is_selected))
            {
                delay_interpolation_type = i;
                config_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    if (static_cast<sfFDN::DelayInterpolationType>(delay_interpolation_type) == sfFDN::DelayInterpolationType::None)
    {
        auto delay_samples = static_cast<uint32_t>(config.delay);
        config_changed |= ImGui::DragScalar("Delay", ImGuiDataType_U32, &delay_samples);
        config.delay = static_cast<float>(delay_samples);
    }
    else
    {
        config_changed |=
            ImGui::DragFloat("Delay", &config.delay, 0.001f, 1.0f, 10000.f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    }

    bool is_time_varying = config.lfo_config.has_value();
    if (ImGui::Checkbox("Time Varying", &is_time_varying))
    {
        config_changed = true;
        if (is_time_varying)
        {
            config.lfo_config = sfFDN::ModulationOptions{};
        }
        else
        {
            config.lfo_config = std::nullopt;
        }
    }

    if (is_time_varying)
    {
        const FDNWidgetVisitor modulation_visitor{fdn_config, state};
        config_changed |= modulation_visitor(*config.lfo_config);

        // Need to make sure that the modulation amplitude is not greater than the delay time
        if (config.lfo_config->amplitude > config.delay)
        {
            config.lfo_config->amplitude = config.delay - 5.f;
            config.lfo_config->amplitude = std::max(0.f, config.lfo_config->amplitude);
            config_changed = true;
        }
    }

    if (config_changed)
    {
        config.interp_type = static_cast<sfFDN::DelayInterpolationType>(delay_interpolation_type);
        config.max_delay = std::max(config.max_delay, static_cast<uint32_t>(config.delay + 128));
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::DelayBankOptions& config) const
{
    bool config_changed = EnsureDelayCount(config, fdn_config.fdn_size);
    config_changed |= DrawDelayInterpolationType(config);

    FDNDelayRange& range = GetDelayRange(state, config);
    DrawDelayRange(range, fdn_config.block_size);

    config_changed |= DrawDelayToolbar(state.delay_widget.make_prime);
    config_changed |= DrawDelayPresetsPopup(config, fdn_config, range);
    config_changed |= DrawDelayOrderingPopup(config.delays);
    config_changed |= DrawDelaySliders(config, range);

    if (config_changed && state.delay_widget.make_prime)
    {
        MakeDelaysPrime(config.delays);
    }

    config.block_size = fdn_config.block_size;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::DelayBankTimeVaryingOptions& config) const
{
    bool config_changed = false;

    if (config.delays.size() != fdn_config.fdn_size)
    {
        config.delays.resize(fdn_config.fdn_size, 512.f);
        config_changed = true;
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::SchroederAllpassSectionOptions& config) const
{
    bool config_changed = false;

    int num_sections = static_cast<int>(config.delays.size());
    if (config.gains.size() != config.delays.size())
    {
        assert(false); // These should always be in sync, so this is unexpected
        config.gains.resize(num_sections, 0.5f);
        config_changed = true;
    }

    config_changed |= ImGui::InputInt("Section Count", &num_sections, 1, 1);
    num_sections = std::clamp(num_sections, 1, 10);

    config.delays.resize(num_sections, 512.f);
    config.gains.resize(num_sections, 0.5f);

    if (ImGui::Button("Rand. Delays"))
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<uint32_t> distr(state.schroeder_delay_minimum, state.schroeder_delay_maximum);

        for (int i = 0; i < num_sections; ++i)
        {
            config.delays[i] = static_cast<float>(distr(eng)); // Generate random delays
            config.delays[i] = static_cast<float>(utils::GetClosestPrime(static_cast<uint32_t>(config.delays[i])));
        }

        config_changed = true;
    }
    ImGui::SameLine();

    ImGui::DragIntRange2("Delay Range", &state.schroeder_delay_minimum, &state.schroeder_delay_maximum, 1, 1, 9999,
                         "%d samples", "%d samples", ImGuiSliderFlags_AlwaysClamp);

    if (ImGui::BeginTable("Schroeder Table", num_sections + 1))
    {
        ImGui::TableSetupColumn("Delay", ImGuiTableColumnFlags_WidthFixed, 120.0f);
        ImGui::TableSetupColumn("Gain", ImGuiTableColumnFlags_WidthFixed, 120.0f);

        ImGui::TableHeadersRow();

        for (int row = 0; row < num_sections; ++row)
        {
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            int delay = static_cast<int>(config.delays[row]);
            ImGui::PushID(row);
            config_changed |= ImGui::DragInt("##delay", &delay, 1, 1, 9999);
            delay = std::clamp(delay, 1, 9999);
            config.delays[row] = static_cast<float>(delay);
            ImGui::PopID();

            ImGui::TableSetColumnIndex(1);
            ImGui::PushID(row + 1000);
            config_changed |= ImGui::DragFloat("##gain", &config.gains[row], 0.01f, -1.0f, 1.0f);
            config.gains[row] = std::clamp(config.gains[row], -1.0f, 1.0f);
            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::MultichannelSchroederAllpassSectionOptions& config) const
{
    bool config_changed = false;

    if (config.sections.size() != fdn_config.fdn_size)
    {
        config.sections.resize(fdn_config.fdn_size);
        config_changed = true;
    }

    int section_count = static_cast<int>(config.sections[0].delays.size());
    config_changed |= ImGui::InputInt("Section Count", &section_count, 1, 1);
    section_count = std::clamp(section_count, 1, 10);

    for (auto& section : config.sections)
    {
        if (section.delays.size() != section_count || section.gains.size() != section_count)
        {
            section.delays.resize(section_count, 512.f);
            section.gains.resize(section_count, 0.5f);
            config_changed = true;
        }
    }

    if (ImGui::Button("Randomize Delays"))
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<uint32_t> distr(state.multichannel_schroeder_delay_minimum,
                                                      state.multichannel_schroeder_delay_maximum);

        for (auto& section : config.sections)
        {
            for (auto& delay : section.delays)
            {
                delay = static_cast<float>(distr(eng)); // Generate random delays
                delay = static_cast<float>(utils::GetClosestPrime(static_cast<uint32_t>(delay)));
            }
        }

        config_changed = true;
    }
    ImGui::SameLine();

    ImGui::DragIntRange2("Delay Range", &state.multichannel_schroeder_delay_minimum,
                         &state.multichannel_schroeder_delay_maximum, 1, 1, 9999, "%d samples", "%d samples",
                         ImGuiSliderFlags_AlwaysClamp);

    if (ImGui::BeginTable("Schroeder Table", section_count + 1))
    {
        for (int col = 0; col < section_count; ++col)
        {
            const std::string column_name = "Delay " + std::to_string(col + 1);
            ImGui::TableSetupColumn(column_name.c_str(), ImGuiTableColumnFlags_WidthFixed, 120.0f);
        }
        ImGui::TableSetupColumn("Gain", ImGuiTableColumnFlags_WidthFixed, 120.0f);

        ImGui::TableHeadersRow();

        for (size_t row = 0; row < config.sections.size(); ++row)
        {
            ImGui::TableNextRow();

            auto& section = config.sections[row];

            for (size_t col = 0; col < section.delays.size(); ++col)
            {
                ImGui::TableSetColumnIndex(static_cast<int>(col));
                int delay = static_cast<int>(section.delays[col]);
                ImGui::PushID(static_cast<int>((row * static_cast<size_t>(section_count)) + col));

                config_changed |= ImGui::DragInt("##delay", &delay, 1, 1, 9999);
                delay = std::clamp(delay, 1, 9999);
                section.delays[col] = static_cast<float>(delay);
                ImGui::PopID();
            }

            ImGui::TableSetColumnIndex(section_count);
            ImGui::PushID(static_cast<int>(row) + 1000);
            config_changed |= ImGui::DragFloat("##gain", section.gains.data(), 0.01f, -1.0f, 1.0f);
            section.gains[0] = std::clamp(section.gains[0], -1.0f, 1.0f);

            // For multichannel Schroeder, all gains in the section are the same
            std::fill(section.gains.begin(), section.gains.end(), section.gains[0]);
            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::HomogenousFilterOptions& config) const
{
    bool config_changed = false;

    config_changed |= ImGui::DragFloat("T60", &config.t60, 0.001f, 0.1f, 20.f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
    config.t60 = std::clamp(config.t60, 0.1f, 20.f);

    config.delay = -1.f; // Dummy value, should get updated in CreateFDNFromConfig
    config.sample_rate = fdn_config.sample_rate;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::TwoBandFilterOptions& config) const
{
    bool config_changed = false;

    config_changed |=
        ImGui::DragFloat2("T60s", config.t60s.data(), 0.001f, 0.1f, 20.f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
    for (auto& t60 : config.t60s)
    {
        t60 = std::clamp(t60, 0.1f, 20.f);
    }

    config.delay = -1.f; // Dummy value, should get updated in CreateFDNFromConfig
    config.sample_rate = fdn_config.sample_rate;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::ThreeBandFilterOptions& config) const
{
    bool config_changed = false;

    config_changed |=
        ImGui::DragFloat3("T60s", config.t60s.data(), 0.001f, 0.1f, 20.f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
    for (auto& t60 : config.t60s)
    {
        t60 = std::clamp(t60, 0.1f, 20.f);
    }

    config_changed |=
        ImGui::DragFloat2("Cutoff", config.freqs.data(), 1.f, 125.f, 16000.f, "%.1f Hz", ImGuiSliderFlags_AlwaysClamp);

    if (ImGui::Button("Edit"))
    {
        state.show_three_band_designer = true;
    }

    if (state.show_three_band_designer)
    {
        config_changed |= Draw3BandDesigner({.t60s = config.t60s, .frequencies = config.freqs},
                                            state.show_three_band_designer, state.three_band_designer);
    }

    config.freqs[0] = std::clamp(config.freqs[0], 125.f, 8000.f);
    config.freqs[1] = std::clamp(config.freqs[1], 8000.f, 16000.f);

    config.q = 1.f / std::numbers::sqrt2_v<float>; // Fixed Q value for now
    config.delay = -1.f;                           // Dummy value, should get updated in CreateFDNFromConfig
    config.sample_rate = fdn_config.sample_rate;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::TenBandFilterOptions& config) const
{
    bool config_changed = false;

    // const char* freq_labels[10] = {"31.25 Hz", "62.5 Hz", "125 Hz", "250 Hz", "500 Hz",
    //                                "1 kHz",    "2 kHz",   "4 kHz",  "8 kHz",  "16 kHz"};

    ImGui::PushItemWidth(40);
    for (auto i = 0u; i < config.t60s.size(); ++i)
    {
        auto& t60 = config.t60s[i];
        ImGui::PushID(static_cast<int>(i));
        config_changed |= ImGui::DragFloat("##T60", &t60, 0.001f, 0.1f, 20.0f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
        t60 = std::clamp(t60, 0.1f, 20.f);
        ImGui::PopID();
        ImGui::SameLine();
        if (i == 4)
        {
            ImGui::NewLine();
        }
    }
    ImGui::PopItemWidth();
    ImGui::NewLine();

    if (ImGui::Button("Edit"))
    {
        state.show_ten_band_designer = true;
    }

    if (state.show_ten_band_designer)
    {
        config_changed |= DrawFilterDesigner(config.t60s, state.show_ten_band_designer, state.ten_band_designer);
    }

    config.sample_rate = fdn_config.sample_rate;
    config.shelf_cutoff = 8000.f; // Fixed shelf cutoff for now
    config.delay = -1.f;          // Dummy value, should get updated in CreateFDNFromConfig

    return config_changed;
}

bool FDNWidgetVisitor::operator()([[maybe_unused]] sfFDN::GraphicEQOptions& config) const
{
    (void)config;
    return false;
}

bool FDNWidgetVisitor::operator()([[maybe_unused]] sfFDN::AllpassFilterOptions& config) const
{
    (void)config;
    return false;
}

bool FDNWidgetVisitor::operator()([[maybe_unused]] sfFDN::CascadedBiquadsOptions& config) const
{
    (void)config;
    return false;
}

bool FDNWidgetVisitor::operator()(sfFDN::FirOptions& config) const
{
    bool config_changed = false;

    if (state.fir_filter_types.size() > 100)
    {
        state.fir_filter_types.clear();
    }

    auto it = state.fir_filter_types.find(reinterpret_cast<std::ptrdiff_t>(&config));
    if (it == state.fir_filter_types.end())
    {
        state.fir_filter_types[reinterpret_cast<std::ptrdiff_t>(&config)] = 0;
        it = state.fir_filter_types.find(reinterpret_cast<std::ptrdiff_t>(&config));
    }

    int filter_type = it->second;
    config_changed |= ImGui::RadioButton("File", &filter_type, 0);
    ImGui::SameLine();
    config_changed |= ImGui::RadioButton("Velvet", &filter_type, 1);

    it->second = filter_type;

    if (filter_type == 0)
    {
        if (config.coeffs.empty())
        {
            config.coeffs = {1.f};
        }
        state.fir_file_dialog.SetTitle("Select FIR file");
        state.fir_file_dialog.SetTypeFilters({".wav"});
        if (ImGui::Button("Select FIR file"))
        {
            state.fir_file_dialog.Open();
        }
        state.fir_file_dialog.Display();
        if (state.fir_file_dialog.HasSelected())
        {
            const std::string filename = state.fir_file_dialog.GetSelected().string();
            config.coeffs = utils::ReadAudioFile(filename, 0);
            config_changed = true;
            state.fir_file_dialog.ClearSelected();
        }
    }
    else if (filter_type == 1)
    {
        config_changed |= DrawVelvetNoiseDecorrelatorConfig(config, fdn_config, state);
    }

    if (ImPlot::BeginPlot("Velvet decorrelator", ImVec2(-1, 200), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes("Samples", "Amplitude", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        // ImPlot::SetupAxisLimits(ImAxis_X1, 0, config.sequence.size(), ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0, 1.0, ImPlotCond_Always);

        ImPlot::PlotLine("Velvet Noise", config.coeffs.data(), static_cast<int>(config.coeffs.size()));
        ImPlot::EndPlot();
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::MultichannelFirOptions& config) const
{
    bool config_changed = false;

    if (config.coeffs.size() != fdn_config.fdn_size)
    {
        config.coeffs.resize(fdn_config.fdn_size, std::vector<float>{1.f});
        config_changed = true;
    }

    // Only support velvet noise decorrelator for now
    constexpr const std::array<const char*, 2> kOvnSequences = {"decorrelator32_oVND15.wav",
                                                                "decorrelator32_oVND30.wav"};
    if (ImGui::BeginCombo("OVN Sequence", kOvnSequences[state.multichannel_fir_file]))
    {
        for (size_t i = 0; i < kOvnSequences.size(); ++i)
        {
            const bool is_selected = (state.multichannel_fir_file == i);
            if (ImGui::Selectable(kOvnSequences[i], is_selected))
            {
                state.multichannel_fir_file = i;
                config_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    const std::filesystem::path file_path =
        std::filesystem::current_path() / "data" / kOvnSequences.at(state.multichannel_fir_file);
    const uint32_t max_channels = utils::GetChannelCountFromAudioFile(file_path.string());

    config_changed |= ImGui::InputInt("Channel", reinterpret_cast<int*>(&state.multichannel_fir_channel), 1, 1);
    state.multichannel_fir_channel =
        std::clamp(state.multichannel_fir_channel, 0u, max_channels > 0 ? max_channels - 1 : 0u);

    if (config_changed)
    {
        for (uint32_t i = state.multichannel_fir_channel; i < state.multichannel_fir_channel + fdn_config.fdn_size; ++i)
        {
            config.coeffs[i - state.multichannel_fir_channel] =
                utils::ReadAudioFile(file_path.string(), i % max_channels);
        }
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()([[maybe_unused]] sfFDN::AttenuationFilterBankOptions& config) const
{
    (void)config;
    return false;
}

bool DrawFDNOptions(sfFDN::DelayBankOptions& config, const sfFDN::FDNConfig& fdn_config, FDNWidgetState& state)
{
    const FDNWidgetVisitor widget{fdn_config, state};
    return widget(config);
}

bool DrawFDNOptions(sfFDN::ParallelGainsOptions& config, const sfFDN::FDNConfig& fdn_config, FDNWidgetState& state)
{
    const FDNWidgetVisitor widget{fdn_config, state};
    return widget(config);
}

bool DrawFDNOptions(sfFDN::attenuation_filter_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state)
{
    FDNWidgetVisitor widget{fdn_config, state};
    return std::visit(widget, config_variant);
}

bool DrawFDNOptions(sfFDN::single_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state)
{
    FDNWidgetVisitor widget{fdn_config, state};
    return std::visit(widget, config_variant);
}

bool DrawFDNOptions(sfFDN::multi_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state)
{
    FDNWidgetVisitor widget{fdn_config, state};
    return std::visit(widget, config_variant);
}

bool DrawFDNOptions(sfFDN::feedback_matrix_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config,
                    FDNWidgetState& state)
{
    FDNWidgetVisitor widget{fdn_config, state};
    return std::visit(widget, config_variant);
}

bool DrawSingleChannelProcessorList(std::vector<sfFDN::single_channel_processor_variant_t>& processors,
                                    sfFDN::FDNConfig& fdn_config, FDNWidgetState& state)
{
    bool config_changed = false;
    ImGui::SeparatorText("Single Channel Processors");
    if (processors.empty())
    {
        ImGui::Text("Empty");
    }
    else
    {
        std::vector<size_t> processors_to_remove;
        for (size_t i = 0; i < processors.size(); ++i)
        {
            auto& processor = processors[i];
            auto processor_id = reinterpret_cast<std::ptrdiff_t>(&processor);
            ImGui::PushID(static_cast<int>(processor_id));
            if (ImGui::Button("edit"))
            {
                ImGui::OpenPopup("edit_processor_popup");
            }
            ImGui::SameLine();
            if (ImGui::Button("remove"))
            {
                processors_to_remove.push_back(i);
            }
            ImGui::SameLine();
            ImGui::Text("%s %zu", utils::GetProcessorName(processor).c_str(), i + 1);

            if (ImGui::BeginPopup("edit_processor_popup", ImGuiWindowFlags_AlwaysAutoResize))
            {
                config_changed |= DrawFDNOptions(processor, fdn_config, state);
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }

        // Remove processors in reverse order to avoid invalidating indices
        for (auto& it : std::ranges::reverse_view(processors_to_remove))
        {
            processors.erase(processors.begin() + static_cast<std::ptrdiff_t>(it));
            config_changed = true;
        }
    }

    if (ImGui::Button("Add..."))
    {
        ImGui::OpenPopup("single_channel_processor_popup");
    }

    auto new_processor = DrawAddSingleChannelProcessorPopup();
    if (new_processor.has_value())
    {
        processors.push_back(std::move(*new_processor));
        config_changed = true;
    }

    return config_changed;
}

std::optional<sfFDN::single_channel_processor_variant_t> DrawAddSingleChannelProcessorPopup()
{
    const std::array single_channel_processor_names = {"Delay", "Schroeder Allpass", "Velvet Noise Decorrelator"};
    std::optional<sfFDN::single_channel_processor_variant_t> new_processor = std::nullopt;
    if (ImGui::BeginPopup("single_channel_processor_popup"))
    {
        for (int i = 0; i < single_channel_processor_names.size(); i++)
        {
            if (ImGui::Selectable(single_channel_processor_names[i]))
            {
                switch (i)
                {
                case 0:
                    new_processor = sfFDN::DelayOptions{};
                    break;
                case 1:
                    new_processor =
                        sfFDN::SchroederAllpassSectionOptions{.delays = {47}, .gains = {0.7f}, .parallel = false};
                    break;
                case 2:
                    new_processor = sfFDN::FirOptions{};
                    break;
                default:
                    break;
                }
            }
        }
        ImGui::EndPopup();
    }

    return new_processor;
}

bool DrawMultiChannelProcessorList(std::vector<sfFDN::multi_channel_processor_variant_t>& processors,
                                   sfFDN::FDNConfig& fdn_config, FDNWidgetState& state)
{
    bool config_changed = false;

    auto id = reinterpret_cast<std::ptrdiff_t>(&processors);
    ImGui::PushID(static_cast<int>(id));

    ImGui::SeparatorText("Multi-Channel Processors");
    if (processors.empty())
    {
        ImGui::Text("Empty");
    }
    else
    {
        std::vector<size_t> processors_to_remove;
        for (size_t i = 0; i < processors.size(); ++i)
        {
            auto& processor = processors[i];
            auto processor_id = reinterpret_cast<std::ptrdiff_t>(&processor);
            ImGui::PushID(static_cast<int>(processor_id));
            if (ImGui::Button("edit"))
            {
                ImGui::OpenPopup("edit_processor_popup");
            }
            ImGui::SameLine();
            if (ImGui::Button("remove"))
            {
                processors_to_remove.push_back(i);
            }
            ImGui::SameLine();

            ImGui::Text("%s %zu", utils::GetProcessorName(processor).c_str(), i + 1);

            if (ImGui::BeginPopup("edit_processor_popup", ImGuiWindowFlags_AlwaysAutoResize))
            {
                config_changed |= DrawFDNOptions(processor, fdn_config, state);
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }

        // Remove processors in reverse order to avoid invalidating indices
        for (const auto index : std::views::reverse(processors_to_remove))
        {
            processors.erase(processors.begin() + static_cast<std::ptrdiff_t>(index));
            config_changed = true;
        }
    }

    if (ImGui::Button("Add..."))
    {
        ImGui::OpenPopup("multi_channel_processor_popup");
    }

    auto new_processor = DrawAddMultiChannelProcessorPopup(fdn_config);
    if (new_processor.has_value())
    {
        processors.push_back(std::move(*new_processor));
        config_changed = true;
    }

    ImGui::PopID();
    return config_changed;
}

std::optional<sfFDN::multi_channel_processor_variant_t> DrawAddMultiChannelProcessorPopup(
    const sfFDN::FDNConfig& fdn_config)
{
    const std::array multi_channel_processor_names = {"Delays", "Schroeder Allpass", "Feedback Matrix",
                                                      "Velvet Noise Decorrelator"};
    std::optional<sfFDN::multi_channel_processor_variant_t> new_processor = std::nullopt;
    if (ImGui::BeginPopup("multi_channel_processor_popup"))
    {
        for (int i = 0; i < multi_channel_processor_names.size(); i++)
        {
            if (ImGui::Selectable(multi_channel_processor_names[i]))
            {
                switch (i)
                {
                case 0:
                {
                    auto delay_bank_options =
                        sfFDN::DelayBankOptions{.delays = std::vector<float>(fdn_config.fdn_size, 1.f),
                                                .block_size = fdn_config.block_size,
                                                .interpolation_type = sfFDN::DelayInterpolationType::None};
                    new_processor.emplace(delay_bank_options);
                    break;
                }
                case 1:
                {
                    auto schroeder_options = sfFDN::MultichannelSchroederAllpassSectionOptions{
                        .sections = std::vector<sfFDN::SchroederAllpassSectionOptions>(
                            fdn_config.fdn_size,
                            sfFDN::SchroederAllpassSectionOptions{.delays = {47}, .gains = {0.7f}})};
                    new_processor.emplace(schroeder_options);
                    break;
                }
                case 2:
                    new_processor.emplace(sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = fdn_config.fdn_size});
                    break;
                case 3:
                    new_processor.emplace(sfFDN::MultichannelFirOptions{
                        .coeffs = std::vector<std::vector<float>>(fdn_config.fdn_size, std::vector<float>{1.f})});
                    break;
                default:
                    break;
                }
            }
        }
        ImGui::EndPopup();
    }
    return new_processor;
}

bool DrawVelvetNoiseDecorrelatorConfig(sfFDN::FirOptions& config, const sfFDN::FDNConfig& fdn_config,
                                       FDNWidgetState& state)
{
    bool config_changed = false;
    (void)fdn_config;
    constexpr std::array<const char*, 2> kOvnSequences = {"decorrelator32_oVND15.wav", "decorrelator32_oVND30.wav"};
    if (ImGui::BeginCombo("OVN Sequence", kOvnSequences.at(state.velvet_file)))
    {
        for (size_t i = 0; i < kOvnSequences.size(); ++i)
        {
            const bool is_selected = (state.velvet_file == i);
            if (ImGui::Selectable(kOvnSequences.at(i), is_selected))
            {
                state.velvet_file = i;
                config_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    const std::filesystem::path file_path = std::filesystem::current_path() / "data" / kOvnSequences[state.velvet_file];
    const uint32_t max_channels = utils::GetChannelCountFromAudioFile(file_path.string());

    config_changed |= ImGui::InputInt("Channel", reinterpret_cast<int*>(&state.velvet_channel), 1, 1);
    state.velvet_channel = std::clamp(state.velvet_channel, 0u, max_channels > 0 ? max_channels - 1 : 0u);

    if (config_changed)
    {
        config.coeffs = utils::ReadAudioFile(file_path.string(), state.velvet_channel);
    }

    return config_changed;
}