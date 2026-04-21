#include "fdn_widget.h"

#include <imgui.h>
#include <implot.h>

#include <imfilebrowser.h>

#include <sffdn/sffdn.h>

#include "utils.h"

#include <random>
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
            bool is_selected = (selected_matrix_type == i);
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
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), " only supported for N that is a power of 2.");
            }
        }
        ImGui::EndCombo();
    }

    return config_changed;
}
} // namespace

bool FDNWidgetVisitor::operator()(sfFDN::ScalarFeedbackMatrixOptions& config)
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

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::CascadedFeedbackMatrixOptions& config)
{
    bool config_changed = false;
    if (config.matrix_size != fdn_config.fdn_size)
    {
        config.matrix_size = fdn_config.fdn_size;
        config_changed = true;
    }

    constexpr uint32_t kMaxStages = 10;
    constexpr uint32_t kMinStages = 0;
    config_changed |= ImGui::InputScalar("Stage Count", ImGuiDataType_U32, &config.stage_count);
    config.stage_count = std::clamp(config.stage_count, kMinStages, kMaxStages);

    constexpr float kMinSparsity = 1.f;
    constexpr float kMaxSparsity = 10.f;
    config_changed |= ImGui::SliderScalar("Sparsity", ImGuiDataType_Float, &config.sparsity, &kMinSparsity,
                                          &kMaxSparsity, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    config.sparsity = std::clamp(config.sparsity, kMinSparsity, kMaxSparsity);

    int selected_matrix_type = static_cast<int>(config.type);
    config_changed |= DrawScalarMatrixTypeComboBox(selected_matrix_type, fdn_config.fdn_size);

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::ModulationOptions& config)
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

bool FDNWidgetVisitor::operator()(sfFDN::ParallelGainsOptions& config)
{
    // TODO: should find a way to make this not static...
    static float min_gain = -1.f;
    static float max_gain = 1.f;

    bool config_changed = false;
    if (config.gains.size() != fdn_config.fdn_size)
    {
        config.gains.resize(fdn_config.fdn_size, 1.f / fdn_config.fdn_size);
        config_changed = true;
    }

    if (ImGui::Button("Distribute"))
    {
        config_changed = true;
        for (auto& gain : config.gains)
        {
            gain = 1.0f / fdn_config.fdn_size; // Distribute gains evenly
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Randomize"))
    {
        config_changed = true;
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(min_gain, max_gain);

        for (auto& gain : config.gains)
        {
            gain = distr(eng); // Generate random gains
        }
    }

    ImGui::SameLine();
    if (ImGui::BeginPopupContextItem("Gains Popup"))
    {
        static float value = 0.0f; // Default value
        if (ImGui::Selectable("Set to 0.5"))
        {
            value = 0.5f;
            config_changed = true;
        }
        if (ImGui::Selectable("Set to -0.5"))
        {
            value = -0.5f;
            config_changed = true;
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::DragFloat("##Value", &value, 0.01f, -1.0f, 1.0f))
        {
            config_changed = true;
        }

        for (auto& gain : config.gains)
        {
            gain = value; // Set all gains to the specified value
        }

        ImGui::EndPopup();
    }

    if (ImGui::Button("Set all to..."))
    {
        config_changed = true;
        ImGui::OpenPopup("Gains Popup");
    }

    ImGui::DragFloatRange2("Gain Range", &min_gain, &max_gain, 0.01f, -1.0f, 1.0f, "%.2f");

    const float spacing = 4;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(spacing, spacing));

    constexpr uint32_t max_sliders_per_row = 8;
    const uint32_t sliders_per_row = std::min(static_cast<uint32_t>(fdn_config.fdn_size), max_sliders_per_row);
    const uint32_t num_rows = (fdn_config.fdn_size + sliders_per_row - 1) / sliders_per_row;

    for (uint32_t row = 0; row < num_rows; ++row)
    {
        for (uint32_t i = 0; i < sliders_per_row; ++i)
        {
            size_t index = row * sliders_per_row + i;
            if (index >= fdn_config.fdn_size)
            {
                break;
            }
            if (i > 0)
            {
                ImGui::SameLine();
            }

            ImGui::PushID(static_cast<int>(index));
            config_changed |=
                ImGui::VSliderFloat("##v", ImVec2(18, 50), &config.gains[index], -1.f, 1.0f, "", ImGuiSliderFlags_None);
            if (ImGui::IsItemActive() || ImGui::IsItemHovered())
                ImGui::SetTooltip("%.3f", config.gains[index]);
            ImGui::PopID();
        }
    }
    ImGui::PopStyleVar();

    ImGui::Text("Adjust all:");
    ImGui::SameLine();
    if (ImGui::Button("-", ImVec2(30, 0)))
    {
        config_changed = true;
        for (auto& gain : config.gains)
        {
            gain *= 0.9f;
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("+", ImVec2(30, 0)))
    {
        config_changed = true;
        for (auto& gain : config.gains)
        {
            gain *= 1.1f;
        }
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::DelayOptions& config)
{
    bool config_changed = false;

    int delay_interpolation_type = static_cast<int>(config.interp_type);
    const std::string combo_preview_value = utils::GetDelayInterpolationTypeName(delay_interpolation_type);
    if (ImGui::BeginCombo("Interpolation Type", combo_preview_value.c_str()))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::DelayInterpolationType::Count); i++)
        {
            bool is_selected = (delay_interpolation_type == i);
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
        uint32_t delay_samples = static_cast<uint32_t>(config.delay);
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
        FDNWidgetVisitor modulation_visitor{fdn_config};
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

bool FDNWidgetVisitor::operator()(sfFDN::DelayBankOptions& config)
{
    bool config_changed = false;

    if (config.delays.size() != fdn_config.fdn_size)
    {
        config.delays.resize(fdn_config.fdn_size, 512.f);
        config_changed = true;
    }

    int delay_interpolation_type = static_cast<int>(config.interpolation_type);
    const std::string combo_preview_value = utils::GetDelayInterpolationTypeName(delay_interpolation_type);
    if (ImGui::BeginCombo("Interpolation Type", combo_preview_value.c_str()))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::DelayInterpolationType::Count); i++)
        {
            bool is_selected = (delay_interpolation_type == i);
            if (ImGui::Selectable(utils::GetDelayInterpolationTypeName(i).c_str(), is_selected))
            {
                delay_interpolation_type = i;
                config.interpolation_type = static_cast<sfFDN::DelayInterpolationType>(delay_interpolation_type);
                config_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    // This is so dumb...
    static std::map<std::ptrdiff_t, std::array<int, 2>> delay_ranges;
    if (delay_ranges.size() > 100)
    {
        delay_ranges.clear();
    }

    auto it = delay_ranges.find(reinterpret_cast<std::ptrdiff_t>(&config));
    if (it == delay_ranges.end())
    {
        delay_ranges[reinterpret_cast<std::ptrdiff_t>(&config)] = {256, 4000};
        it = delay_ranges.find(reinterpret_cast<std::ptrdiff_t>(&config));
    }

    int min_delay = it->second[0];
    int max_delay = it->second[1];
    ImGui::DragIntRange2("Delay Range", &min_delay, &max_delay, 1, fdn_config.block_size, 0, "%d samples", "%d samples",
                         ImGuiSliderFlags_AlwaysClamp);
    min_delay = std::max(min_delay, 0);
    max_delay = std::max(max_delay, min_delay + 1);

    it->second[0] = min_delay;
    it->second[1] = max_delay;

    if (ImGui::Button("Presets"))
    {
        ImGui::OpenPopup("Delay Presets");
    }

    if (ImGui::BeginPopup("Delay Presets"))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::DelayLengthType::Count); ++i)
        {
            if (ImGui::Selectable(utils::GetDelayLengthTypeName(i).c_str()))
            {
                config.delays = sfFDN::GetDelayLengths(fdn_config.fdn_size, min_delay, max_delay,
                                                       static_cast<sfFDN::DelayLengthType>(i));
                config_changed = true;
            }
        }
        ImGui::EndPopup();
    }

    for (size_t i = 0; i < config.delays.size(); ++i)
    {
        std::string label = "Delay " + std::to_string(i);
        if (config.interpolation_type == sfFDN::DelayInterpolationType::None)
        {
            int delay_samples = static_cast<int>(config.delays[i]);
            config_changed |= ImGui::SliderInt(label.c_str(), &delay_samples, min_delay, max_delay);
            config.delays[i] = static_cast<float>(delay_samples);
        }
        else
        {
            config_changed |= ImGui::SliderFloat(label.c_str(), &config.delays[i], static_cast<float>(min_delay),
                                                 static_cast<float>(max_delay));
        }
    }

    config.block_size = fdn_config.block_size;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::DelayBankTimeVaryingOptions& config)
{
    bool config_changed = false;

    if (config.delays.size() != fdn_config.fdn_size)
    {
        config.delays.resize(fdn_config.fdn_size, 512.f);
        config_changed = true;
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::SchroederAllpassSectionOptions& config)
{
    bool config_changed = false;

    int num_sections = static_cast<int>(config.delays.size());
    if (config.gains.size() != num_sections)
    {
        assert(false); // These should always be in sync, so this is unexpected
        config.gains.resize(num_sections, 0.5f);
        config_changed = true;
    }

    config_changed |= ImGui::InputInt("Section Count", &num_sections, 1, 1);
    num_sections = std::clamp(num_sections, 1, 10);

    config.delays.resize(num_sections, 512.f);
    config.gains.resize(num_sections, 0.5f);

    static int delay_min = 1;
    static int delay_max = 1000;
    if (ImGui::Button("Rand. Delays"))
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<uint32_t> distr(delay_min, delay_max);

        for (uint32_t i = 0; i < num_sections; ++i)
        {
            config.delays[i] = distr(eng); // Generate random delays
            config.delays[i] = utils::GetClosestPrime(static_cast<uint32_t>(config.delays[i]));
        }

        config_changed = true;
    }
    ImGui::SameLine();

    ImGui::DragIntRange2("Delay Range", &delay_min, &delay_max, 1, 1, 9999, "%d samples", "%d samples",
                         ImGuiSliderFlags_AlwaysClamp);

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
            config.delays[row] = delay;
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

bool FDNWidgetVisitor::operator()(sfFDN::MultichannelSchroederAllpassSectionOptions& config)
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

    static int delay_min = 1;
    static int delay_max = 1500;
    if (ImGui::Button("Randomize Delays"))
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_int_distribution<uint32_t> distr(delay_min, delay_max);

        for (auto& section : config.sections)
        {
            for (uint32_t i = 0; i < section.delays.size(); ++i)
            {
                section.delays[i] = distr(eng); // Generate random delays
                section.delays[i] = utils::GetClosestPrime(static_cast<uint32_t>(section.delays[i]));
            }
        }

        config_changed = true;
    }
    ImGui::SameLine();

    ImGui::DragIntRange2("Delay Range", &delay_min, &delay_max, 1, 1, 9999, "%d samples", "%d samples",
                         ImGuiSliderFlags_AlwaysClamp);

    if (ImGui::BeginTable("Schroeder Table", section_count + 1))
    {
        for (int col = 0; col < section_count; ++col)
        {
            std::string column_name = "Delay " + std::to_string(col + 1);
            ImGui::TableSetupColumn(column_name.c_str(), ImGuiTableColumnFlags_WidthFixed, 120.0f);
        }
        ImGui::TableSetupColumn("Gain", ImGuiTableColumnFlags_WidthFixed, 120.0f);

        ImGui::TableHeadersRow();

        for (int row = 0; row < config.sections.size(); ++row)
        {
            ImGui::TableNextRow();

            auto& section = config.sections[row];

            for (int col = 0; col < section.delays.size(); ++col)
            {
                ImGui::TableSetColumnIndex(col);
                int delay = static_cast<int>(section.delays[col]);
                ImGui::PushID((row * section_count) + col);

                config_changed |= ImGui::DragInt("##delay", &delay, 1, 1, 9999);
                delay = std::clamp(delay, 1, 9999);
                section.delays[col] = delay;
                ImGui::PopID();
            }

            ImGui::TableSetColumnIndex(section_count);
            ImGui::PushID(row + 1000);
            config_changed |= ImGui::DragFloat("##gain", &section.gains[0], 0.01f, -1.0f, 1.0f);
            section.gains[0] = std::clamp(section.gains[0], -1.0f, 1.0f);

            // For multichannel Schroeder, all gains in the section are the same
            std::fill(section.gains.begin(), section.gains.end(), section.gains[0]);
            ImGui::PopID();
        }

        ImGui::EndTable();
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::HomogenousFilterOptions& config)
{
    bool config_changed = false;

    config_changed |= ImGui::DragFloat("T60", &config.t60, 0.001f, 0.1f, 20.f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
    config.t60 = std::clamp(config.t60, 0.1f, 20.f);

    config.delay = -1.f; // Dummy value, should get updated in CreateFDNFromConfig
    config.sample_rate = fdn_config.sample_rate;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::TwoBandFilterOptions& config)
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

bool FDNWidgetVisitor::operator()(sfFDN::ThreeBandFilterOptions& config)
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

    config.freqs[0] = std::clamp(config.freqs[0], 125.f, 8000.f);
    config.freqs[1] = std::clamp(config.freqs[1], 8000.f, 16000.f);

    config.q = 1.f / std::numbers::sqrt2_v<float>; // Fixed Q value for now
    config.delay = -1.f;                           // Dummy value, should get updated in CreateFDNFromConfig
    config.sample_rate = fdn_config.sample_rate;

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::TenBandFilterOptions& config)
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
        if (i == 0 || i % 5 != 0)
        {
            ImGui::SameLine();
        }
    }
    ImGui::PopItemWidth();

    config.sample_rate = fdn_config.sample_rate;
    config.shelf_cutoff = 8000.f; // Fixed shelf cutoff for now
    config.delay = -1.f;          // Dummy value, should get updated in CreateFDNFromConfig

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::GraphicEQOptions&)
{
    return false;
}

bool FDNWidgetVisitor::operator()(sfFDN::AllpassFilterOptions&)
{
    return false;
}

bool FDNWidgetVisitor::operator()(sfFDN::CascadedBiquadsOptions&)
{
    return false;
}

bool FDNWidgetVisitor::operator()(sfFDN::FirOptions& config)
{
    bool config_changed = false;

    static std::map<std::ptrdiff_t, int> filter_type_map;
    if (filter_type_map.size() > 100)
    {
        filter_type_map.clear();
    }

    auto it = filter_type_map.find(reinterpret_cast<std::ptrdiff_t>(&config));
    if (it == filter_type_map.end())
    {
        filter_type_map[reinterpret_cast<std::ptrdiff_t>(&config)] = 0;
        it = filter_type_map.find(reinterpret_cast<std::ptrdiff_t>(&config));
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
        static ImGui::FileBrowser file_dialog;
        file_dialog.SetTitle("Select FIR file");
        file_dialog.SetTypeFilters({".wav"});
        if (ImGui::Button("Select FIR file"))
        {
            file_dialog.Open();
        }
        file_dialog.Display();
        if (file_dialog.HasSelected())
        {
            std::string filename = file_dialog.GetSelected().string();
            config.coeffs = utils::ReadAudioFile(filename, 0);
            config_changed = true;
            file_dialog.ClearSelected();
        }
    }
    else if (filter_type == 1)
    {
        config_changed |= DrawVelvetNoiseDecorrelatorConfig(config, fdn_config);
    }

    if (ImPlot::BeginPlot("Velvet decorrelator", ImVec2(-1, 200), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes("Samples", "Amplitude", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        // ImPlot::SetupAxisLimits(ImAxis_X1, 0, config.sequence.size(), ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0, 1.0, ImPlotCond_Always);

        ImPlot::PlotLine("Velvet Noise", config.coeffs.data(), config.coeffs.size());
        ImPlot::EndPlot();
    }

    return config_changed;
}

bool FDNWidgetVisitor::operator()(sfFDN::AttenuationFilterBankOptions&)
{
    return false;
}

bool DrawFDNOptions(sfFDN::DelayBankOptions& config, const sfFDN::FDNConfig& fdn_config)
{
    FDNWidgetVisitor widget{fdn_config};
    return widget(config);
}

bool DrawFDNOptions(sfFDN::ParallelGainsOptions& config, const sfFDN::FDNConfig& fdn_config)
{
    FDNWidgetVisitor widget{fdn_config};
    return widget(config);
}

bool DrawFDNOptions(sfFDN::attenuation_filter_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config)
{
    FDNWidgetVisitor widget{fdn_config};
    return std::visit(widget, config_variant);
}

bool DrawFDNOptions(sfFDN::single_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config)
{
    FDNWidgetVisitor widget{fdn_config};
    return std::visit(widget, config_variant);
}

bool DrawFDNOptions(sfFDN::multi_channel_processor_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config)
{
    FDNWidgetVisitor widget{fdn_config};
    return std::visit(widget, config_variant);
}

bool DrawFDNOptions(sfFDN::feedback_matrix_variant_t& config_variant, const sfFDN::FDNConfig& fdn_config)
{
    FDNWidgetVisitor widget{fdn_config};
    return std::visit(widget, config_variant);
}

bool DrawSingleChannelProcessorList(std::vector<sfFDN::single_channel_processor_variant_t>& processors,
                                    sfFDN::FDNConfig& fdn_config)
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
            std::ptrdiff_t processor_id = reinterpret_cast<std::ptrdiff_t>(&processor);
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
                config_changed |= DrawFDNOptions(processor, fdn_config);
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }

        // Remove processors in reverse order to avoid invalidating indices
        for (auto it = processors_to_remove.rbegin(); it != processors_to_remove.rend(); ++it)
        {
            processors.erase(processors.begin() + *it);
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
        ImGui::EndPopup();
    }

    return new_processor;
}

bool DrawMultiChannelProcessorList(std::vector<sfFDN::multi_channel_processor_variant_t>& processors,
                                   sfFDN::FDNConfig& fdn_config, bool is_loop_filter)
{
    bool config_changed = false;

    std::ptrdiff_t id = reinterpret_cast<std::ptrdiff_t>(&processors);
    ImGui::PushID(static_cast<int>(id));

    int index_offset = is_loop_filter ? 1 : 0;

    if (is_loop_filter)
    {
        assert(!processors.empty());
        assert(std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(processors[0]));
    }

    ImGui::SeparatorText("Multi-Channel Processors");
    if (processors.size() <= static_cast<size_t>(index_offset))
    {
        ImGui::Text("Empty");
    }
    else
    {
        std::vector<size_t> processors_to_remove;
        for (size_t i = index_offset; i < processors.size(); ++i)
        {
            auto& processor = processors[i];
            std::ptrdiff_t processor_id = reinterpret_cast<std::ptrdiff_t>(&processor);
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
                config_changed |= DrawFDNOptions(processor, fdn_config);
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }

        // Remove processors in reverse order to avoid invalidating indices
        for (auto it = processors_to_remove.rbegin(); it != processors_to_remove.rend(); ++it)
        {
            processors.erase(processors.begin() + *it);
            config_changed = true;
        }
    }

    if (ImGui::Button("Add..."))
    {
        ImGui::OpenPopup("multi_channel_processor_popup");
    }

    const std::array multi_channel_processor_names = {"Delays", "Schroeder Allpass", "Feedback Matrix"};
    std::optional<sfFDN::multi_channel_processor_variant_t> new_processor = std::nullopt;
    if (ImGui::BeginPopup("multi_channel_processor_popup"))
    {
        for (int i = 0; i < multi_channel_processor_names.size(); i++)
            if (ImGui::Selectable(multi_channel_processor_names[i]))
            {
                switch (i)
                {
                case 0:
                    processors.emplace_back(sfFDN::DelayBankOptions{});
                    config_changed = true;
                    break;
                case 1:
                    processors.emplace_back(sfFDN::MultichannelSchroederAllpassSectionOptions{});
                    config_changed = true;
                    break;
                case 2:
                    processors.emplace_back(sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = fdn_config.fdn_size});
                    config_changed = true;
                    break;
                default:
                    break;
                }
            }
        ImGui::EndPopup();
    }

    ImGui::PopID();
    return config_changed;
}

bool DrawVelvetNoiseDecorrelatorConfig(sfFDN::FirOptions& config, const sfFDN::FDNConfig&)
{
    bool config_changed = false;
    constexpr const char* kOvnSequences[] = {"decorrelator32_oVND15.wav", "decorrelator32_oVND30.wav"};
    static uint32_t selected_file = 0;
    if (ImGui::BeginCombo("OVN Sequence", kOvnSequences[selected_file]))
    {
        for (int i = 0; i < IM_ARRAYSIZE(kOvnSequences); i++)
        {
            bool is_selected = (selected_file == i);
            if (ImGui::Selectable(kOvnSequences[i], is_selected))
            {
                selected_file = i;
                config_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    std::filesystem::path file_path = std::filesystem::current_path() / "data" / kOvnSequences[selected_file];
    uint32_t max_channels = utils::GetChannelCountFromAudioFile(file_path.string());

    static uint32_t channel = 0;
    config_changed |= ImGui::InputInt("Channel", reinterpret_cast<int*>(&channel), 1, 1);
    channel = std::clamp(channel, 0u, max_channels > 0 ? max_channels - 1 : 0u);

    if (config_changed)
    {
        config.coeffs = utils::ReadAudioFile(file_path.string(), channel);
    }

    return config_changed;
}