#include "widget.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>

#include <implot.h>

#include <sffdn/sffdn.h>
#include <string>
#include <sys/types.h>

#include "imgui.h"
#include "settings.h"
#include "utils.h"

namespace
{
constexpr size_t kNBands = 10;

std::vector<float> GetMatrixFromClipboard(uint32_t N)
{
    std::vector<float> feedback_matrix(N * N, 0.0f);
    const char* clipboard_text = ImGui::GetClipboardText();
    std::string clipboard_string(clipboard_text);

    // split by newline
    std::vector<std::string> lines;
    std::istringstream stream(clipboard_string);
    std::string line;
    while (std::getline(stream, line))
    {
        lines.push_back(line);
    }

    size_t line_count = std::min(lines.size(), static_cast<size_t>(N));

    for (size_t i = 0; i < line_count; ++i)
    {
        // Each line should have N values separated by spaces or tabs
        std::istringstream line_stream(lines[i]);
        std::vector<float> values;
        float value = 0;
        while (line_stream >> value)
        {
            values.push_back(value);
        }

        // Only keep the first N values for each row
        for (size_t j = 0; j < N && j < values.size(); ++j)
        {
            feedback_matrix[i * N + j] = values[j];
        }
    }

    return feedback_matrix;
}

bool DrawFilterDesigner(std::span<float> t60s, bool& show_delay_filter_designer)
{
    if (!ImGui::Begin("Filter Designer", &show_delay_filter_designer))
    {
        ImGui::End();
        return false;
    }

    bool config_changed = false;

    constexpr uint32_t kTestDelay = 593.f; // arbitrary, needed to design the filter for the GUI

    // Number of bands in the filter designer
    assert(t60s.size() == kNBands);
    static std::vector<float> frequencies(0, 0.f);
    std::vector<float> gains(kNBands, 0.0f);

    // Oversampled vectors for plotting
    static std::vector<float> gains_plot;
    static std::vector<float> t60s_plot;
    static std::vector<float> frequencies_plot;
    static std::vector<float> filter_freqs_plot;

    bool point_changed = false;

    if (frequencies.size() == 0)
    {
        frequencies.resize(kNBands);
        constexpr float kUpperLimit = 16000.0f;
        for (size_t i = 0; i < kNBands; ++i)
        {
            frequencies[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
        }
    }

    if (frequencies_plot.size() == 0) // Only runs on first call
    {
        frequencies_plot = frequencies; // utils::LogSpace(std::log10(frequencies[0] + 1e-6f),
                                        // std::log10(frequencies.back() - 1.f), 256);
        filter_freqs_plot = utils::LogSpace(std::log10(1.f), std::log10(Settings::Instance().SampleRate() / 2.f), 512);
        t60s_plot = utils::pchip(frequencies, t60s, frequencies_plot);

        gains = utils::T60ToGainsDb(t60s, kTestDelay, Settings::Instance().SampleRate());
        gains_plot = gains;   // utils::pchip(frequencies, gains, frequencies_plot);
        point_changed = true; // Force initial plot update
    }

    if (ImPlot::BeginPlot("Filter Designer", ImVec2(-1, ImGui::GetWindowHeight() * 0.45f), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes("Frequency (Hz)", "RT60 (s)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.01f, 5.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, Settings::Instance().SampleRate() / 2);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, 0.f, 10.f);

        static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

        for (size_t i = 0; i < frequencies.size(); ++i)
        {
            double freq = frequencies[i]; // The frequency should stay constant
            double t60 = t60s[i];
            point_changed |= ImPlot::DragPoint(i, &freq, &t60, ImVec4(0, 0.9f, 0, 0), 10);
            t60s[i] = std::clamp(static_cast<float>(t60), 0.01f, 10.0f); // Update the t60 value
        }

        if (point_changed)
        {
            t60s_plot = utils::pchip(frequencies, t60s, frequencies_plot);
        }

        // Plot the RT60 values
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f);
        ImPlot::PlotScatter("RT60", frequencies.data(), t60s.data(), t60s.size());

        ImPlot::PlotLine("RT60 Line", frequencies_plot.data(), t60s_plot.data(), t60s_plot.size());
        ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.250f);
        ImPlot::PlotShaded("RT60 Area", frequencies_plot.data(), t60s_plot.data(), t60s_plot.size(), 0.f);

        ImPlot::EndPlot();
    }

    static std::vector<float> H;
    static float shelf_cutoff = 8000.f;
    point_changed |= ImGui::SliderFloat("Shelf Cutoff (Hz)", &shelf_cutoff, 1000.0f, 10000.0f, "%.0f Hz");

    if (point_changed)
    {
        gains = utils::T60ToGainsDb(t60s, kTestDelay, Settings::Instance().SampleRate());
        gains_plot = gains; // utils::pchip(frequencies, gains, frequencies_plot);
        std::vector<float> t60s_f(t60s.begin(), t60s.end());
        std::vector<float> sos =
            sfFDN::GetTwoFilter(t60s_f, kTestDelay, Settings::Instance().SampleRate(), shelf_cutoff);

        H = utils::AbsFreqz(sos, filter_freqs_plot, Settings::Instance().SampleRate());

        // To db gain
        for (float& i : H)
        {
            i = 20.f * std::log10(i);
        }
    }

    if (ImPlot::BeginPlot("Filter preview", ImVec2(-1, ImGui::GetWindowHeight() * 0.45f), ImPlotFlags_None))
    {

        ImPlot::SetupAxes("Frequency (Hz)", "Gain (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, Settings::Instance().SampleRate() / 2.f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -10.0f, 1.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, Settings::Instance().SampleRate() / 2.f);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -60.f, 10.f);

        static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

        ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.70f, 0.20f, 1.0f), 4.0f);
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f);
        ImPlot::PlotLine("Target Gain", frequencies_plot.data(), gains_plot.data(), frequencies_plot.size());

        if (H.size() > 0)
        {
            ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.20f, 0.20f, 1.0f), 3.0f);
            ImPlot::PlotLine("Filter Response", filter_freqs_plot.data(), H.data(), filter_freqs_plot.size());
        }

        ImPlot::EndPlot();
    }

    if (point_changed)
    {
        config_changed = true;
    }

    if (ImGui::Button("Apply"))
    {
        config_changed = true;
        show_delay_filter_designer = false;
    }

    ImGui::End();
    return config_changed;
}

void PlotCascadedFeedbackMatrix(const sfFDN::CascadedFeedbackMatrixInfo& info)
{
    // 2 stage per row
    int num_rows = (info.K + 1) / 2; // +1 to handle odd K

    int subplot_height = num_rows * 200; // Height of each subplot row

    constexpr ImPlotAxisFlags axes_flags = ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels;
    if (ImPlot::BeginSubplots("Cascaded Feedback Matrix", num_rows, 4, ImVec2(800, subplot_height),
                              ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
        for (size_t i = 0; i < info.K; ++i)
        {

            std::span<const float> matrix = std::span(info.matrices).subspan(i * info.N * info.N, info.N * info.N);

            if (ImPlot::BeginPlot("##matrix", ImVec2(50, 50), ImPlotFlags_CanvasOnly))
            {
                ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
                const char* label_fmt = info.N < 4 ? "%.2f" : nullptr; // Adjust label format based on N size
                ImPlot::PlotHeatmap("heat", matrix.data(), info.N, info.N, -1, 1, label_fmt, ImPlotPoint(0, 0),
                                    ImPlotPoint(1, 1), 0);

                ImPlot::EndPlot();
            }

            // The last stage does not have delays
            if (i < info.K - 1)
            {
                std::span<const uint32_t> delays =
                    std::span(info.delays.data() + i * info.N, info.N); // Delays for the current stage
                if (ImPlot::BeginPlot("##delays", ImVec2(50, 50), ImPlotFlags_NoLegend))
                {
                    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_None, axes_flags);

                    ImPlot::PlotBars("delays", delays.data(), info.N, 0.5, 0, ImPlotBarsFlags_Horizontal);
                    ImPlot::EndPlot();
                }
            }
        }

        ImPlot::EndSubplots();
    }
}

bool DrawGainsWidget(std::span<float> gains)
{
    bool config_changed = false;
    const size_t N = gains.size();
    if (ImGui::Button("Distribute"))
    {
        config_changed = true;
        for (uint32_t i = 0; i < N; ++i)
        {
            gains[i] = 1.0f / N; // Distribute gains evenly
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Randomize"))
    {
        config_changed = true;
        std::random_device rd;                                    // Obtain a random number from hardware
        std::mt19937 eng(rd());                                   // Seed the generator
        std::uniform_real_distribution<float> distr(-1.0f, 1.0f); // Define the range

        for (uint32_t i = 0; i < N; ++i)
        {
            gains[i] = distr(eng); // Generate random gains
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

        for (uint32_t i = 0; i < N; ++i)
        {
            gains[i] = value; // Set all gains to the specified value
        }

        ImGui::EndPopup();
    }

    if (ImGui::Button("Set all to..."))
    {
        config_changed = true;
        ImGui::OpenPopup("Gains Popup");
    }

    for (uint32_t i = 0; i < N; ++i)
    {
        std::string label = "Input Gain " + std::to_string(i + 1);
        config_changed |= ImGui::SliderFloat(label.c_str(), &gains[i], -1.f, 1.0f, "%.2f");
    }

    return config_changed;
}
} // namespace

void DrawInputOutputGainsPlot(const FDNConfig& config)
{
    if (ImPlot::BeginSubplots("##Input/Output_Gains", 2, 1, ImVec2(300, 200),
                              ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
        constexpr ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        if (ImPlot::BeginPlot("Input Gains", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);
            ImPlot::SetupAxesLimits(-0.45, config.input_gains.size() - 0.45, -1, 1, ImPlotCond_Always);
            ImPlot::PlotBars("Input Gains", config.input_gains.data(), config.input_gains.size(), 0.90, 0,
                             ImPlotBarsFlags_None);
            ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("Output Gains", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);
            ImPlot::SetupAxesLimits(-0.45, config.output_gains.size() - 0.45, -1, 1, ImPlotCond_Always);
            ImPlot::PlotBars("Output Gains", config.output_gains.data(), config.output_gains.size(), 0.90, 0,
                             ImPlotBarsFlags_None);
            ImPlot::EndPlot();
        }

        ImPlot::EndSubplots();
    }
}

void DrawDelaysPlot(const FDNConfig& config, uint32_t max_delay)
{
    if (ImPlot::BeginPlot("Delays", ImVec2(300, 100), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
        constexpr ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);

        ImPlot::SetupAxesLimits(-1, config.delays.size(), 0, max_delay, ImPlotCond_Always);

        ImPlot::PlotBars("##Delays", config.delays.data(), config.delays.size(), 0.90, 0, ImPlotBarsFlags_None);
        ImPlot::EndPlot();
    }
}

void DrawFeedbackMatrixPlot(const FDNConfig& config)
{
    constexpr ImPlotColormap feedback_matrix_colormap = ImPlotColormap_Plasma;

    static std::vector<float> feedback_matrix;
    const uint32_t N = config.N;

    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::vector<float>>)
            {
                feedback_matrix = arg;
            }
            else if constexpr (std::is_same_v<T, sfFDN::CascadedFeedbackMatrixInfo>)
            {
                auto first_matrix = std::span(arg.matrices).first(N * N);

                feedback_matrix.resize(N * N);
                std::copy(first_matrix.begin(), first_matrix.end(), feedback_matrix.begin());
            }
        },
        config.matrix_info);

    ImPlot::PushColormap(feedback_matrix_colormap);
    if (ImPlot::BeginPlot("Feedback Matrix", ImVec2(300, 300), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {

        constexpr ImPlotAxisFlags axes_flags =
            ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel | ImPlotAxisFlags_NoTickLabels;
        ImPlot::SetupAxes(nullptr, nullptr, axes_flags, axes_flags);
        const char* label_fmt = N < 10 ? "%.2f" : nullptr; // Adjust label format based on N size
        ImPlot::PlotHeatmap("heat", feedback_matrix.data(), N, N, -1, 1, label_fmt, ImPlotPoint(0, 0),
                            ImPlotPoint(1, 1), 0);

        ImPlot::EndPlot();
    }
    ImPlot::PopColormap();
}

bool DrawInputGainsWidget(FDNConfig& config)
{
    if (config.input_gains.size() != config.N)
    {
        config.input_gains.resize(config.N, 0.5f);
    }

    bool config_changed = false;
    if (ImGui::TreeNode("Edit input gains"))
    {
        config_changed |= DrawGainsWidget(config.input_gains);
        ImGui::TreePop();
    }

    return config_changed;
}

bool DrawOutputGainsWidget(FDNConfig& config)
{
    if (config.output_gains.size() != config.N)
    {
        config.output_gains.resize(config.N, 0.5f);
    }

    bool config_changed = false;
    if (ImGui::TreeNode("Edit output gains"))
    {
        config_changed |= DrawGainsWidget(config.output_gains);
        ImGui::TreePop();
    }

    return config_changed;
}

bool DrawDelayLengthsWidget(FDNConfig& config, int& min_delay, int& max_delay, uint32_t random_seed)
{
    bool config_changed = false;
    bool should_update_delays = false;
    static int selected_delay_length_type = static_cast<int>(sfFDN::DelayLengthType::Random);
    static int selected_sort_type = 0;
    bool sort_type_changed = false;

    static float mean_delay = 50.f;
    static float std_dev = 2.8f;

    const uint32_t N = config.N;

    if (config.delays.size() != N)
    {
        config.delays.resize(N, min_delay);
        should_update_delays = true;
    }

    if (ImGui::TreeNode("Edit delays"))
    {
        constexpr int kMinDelay = 100; // Minimum delay in samples

        if (ImGui::BeginCombo("Delay Length Type", utils::GetDelayLengthTypeName(selected_delay_length_type).c_str()))
        {
            for (int i = 0; i < static_cast<int>(sfFDN::DelayLengthType::Count) + 1; i++)
            {
                bool is_selected = (selected_delay_length_type == i);
                if (ImGui::Selectable(utils::GetDelayLengthTypeName(i).c_str(), is_selected))
                {
                    selected_delay_length_type = i;
                    should_update_delays = true;
                }
            }
            ImGui::EndCombo();
        }

        if (selected_delay_length_type < static_cast<int>(sfFDN::DelayLengthType::Count))
        {
            if (ImGui::DragIntRange2("Delay Range", &min_delay, &max_delay, 1, kMinDelay, 0, "%d samples", "%d samples",
                                     ImGuiSliderFlags_AlwaysClamp))
            {
                should_update_delays = true;
            }
            // Need to manually clamp because DragIntRange2 doesn't seem to respect the min/max values
            min_delay = std::clamp(min_delay, kMinDelay, max_delay);
            max_delay = std::max(max_delay, kMinDelay);
        }
        else
        {
            should_update_delays |= ImGui::SliderFloat("Mean Delay", &mean_delay, 0.f, 1000.f);
            should_update_delays |= ImGui::SliderFloat("Std Dev", &std_dev, 0.2f, 3.f);
        }

        static bool clamp_to_prime = false;
        config_changed |= ImGui::Checkbox("Clamp to Prime", &clamp_to_prime);

        constexpr std::array<const char*, 3> sort_type = {"None", "Ascending", "Descending"};
        if (ImGui::BeginCombo("Sort Type", sort_type[selected_sort_type]))
        {
            for (int i = 0; i < sort_type.size(); ++i)
            {
                bool is_selected = (selected_sort_type == i);
                if (ImGui::Selectable(sort_type[i], is_selected))
                {
                    selected_sort_type = i;
                    sort_type_changed = true;
                }
            }
            ImGui::EndCombo();
        }

        for (uint32_t i = 0; i < N; ++i)
        {
            std::string label = "Delay " + std::to_string(i + 1);
            int delay = static_cast<int>(config.delays[i]);
            config_changed |= ImGui::SliderInt(label.c_str(), &delay, kMinDelay, max_delay, nullptr);

            config.delays[i] = std::max<size_t>(static_cast<size_t>(delay), kMinDelay);

            if (clamp_to_prime)
            {
                config.delays[i] = utils::GetClosestPrime(config.delays[i]);
            }
        }

        if (should_update_delays)
        {
            config_changed = true;
            if (selected_delay_length_type < static_cast<int>(sfFDN::DelayLengthType::Count))
            {
                config.delays = sfFDN::GetDelayLengths(N, min_delay, max_delay,
                                                       static_cast<sfFDN::DelayLengthType>(selected_delay_length_type),
                                                       random_seed);
            }
            else
            {
                assert(selected_delay_length_type == static_cast<int>(sfFDN::DelayLengthType::Count));
                // Mean delay
                config.delays =
                    sfFDN::GetDelayLengthsFromMean(N, mean_delay, std_dev, Settings::Instance().SampleRate());
            }
            sort_type_changed = true; // Force sort if we updated delays
        }

        if (sort_type_changed)
        {
            config_changed = true;
            if (selected_sort_type == 1) // Ascending
            {
                std::ranges::sort(config.delays);
            }
            else if (selected_sort_type == 2) // Descending
            {
                std::ranges::sort(config.delays, std::greater<uint32_t>());
            }
        }

        ImGui::TreePop();
    }

    return config_changed;
}

bool DrawExtraDelayWidget(FDNConfig& config, bool force_update)
{
    bool config_changed = force_update;

    constexpr int kMinDelay = 0;
    constexpr int kMaxDelay = 1000;

    const uint32_t N = config.N;
    config.input_stage_delays.resize(N, 0);

    for (uint32_t i = 0; i < N; ++i)
    {
        std::string label = "Delay " + std::to_string(i + 1);
        int delay = static_cast<int>(config.input_stage_delays[i]);
        config_changed |= ImGui::SliderInt(label.c_str(), &delay, kMinDelay, kMaxDelay, nullptr);

        config.input_stage_delays[i] = std::max<size_t>(static_cast<size_t>(delay), kMinDelay);
    }

    return config_changed;
}

bool DrawExtraSchroederAllpassWidget(FDNConfig& config, bool force_update)
{
    bool config_changed = force_update;

    const uint32_t N = config.N;
    config.schroeder_allpass_delays.resize(N, 0);
    config.schroeder_allpass_gains.resize(N, 0.0f);

    if (ImGui::Button("Edit"))
    {
        ImGui::OpenPopup("Edit Schroeder Section");
    }

    if (ImGui::BeginPopupModal("Edit Schroeder Section", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        static int section_count = 1;
        ImGui::InputInt("Section Count", &section_count, 1, 1);
        section_count = std::clamp(section_count, 1, 10);

        static std::vector<uint32_t> delays(section_count * config.N, 0);
        delays.resize(section_count * config.N, 0);

        static std::vector<float> gains(config.N, 0.0f);

        if (ImGui::BeginTable("Schroeder Table", section_count + 1))
        {
            for (int col = 0; col < section_count; ++col)
            {
                std::string column_name = "Delay " + std::to_string(col + 1);
                ImGui::TableSetupColumn(column_name.c_str(), ImGuiTableColumnFlags_WidthFixed, 120.0f);
            }
            ImGui::TableSetupColumn("Gain", ImGuiTableColumnFlags_WidthFixed, 120.0f);

            ImGui::TableHeadersRow();

            for (int row = 0; row < config.N; ++row)
            {
                ImGui::TableNextRow();

                for (int col = 0; col < section_count; ++col)
                {
                    ImGui::TableSetColumnIndex(col);
                    int delay = static_cast<int>(delays[(row * section_count) + col]);
                    ImGui::PushID((row * section_count) + col);

                    ImGui::DragInt("##delay", &delay, 1, 1, 9999);
                    delay = std::clamp(delay, 1, 9999);
                    delays[(row * section_count) + col] = delay;
                    ImGui::PopID();
                }

                ImGui::TableSetColumnIndex(section_count);
                ImGui::PushID(row + 1000);
                ImGui::DragFloat("##gain", &gains[row], 0.01f, -1.0f, 1.0f);
                gains[row] = std::clamp(gains[row], -1.0f, 1.0f);
                ImGui::PopID();
            }

            ImGui::EndTable();
        }

        if (ImGui::Button("Apply"))
        {
            config.schroeder_allpass_delays = delays;
            config.schroeder_allpass_gains = gains;
            config_changed = true;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
    return config_changed;
}

bool DrawScalarMatrixWidget(FDNConfig& config, uint32_t random_seed)
{
    bool config_changed = false;
    bool should_update_feedback_matrix = false;
    static bool cascade_matrix = false;
    static float sparsity = 1.f;
    static int num_stages = 2;
    bool manual_edit = false;
    static float diffusion_theta = 1.f;

    static std::vector<float> feedback_matrix;
    static sfFDN::CascadedFeedbackMatrixInfo cascaded_feedback_matrix_info;

    const uint32_t N = config.N;
    if (feedback_matrix.size() != N * N)
    {
        feedback_matrix.resize(N * N, 0.0f);
        should_update_feedback_matrix = true;
    }

    static int selected_matrix_type = 4; // Default to Hadamard
    if (ImGui::TreeNode("Edit matrix"))
    {
        const std::string combo_preview_value =
            utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type));
        if (ImGui::BeginCombo("Matrix Type", combo_preview_value.c_str()))
        {
            for (int i = 0; i < static_cast<int>(sfFDN::ScalarMatrixType::Count); i++)
            {
                bool is_selected = (selected_matrix_type == i);

                ImGuiSelectableFlags flags = ImGuiSelectableFlags_None;

                if (!utils::IsPowerOfTwo(N) &&
                    (static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard ||
                     static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::VariableDiffusion))
                {
                    flags |= ImGuiSelectableFlags_Disabled;
                }

                if (ImGui::Selectable(utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(i)).c_str(),
                                      is_selected, flags))
                {
                    selected_matrix_type = i;
                    should_update_feedback_matrix = true;
                }

                if (!utils::IsPowerOfTwo(N) &&
                    (static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard ||
                     static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::VariableDiffusion))
                {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), " only supported for N that is a power of 2.");
                }
            }
            ImGui::EndCombo();
        }

        if (selected_matrix_type == static_cast<int>(sfFDN::ScalarMatrixType::VariableDiffusion))
        {
            should_update_feedback_matrix |= ImGui::SliderFloat("Diffusion Theta", &diffusion_theta, 0.f, 1.0f, "%.3f");
        }

        if (ImGui::Button("Manual Edit"))
        {
            ImGui::OpenPopup("Matrix Edit Popup");
        }

        if (ImGui::BeginPopupModal("Matrix Edit Popup", nullptr,
                                   ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize))
        {
            if (ImGui::Button("Clear"))
            {
                feedback_matrix.assign(N * N, 0.0f);
                config_changed = true;
                manual_edit = true;
                should_update_feedback_matrix = true;
            }

            ImGui::SameLine();
            if (ImGui::Button("Read from clipboard"))
            {
                feedback_matrix = GetMatrixFromClipboard(N);
            }

            ImGui::PushItemWidth(50);
            for (size_t i = 0; i < N; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    if (j != 0)
                    {
                        ImGui::SameLine(); // Align inputs in a grid
                    }
                    ImGui::PushID((i * N) + j);
                    config_changed |= ImGui::InputScalar("##MatrixValue", ImGuiDataType_Float,
                                                         &feedback_matrix[(i * N) + j], nullptr, nullptr, "%.3f", 0);
                    ImGui::PopID();

                    manual_edit |= config_changed;
                    should_update_feedback_matrix |= config_changed;
                }
            }

            ImGui::PopItemWidth();

            if (ImGui::Button("Set"))
            {
                ImGui::CloseCurrentPopup();
                manual_edit = true;
                should_update_feedback_matrix = true;
            }
            ImGui::EndPopup();
        }

        should_update_feedback_matrix |= ImGui::Checkbox("Cascade Matrix", &cascade_matrix);
        if (cascade_matrix)
        {
            should_update_feedback_matrix |= ImGui::SliderFloat("Sparsity", &sparsity, 1.f, 10.0f);
            should_update_feedback_matrix |= ImGui::InputInt("Num Stages", &num_stages, 1, 1);

            num_stages = std::clamp(num_stages, 1, 10);
            sparsity = std::clamp(sparsity, 1.f, 10.f);

            if (ImGui::Button("View"))
            {
                // Open a new window to display the matrix
                ImGui::OpenPopup("Cascaded Matrix View");
            }

            if (ImGui::BeginPopupModal("Cascaded Matrix View", nullptr,
                                       ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize))
            {
                PlotCascadedFeedbackMatrix(cascaded_feedback_matrix_info);

                if (ImGui::Button("OK", ImVec2(120, 0)))
                {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }
        }
        ImGui::TreePop();
    }

    if (should_update_feedback_matrix)
    {
        auto matrix_type = static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type);
        if (manual_edit)
        {
            config.matrix_info = feedback_matrix;
        }
        else if (matrix_type == sfFDN::ScalarMatrixType::NestedAllpass)
        {
            std::vector<float> input_gains(N, 0.0f);
            std::vector<float> output_gains(N, 0.0f);
            feedback_matrix = sfFDN::NestedAllpassMatrix(N, random_seed, input_gains, output_gains);

            config.input_gains = input_gains;
            config.output_gains = output_gains;
            config.matrix_info = feedback_matrix;
        }
        else
        {
            std::optional<float> extra_arg = std::nullopt;
            if (matrix_type == sfFDN::ScalarMatrixType::VariableDiffusion)
            {
                extra_arg = diffusion_theta;
            }

            feedback_matrix = sfFDN::GenerateMatrix(N, matrix_type, random_seed, extra_arg);

            config.matrix_info = feedback_matrix;
        }

        if (cascade_matrix)
        {
            cascaded_feedback_matrix_info = sfFDN::ConstructCascadedFeedbackMatrix(
                N, num_stages, sparsity, static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type), 1.f);

            config.matrix_info = cascaded_feedback_matrix_info;
        }

        config_changed = true;
    }

    return config_changed;
}

bool DrawDelayFilterWidget(FDNConfig& config)
{
    bool config_changed = false;
    static float feedback_gain = 0.9999f;
    static float t60_dc = 2.f;
    static float t60_ny = 1.f;
    static std::vector<float> t60s(kNBands, 2.f);

    static bool show_delay_filter_designer = false;
    static DelayFilterType delay_filter_type = DelayFilterType::Proportional;
    if (ImGui::TreeNode("Delay Filters"))
    {
        constexpr std::array<const char*, 3> kFilterTypeNames = {"Proportional", "One Pole", "Octave Band Filter"};
        static int selected_filter_type = 0;
        const char* combo_preview_value = kFilterTypeNames[selected_filter_type];
        if (ImGui::BeginCombo("Filter Type", combo_preview_value))
        {
            for (int i = 0; i < kFilterTypeNames.size(); i++)
            {
                bool is_selected = (selected_filter_type == i);
                if (ImGui::Selectable(kFilterTypeNames[i], is_selected))
                {
                    delay_filter_type = static_cast<DelayFilterType>(i);
                    selected_filter_type = i;
                    config_changed = true;
                }
            }
            ImGui::EndCombo();
        }

        // Proportinal feedback Gain
        if (delay_filter_type == DelayFilterType::Proportional)
        {
            const float kFbGainStep = 0.01f;
            const float kFbGainStepFast = 0.25f;
            config_changed |= ImGui::InputScalar("RT60", ImGuiDataType_Float, &feedback_gain, &kFbGainStep,
                                                 &kFbGainStepFast, "%.5f", 0);
            feedback_gain = std::clamp(feedback_gain, 0.1f, 10.0f);
        }
        else if (delay_filter_type == DelayFilterType::OnePole) // One Pole
        {
            constexpr float kOffsetFromStart = 125.f;
            ImGui::Text("RT60 DC: ");
            ImGui::SameLine(kOffsetFromStart);
            ImGui::SetNextItemWidth(200);

            config_changed |= (ImGui::InputFloat("RT60 DC", &t60_dc, 0.01f, 0.1f, "%.2f"));
            t60_dc = std::clamp(t60_dc, 0.01f, 10.0f);

            ImGui::Text("RT60 Nyquist: ");
            ImGui::SameLine(kOffsetFromStart);
            ImGui::SetNextItemWidth(200);

            config_changed |= (ImGui::InputFloat("RT60 Nyquist", &t60_ny, 0.01f, 0.1f, "%.2f"));
            t60_ny = std::clamp(t60_ny, 0.01f, 10.0f);
        }
        else if (delay_filter_type == DelayFilterType::TwoFilter)
        {
            if (t60s.size() != kNBands)
            {
                t60s.resize(kNBands, 1.f);
            }
            if (ImGui::Button("Edit"))
            {
                show_delay_filter_designer = true;
                config_changed = true;
            }
        }

        if (show_delay_filter_designer)
        {
            config_changed |= DrawFilterDesigner(t60s, show_delay_filter_designer);
        }
        ImGui::TreePop();
    }

    if (config_changed)
    {
        if (delay_filter_type == DelayFilterType::Proportional)
        {
            config.attenuation_t60s.resize(1);
            config.attenuation_t60s[0] = feedback_gain;
        }
        else if (delay_filter_type == DelayFilterType::OnePole)
        {
            config.attenuation_t60s = {{t60_dc, t60_ny}};
        }
        else if (delay_filter_type == DelayFilterType::TwoFilter)
        {
            config.attenuation_t60s = t60s;
        }
    }

    return config_changed;
}

bool DrawToneCorrectionFilterDesigner(FDNConfig& config)
{
    static bool show_tc_filter_designer = false;
    static std::vector<float> tc_gains(kNBands, 0.f);
    static std::vector<float> frequencies(0);
    static std::vector<float> frequencies_plot;   // Oversampled vectors for plotting
    static std::vector<float> frequency_response; // Frequency response for plotting
    static bool enabled = false;

    if (frequencies.size() == 0)
    {
        frequencies.resize(kNBands);
        constexpr float kUpperLimit = 16000.0f;
        for (size_t i = 0; i < kNBands; ++i)
        {
            frequencies[i] = kUpperLimit / std::pow(2.0f, static_cast<float>(kNBands - 1 - i));
        }
    }

    bool point_changed = false;
    if (frequencies_plot.size() == 0) // Only runs on first call
    {
        frequencies_plot = utils::LogSpace(std::log10(1.f), std::log10(Settings::Instance().SampleRate() / 2.f), 1024);

        point_changed = true; // Force initial plot update
    }

    bool config_changed = false;

    if (ImGui::TreeNode("Tone Correction Filters"))
    {
        if (ImGui::Checkbox("Enabled", &enabled))
        {
            config_changed = true;
        }

        if (enabled)
        {
            if (ImGui::Button("Edit"))
            {
                show_tc_filter_designer = true;
                config_changed = true;
            }
        }

        ImGui::TreePop();
    }

    if (show_tc_filter_designer && ImGui::Begin("Filter Designer"))
    {

        if (ImPlot::BeginPlot("Filter preview", ImVec2(-1, ImGui::GetWindowHeight() * 0.92f), ImPlotFlags_None))
        {
            ImPlot::SetupAxes("Frequency (Hz)", "Gain (dB)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, Settings::Instance().SampleRate() / 2.f, ImPlotCond_Always);
            // ImPlot::SetupAxisLimits(ImAxis_Y1, -10.0f, 10.0f, ImPlotCond_Once);
            ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

            ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0.f, Settings::Instance().SampleRate() / 2.0f);
            ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -60.f, 60.f);

            static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
            ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

            for (size_t i = 0; i < frequencies.size(); ++i)
            {
                double freq = frequencies[i]; // The frequency should stay constant
                double gain = tc_gains[i];
                point_changed |= ImPlot::DragPoint(i, &freq, &gain, ImVec4(0, 0.9f, 0, 0), 10);
                tc_gains[i] = std::clamp(static_cast<float>(gain), -25.f, 25.0f); // Update the gain value
            }

            if (point_changed)
            {
                config_changed = true;

                std::vector<float> sos =
                    sfFDN::DesignGraphicEQ(tc_gains, frequencies, Settings::Instance().SampleRate());
                frequency_response = utils::AbsFreqz(sos, frequencies_plot, Settings::Instance().SampleRate());

                // To db gain
                for (float& i : frequency_response)
                {
                    i = 20.f * std::log10(i);
                }
            }

            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 7.0f);
            ImPlot::PlotScatter("RT60", frequencies.data(), tc_gains.data(), tc_gains.size());

            if (frequency_response.size() > 0)
            {
                ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.20f, 0.20f, 1.0f), 3.0f);
                ImPlot::PlotLine("Filter Response", frequencies_plot.data(), frequency_response.data(),
                                 frequencies_plot.size());
            }

            if (ImGui::Button("Apply"))
            {
                show_tc_filter_designer = false;
                config_changed = true;
            }
            ImPlot::EndPlot();
        }

        ImGui::End();
    }

    if (config_changed)
    {
        if (enabled)
        {
            config.tc_gains = tc_gains;
            config.tc_frequencies = frequencies;
        }
        else
        {
            config.tc_gains.clear();
            config.tc_frequencies.clear();
        }
    }

    return config_changed;
}

bool DrawEarlyRIRPicker(std::span<const float> impulse_response, std::span<const float> time_data, double& ir_duration)
{
    bool duration_changed = false;

    ImPlot::SetupAxes("Sample", nullptr, ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -1.0f, 1.0f, ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, time_data.back(), ImPlotCond_Always);

    ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -1.0f, 1.0f);
    ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, time_data.back());

    ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f);
    ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.70f, 0.70f, 0.90f, 1.0f));
    ImPlot::PlotLine("IR", time_data.data(), impulse_response.data(), impulse_response.size());
    ImPlot::PopStyleColor();
    ImPlot::PopStyleVar();

    duration_changed = ImPlot::DragLineX(0, &ir_duration, ImVec4(1.f, 1.f, 1.f, 1.f), 1.f, ImPlotDragToolFlags_None);

    // Clamp the duration because if it is dragged too far, it may go out of bounds and get lost forever
    ir_duration = std::clamp(ir_duration, 0.1, static_cast<double>(time_data.back() * 0.95));

    std::array<double, 2> early_xs = {0.f, ir_duration};
    std::array<double, 2> early_ys1 = {-1.f, -1.f};
    std::array<double, 2> early_ys2 = {1.f, 1.f};
    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.250f);
    ImPlot::PlotShaded("Early RIR", early_xs.data(), early_ys1.data(), early_ys2.data(), 2);

    return duration_changed;
}