#include "widget.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>

#include <implot.h>

#include <sffdn/sffdn.h>

#include "app.h"
#include "fdn_info.h"
#include "utils.h"

namespace
{

constexpr size_t kSampleRate = 48000; // Define a constant sample rate for audio processing
bool DrawFilterDesigner(FDNConfig& fdn_config, bool& show_delay_filter_designer)
{
    if (!ImGui::Begin("Filter Designer", &show_delay_filter_designer))
    {
        ImGui::End();
        return false;
    }

    bool config_changed = false;

    constexpr uint32_t kTestDelay = 593.f; // arbitrary, needed to design the filter for the GUI

    constexpr size_t kNBands = 10; // Number of bands in the filter designer
    static std::vector<float> t60s(kNBands, 2.f);
    static std::vector<float> frequencies(0, 0.f);
    std::vector<float> gains(kNBands, 0.0f);

    // Oversampled vectors for plotting
    static std::vector<float> gains_plot;
    static std::vector<float> t60s_plot;
    static std::vector<float> frequencies_plot;
    static std::vector<float> filter_freqs_plot;

    bool point_changed = false;

    // Initialize with default values if empty
    if (fdn_config.t60s.size() == 0)
    {
        fdn_config.t60s = t60s;
    }
    // else // Use the existing t60s from the config
    // {
    //     t60s = fdn_config.t60s;
    // }

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
        frequencies_plot =
            utils::LogSpace(std::log10(frequencies[0] + 1e-6f), std::log10(frequencies.back() - 1.f), 256);
        filter_freqs_plot = utils::LogSpace(std::log10(1.f), std::log10(kSampleRate / 2.f), 512);
        t60s_plot = utils::pchip(frequencies, t60s, frequencies_plot);

        gains = utils::T60ToGainsDb(t60s, kTestDelay, kSampleRate);
        gains_plot = utils::pchip(frequencies, gains, frequencies_plot);
        point_changed = true; // Force initial plot update
    }

    if (ImPlot::BeginPlot("Filter Designer", ImVec2(-1, ImGui::GetWindowHeight() * 0.45f), ImPlotFlags_NoLegend))
    {
        ImPlot::SetupAxes("Frequency (Hz)", "RT60 (s)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.01f, 5.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, kSampleRate / 2);
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
        gains = utils::T60ToGainsDb(t60s, kTestDelay, kSampleRate);
        gains_plot = utils::pchip(frequencies, gains, frequencies_plot);
        std::vector<float> t60s_f(t60s.begin(), t60s.end());
        std::vector<float> sos = sfFDN::GetTwoFilter(t60s_f, kTestDelay, kSampleRate, shelf_cutoff);

        H = utils::AbsFreqz(sos, filter_freqs_plot, kSampleRate);

        // To db gain
        for (size_t i = 0; i < H.size(); ++i)
        {
            H[i] = 20.f * std::log10(H[i]);
        }
    }

    if (ImPlot::BeginPlot("Filter preview", ImVec2(-1, ImGui::GetWindowHeight() * 0.45f), ImPlotFlags_None))
    {

        ImPlot::SetupAxes("Frequency (Hz)", "Gain (dB)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 20.0f, 20000.0f, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -10.0f, 1.0f, ImPlotCond_Once);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);

        ImPlot::SetupAxisLimitsConstraints(ImAxis_X1, 0, kSampleRate / 2);
        ImPlot::SetupAxisLimitsConstraints(ImAxis_Y1, -60.f, 10.f);

        static std::vector<double> frequencies_d(frequencies.begin(), frequencies.end());
        ImPlot::SetupAxisTicks(ImAxis_X1, frequencies_d.data(), frequencies_d.size(), nullptr, false);

        ImPlot::SetNextLineStyle(ImVec4(0.70f, 0.70f, 0.20f, 1.0f), 4.0f);
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
        fdn_config.delay_filter_type = DelayFilterType::TwoFilter;
        fdn_config.t60s = t60s;
        config_changed = true;
    }

    if (ImGui::Button("Apply"))
    {
        std::cout << "Applying filter design..." << std::endl;
        fdn_config.delay_filter_type = DelayFilterType::TwoFilter;
        fdn_config.t60s = t60s;
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

void DrawInputOutputGainsPlot(const sfFDN::FDN* fdn)
{
    if (ImPlot::BeginSubplots("##Input/Output_Gains", 2, 1, ImVec2(300, 200),
                              ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
        static std::vector<float> input_gains;
        static std::vector<float> output_gains;

        fdn_info::GetInputAndOutputGains(fdn, input_gains, output_gains);

        constexpr ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        if (ImPlot::BeginPlot("Input Gains", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);
            ImPlot::SetupAxesLimits(-0.45, input_gains.size() - 0.45, -1, 1, ImPlotCond_Always);
            ImPlot::PlotBars("Input Gains", input_gains.data(), input_gains.size(), 0.90, 0, ImPlotBarsFlags_None);
            ImPlot::EndPlot();
        }

        if (ImPlot::BeginPlot("Output Gains", ImVec2(-1, -1), ImPlotFlags_NoLegend))
        {
            ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);
            ImPlot::SetupAxesLimits(-0.45, output_gains.size() - 0.45, -1, 1, ImPlotCond_Always);
            ImPlot::PlotBars("Output Gains", output_gains.data(), output_gains.size(), 0.90, 0, ImPlotBarsFlags_None);
            ImPlot::EndPlot();
        }

        ImPlot::EndSubplots();
    }
}

void DrawDelaysPlot(const sfFDN::FDN* fdn, uint32_t max_delay)
{
    if (ImPlot::BeginPlot("Delays", ImVec2(300, 100), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText))
    {
        static std::vector<uint32_t> delays;
        fdn_info::GetDelays(fdn, delays);

        constexpr ImPlotAxisFlags axes_flags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines;
        ImPlot::SetupAxes(nullptr, nullptr, axes_flags | ImPlotAxisFlags_NoTickLabels, axes_flags);

        ImPlot::SetupAxesLimits(-1, delays.size(), 0, max_delay, ImPlotCond_Always);

        ImPlot::PlotBars("##Delays", delays.data(), delays.size(), 0.90, 0, ImPlotBarsFlags_None);
        ImPlot::EndPlot();
    }
}

void DrawFeedbackMatrixPlot(const sfFDN::FDN* fdn)
{
    constexpr ImPlotColormap feedback_matrix_colormap = ImPlotColormap_Plasma;

    static std::vector<float> feedback_matrix;
    static uint32_t N = 0;
    if (fdn_info::GetFeedbackMatrix(fdn, feedback_matrix, N))
    {
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
    else
    {
        ImGui::Text("Unable to display feedback matrix");
    }
}

bool DrawInputGainsWidget(sfFDN::FDN* fdn, std::span<float> gains)
{
    static std::vector<float> input_gains;
    static std::vector<float> output_gains;
    fdn_info::GetInputAndOutputGains(fdn, input_gains, output_gains);

    bool config_changed = DrawGainsWidget(input_gains);

    if (config_changed)
    {
        fdn->SetInputGains(input_gains);
    }

    return config_changed;
}

bool DrawOutputGainsWidget(sfFDN::FDN* fdn, std::span<float> gains)
{
    static std::vector<float> input_gains;
    static std::vector<float> output_gains;
    fdn_info::GetInputAndOutputGains(fdn, input_gains, output_gains);

    bool config_changed = DrawGainsWidget(output_gains);
    if (config_changed)
    {
        fdn->SetOutputGains(output_gains);
    }

    return config_changed;
}

bool DrawDelayLengthsWidget(size_t N, std::span<uint32_t> delays, int& min_delay, int& max_delay, uint32_t random_seed,
                            bool refresh)
{
    bool config_changed = false;
    bool should_update_delays = refresh;
    static sfFDN::DelayLengthType selected_delay_length_type = sfFDN::DelayLengthType::Random;
    static int selected_sort_type = 0;
    bool sort_type_changed = false;

    if (!ImGui::TreeNode("Edit delays"))
    {
        return false;
    }

    constexpr int kMinDelay = 100; // Minimum delay in samples
    if (ImGui::DragIntRange2("Delay Range", &min_delay, &max_delay, 1, kMinDelay, 0, "%d samples", "%d samples",
                             ImGuiSliderFlags_AlwaysClamp))
    {
        should_update_delays = true;
    }

    // Need to manually clamp because DragIntRange2 doesn't seem to respect the min/max values
    min_delay = std::clamp(min_delay, kMinDelay, max_delay);
    max_delay = std::max(max_delay, kMinDelay);

    if (ImGui::BeginCombo("Delay Length Type", utils::GetDelayLengthTypeName(selected_delay_length_type).c_str()))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::DelayLengthType::Count); i++)
        {
            bool is_selected = (selected_delay_length_type == static_cast<sfFDN::DelayLengthType>(i));
            if (ImGui::Selectable(utils::GetDelayLengthTypeName(static_cast<sfFDN::DelayLengthType>(i)).c_str(),
                                  is_selected))
            {
                selected_delay_length_type = static_cast<sfFDN::DelayLengthType>(i);
                should_update_delays = true;
            }
        }
        ImGui::EndCombo();
    }

    static bool clamp_to_prime = false;
    config_changed |= ImGui::Checkbox("Clamp to Prime", &clamp_to_prime);

    constexpr const char* sort_type[] = {"None", "Ascending", "Descending"};
    if (ImGui::BeginCombo("Sort Type", sort_type[selected_sort_type]))
    {
        for (int i = 0; i < IM_ARRAYSIZE(sort_type); ++i)
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
        int delay = static_cast<int>(delays[i]);
        config_changed |= ImGui::SliderInt(label.c_str(), &delay, kMinDelay, max_delay, nullptr);

        delays[i] = std::max<size_t>(static_cast<size_t>(delay), kMinDelay);

        if (clamp_to_prime)
        {
            delays[i] = utils::GetClosestPrime(delays[i]);
        }
    }

    if (should_update_delays)
    {
        config_changed = true;
        auto delay_vec = sfFDN::GetDelayLengths(N, min_delay, max_delay, selected_delay_length_type, random_seed);
        assert(delays.size() == delay_vec.size());
        for (size_t i = 0; i < N; ++i)
        {
            delays[i] = delay_vec[i];
        }
        sort_type_changed = true; // Force sort if we updated delays
    }

    if (sort_type_changed)
    {
        config_changed = true;
        if (selected_sort_type == 1) // Ascending
        {
            std::ranges::sort(delays);
        }
        else if (selected_sort_type == 2) // Descending
        {
            std::ranges::sort(delays, std::greater<uint32_t>());
        }
    }

    ImGui::TreePop();
    return config_changed;
}

bool DrawFeedbackMatrixWidget(FDNConfig& fdn_config, uint32_t random_seed, bool refresh)
{
    return DrawScalarMatrixWidget(fdn_config, random_seed, refresh);
}

bool DrawScalarMatrixWidget(FDNConfig& fdn_config, uint32_t random_seed, bool refresh)
{
    bool config_changed = false;
    bool should_update_feedback_matrix = refresh;
    static bool cascade_matrix = false;
    static float sparsity = 1.f;
    static int num_stages = 2;

    static int selected_matrix_type = 0;
    if (!ImGui::TreeNode("Edit matrix"))
    {
        return false;
    }

    const std::string combo_preview_value =
        utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type));
    if (ImGui::BeginCombo("Matrix Type", combo_preview_value.c_str()))
    {
        for (int i = 0; i < static_cast<int>(sfFDN::ScalarMatrixType::Count); i++)
        {
            bool is_selected = (selected_matrix_type == i);

            ImGuiSelectableFlags flags = ImGuiSelectableFlags_None;

            if (!utils::IsPowerOfTwo(fdn_config.N) &&
                static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard)
            {
                flags |= ImGuiSelectableFlags_Disabled;
            }

            if (ImGui::Selectable(utils::GetMatrixName(static_cast<sfFDN::ScalarMatrixType>(i)).c_str(), is_selected,
                                  flags))
            {
                selected_matrix_type = i;
                should_update_feedback_matrix = true;
            }

            if (!utils::IsPowerOfTwo(fdn_config.N) &&
                static_cast<sfFDN::ScalarMatrixType>(i) == sfFDN::ScalarMatrixType::Hadamard)
            {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), " only supported for N that is a power of 2.");
            }
        }
        ImGui::EndCombo();
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
            fdn_config.feedback_matrix.assign(fdn_config.N * fdn_config.N, 0.0f);
            config_changed = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("Read from clipboard"))
        {
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

            size_t line_count = std::min(lines.size(), static_cast<size_t>(fdn_config.N));

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
                for (size_t j = 0; j < fdn_config.N && j < values.size(); ++j)
                {
                    fdn_config.feedback_matrix[i * fdn_config.N + j] = values[j];
                }
            }
        }

        ImGui::PushItemWidth(50);
        for (size_t i = 0; i < fdn_config.N; ++i)
        {
            for (size_t j = 0; j < fdn_config.N; ++j)
            {
                if (j != 0)
                {
                    ImGui::SameLine(); // Align inputs in a grid
                }
                ImGui::PushID((i * fdn_config.N) + j);
                config_changed |= ImGui::InputScalar("##MatrixValue", ImGuiDataType_Float,
                                                     &fdn_config.feedback_matrix[(i * fdn_config.N) + j], nullptr,
                                                     nullptr, "%.3f", 0);
                ImGui::PopID();
            }
        }

        ImGui::PopItemWidth();

        if (ImGui::Button("Set"))
        {
            ImGui::CloseCurrentPopup();
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
            PlotCascadedFeedbackMatrix(fdn_config.cascaded_feedback_matrix_info);

            if (ImGui::Button("OK", ImVec2(120, 0)))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    if (should_update_feedback_matrix)
    {
        auto matrix_type = static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type);
        if (matrix_type == sfFDN::ScalarMatrixType::NestedAllpass)
        {
            fdn_config.feedback_matrix =
                sfFDN::NestedAllpassMatrix(fdn_config.N, random_seed, fdn_config.input_gains, fdn_config.output_gains);
        }
        else
        {
            fdn_config.feedback_matrix = sfFDN::GenerateMatrix(fdn_config.N, matrix_type, random_seed);
        }

        if (cascade_matrix)
        {
            fdn_config.cascaded_feedback_matrix_info = sfFDN::ConstructCascadedFeedbackMatrix(
                fdn_config.N, num_stages, sparsity, static_cast<sfFDN::ScalarMatrixType>(selected_matrix_type), 1.f);
        }

        fdn_config.is_cascaded = cascade_matrix;
        fdn_config.sparsity = sparsity;
        fdn_config.num_stages = num_stages;
        fdn_config.cascade_gain = 1.f;
        config_changed = true;
    }

    ImGui::TreePop();
    return config_changed;
}

bool DrawDelayFilterWidget(FDNConfig& fdn_config)
{
    bool config_changed = false;

    static bool show_delay_filter_designer = false;

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
                selected_filter_type = i;
                config_changed = true;
            }
        }
        ImGui::EndCombo();
    }

    // Proportinal feedback Gain
    if (selected_filter_type == 0) // Proportional
    {
        const float kFbGainStep = 0.00001f;
        const float kFbGainStepFast = 0.0001f;
        config_changed |= ImGui::InputScalar("Feedback Gain", ImGuiDataType_Float, &fdn_config.feedback_gain,
                                             &kFbGainStep, &kFbGainStepFast, "%.5f", 0);
        fdn_config.feedback_gain =
            std::clamp(fdn_config.feedback_gain, -1.0f, 1.0f); // Ensure feedback gain is within [0, 1]
        fdn_config.delay_filter_type = DelayFilterType::Proportional;
    }
    else if (selected_filter_type == 1) // One Pole
    {
        constexpr float kOffsetFromStart = 125.f;
        ImGui::Text("RT60 DC: ");
        ImGui::SameLine(kOffsetFromStart);
        ImGui::SetNextItemWidth(200);
        if (ImGui::InputFloat("RT60 DC", &fdn_config.t60_dc, 0.01f, 0.1f, "%.2f"))
        {
            config_changed = true;
        }

        fdn_config.t60_dc = std::clamp(fdn_config.t60_dc, 0.01f, 10.0f);

        ImGui::Text("RT60 Nyquist: ");
        ImGui::SameLine(kOffsetFromStart);
        ImGui::SetNextItemWidth(200);
        if (ImGui::InputFloat("RT60 Nyquist", &fdn_config.t60_ny, 0.01f, 0.1f, "%.2f"))
        {
            config_changed = true;
        }

        fdn_config.t60_ny = std::clamp(fdn_config.t60_ny, 0.01f, 10.0f);
        fdn_config.delay_filter_type = DelayFilterType::OnePole;
    }
    else if (selected_filter_type == 2) // TwoFilter
    {
        if (ImGui::Button("Edit"))
        {
            std::cout << "Opening filter designer...\n";
            show_delay_filter_designer = true;
            config_changed = true;
        }
        fdn_config.delay_filter_type = DelayFilterType::TwoFilter;
    }

    if (show_delay_filter_designer)
    {
        bool filter_changed = DrawFilterDesigner(fdn_config, show_delay_filter_designer);
        config_changed |= filter_changed;
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

    std::array<double, 2> early_xs = {0.f, ir_duration};
    std::array<double, 2> early_ys1 = {-1.f, -1.f};
    std::array<double, 2> early_ys2 = {1.f, 1.f};
    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.250f);
    ImPlot::PlotShaded("Early RIR", early_xs.data(), early_ys1.data(), early_ys2.data(), 2);

    return duration_changed;
}