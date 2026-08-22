#include "app.h"

#include <audio_utils/audio_analysis.h>

#include "icons.h"
#include "settings.h"
#include "theme.h"
#include "utils.h"

#include <sffdn/sffdn.h>

#include <imgui.h>
#include <implot.h>
#include <nlohmann/json.hpp>
#include <quill/LogMacros.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <vector>

namespace
{
constexpr size_t kSystemBlockSize = 1024; // System block size for audio processing

bool FancyButtons(const char* label1, const char* label2, uint32_t& selected)
{
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    uint32_t new_selected = selected;

    if (selected == 0)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentActive));
    }
    if (ImGui::Button(label1, ImVec2(50, 0)))
    {
        new_selected = 0;
    }

    if (selected == 0)
    {
        ImGui::PopStyleColor(3);
    }

    ImGui::SameLine();
    if (selected == 1)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::Accent));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentHovered));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::AccentActive));
    }

    if (ImGui::Button(label2, ImVec2(50, 0)))
    {
        new_selected = 1;
    }

    if (selected == 1)
    {
        ImGui::PopStyleColor(3);
    }
    ImGui::PopStyleVar();

    const bool changed = (selected != new_selected);
    selected = new_selected;
    return changed;
}
} // namespace

void FDNToolboxApp::DrawMainMenuBar()
{
    static bool show_audio_config_window = false;
    if (ImGui::BeginMainMenuBar())
    {
        fdn_sandbox::theme::DrawWordmark();
        DrawFileMenu();
        DrawOptionsMenu(show_audio_config_window);
        DrawViewMenu();
        ImGui::EndMainMenuBar();
    }

    DrawAudioConfigurationWindow(show_audio_config_window);
    ProcessFileBrowserSelections();
}

void FDNToolboxApp::DrawFileMenu()
{
    if (!ImGui::BeginMenu("File"))
    {
        return;
    }

    if (ImGui::MenuItem("Save IR"))
    {
        save_ir_browser.Open();
    }
    if (ImGui::MenuItem("Load Config"))
    {
        load_config_browser.Open();
    }
    if (ImGui::MenuItem("Save Config"))
    {
        save_config_browser.Open();
    }
    if (ImGui::MenuItem("Load RIR"))
    {
        load_rir_browser.Open();
    }
    ImGui::EndMenu();
}

void FDNToolboxApp::DrawOptionsMenu(bool& show_audio_config_window)
{
    if (!ImGui::BeginMenu("Options"))
    {
        return;
    }

    ImGui::MenuItem(fdn_sandbox::icons::AudioSettings, nullptr, &show_audio_config_window);
    ImGui::Separator();
    if (ImGui::MenuItem("Save current to B"))
    {
        fdn_config_B_ = fdn_config_;
    }
    if (ImGui::MenuItem("Save current to A"))
    {
        fdn_config_A_ = fdn_config_;
    }
    ImGui::EndMenu();
}

void FDNToolboxApp::DrawViewMenu()
{
    if (!ImGui::BeginMenu("View"))
    {
        return;
    }
    if (ImGui::MenuItem("Reset Layout"))
    {
        reset_layout_requested_ = true;
    }
    ImGui::EndMenu();
}

void FDNToolboxApp::DrawConfigurationSwitcher()
{
    if (!FancyButtons("A", "B", selected_config_slot_))
    {
        return;
    }

    if (selected_config_slot_ == 0)
    {
        fdn_config_B_ = fdn_config_;
        fdn_config_ = fdn_config_A_;
    }
    else
    {
        fdn_config_A_ = fdn_config_;
        fdn_config_ = fdn_config_B_;
    }
    UpdateFDN();
}

void FDNToolboxApp::DrawFrameRateStatus() const
{
    const float fps = ImGui::GetIO().Framerate;
    if (fps >= 58.f)
    {
        ImGui::Text("FPS: %.1f", fps);
        return;
    }
    const auto color_role =
        fps >= 50.f ? fdn_sandbox::theme::ColorRole::StatusWarning : fdn_sandbox::theme::ColorRole::StatusError;
    ImGui::TextColored(fdn_sandbox::theme::Color(color_role), "FPS: %.1f", fps);
}

void FDNToolboxApp::DrawAudioConfigurationWindow(bool& show_audio_config_window)
{
    if (!show_audio_config_window)
    {
        return;
    }
    if (ImGui::Begin("Audio", &show_audio_config_window))
    {
        DrawAudioDeviceGUI();
        ImGui::End();
    }
}

void FDNToolboxApp::ProcessFileBrowserSelections()
{
    save_ir_browser.Display();
    load_config_browser.Display();
    save_config_browser.Display();
    load_rir_browser.Display();

    if (save_ir_browser.HasSelected())
    {
        SaveSelectedImpulseResponse();
    }
    if (load_config_browser.HasSelected())
    {
        LoadSelectedConfiguration();
    }
    if (save_config_browser.HasSelected())
    {
        SaveSelectedConfiguration();
    }
    if (load_rir_browser.HasSelected())
    {
        LoadSelectedRIR();
    }
}

void FDNToolboxApp::SaveSelectedImpulseResponse()
{
    std::string filename = save_ir_browser.GetSelected().string();
    if (!filename.ends_with(".wav"))
    {
        filename += ".wav";
    }
    if (utils::WriteAudioFile(filename, fdn_analyzer_.GetImpulseResponse(), Settings::Instance().SampleRate()))
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "ir-saved", "Impulse response saved",
                                  filename);
    }
    else
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "ir-save-failed",
                                  "Failed to save impulse response", filename, 6.0);
    }
    save_ir_browser.ClearSelected();
}

void FDNToolboxApp::LoadSelectedConfiguration()
{
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "config-load-not-implemented",
                              "Configuration loading", "Loading configuration files is not implemented yet.");
    load_config_browser.ClearSelected();
}

void FDNToolboxApp::SaveSelectedConfiguration()
{
    std::string filename = save_config_browser.GetSelected().string();
    if (!filename.ends_with(".json"))
    {
        filename += ".json";
    }
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "config-save-not-implemented",
                              "Configuration saving", "Saving configuration files is not implemented yet.");
    save_config_browser.ClearSelected();
}

void FDNToolboxApp::LoadSelectedRIR()
{
    const std::string filename = load_rir_browser.GetSelected().string();
    std::vector<float> buffer;
    int file_sample_rate = Settings::Instance().SampleRate();
    int file_num_channels = 0;
    if (!audio_utils::audio_file::ReadWavFile(filename, buffer, file_sample_rate, file_num_channels))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to load RIR file: {}", filename);
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "rir-load-failed", "Failed to load RIR",
                                  filename, 6.0);
        load_rir_browser.ClearSelected();
        return;
    }
    if (file_num_channels != 1)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "RIR file must be mono. Loaded file has {} channels.",
                  file_num_channels);
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "rir-invalid-channels", "Invalid RIR",
                                  std::format("Expected mono audio, found {} channels.", file_num_channels), 6.0);
        load_rir_browser.ClearSelected();
        return;
    }

    LOG_INFO(Settings::Instance().GetLogger(), "Loaded RIR file: {} ({} Hz, {} channels, {} samples)", filename,
             file_sample_rate, file_num_channels, buffer.size());
    loaded_rir_filename_ = filename;
    auto convolution_reverb = std::make_unique<sfFDN::PartitionedConvolver>(kSystemBlockSize, buffer);
    LOG_INFO(Settings::Instance().GetLogger(), "Created PartitionedConvolver with {}",
             convolution_reverb->GetShortInfo());
    rir_analyzer_.SetImpulseResponse(std::move(buffer));
    convolution_reverb_ = std::move(convolution_reverb);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "rir-loaded", "RIR loaded",
                              std::format("{} ({} samples)", std::filesystem::path(filename).filename().string(),
                                          rir_analyzer_.GetImpulseResponseSize()));
    load_rir_browser.ClearSelected();
}
void FDNToolboxApp::DrawSettingsWindow()
{
    if (!ImGui::Begin("Settings"))
    {
        ImGui::End();
        return;
    }

    ImGui::Text("Sample Rate: %u Hz", Settings::Instance().SampleRate());
    ImGui::Separator();

    ImGui::Text("Spectrogram Settings");
    ImGui::Separator();

    constexpr int kColOffset = 100;

    constexpr std::array kSpectrogramTypes = {"STFT", "Mel"};
    static int selected_spectrogram_type = static_cast<int>(spectrogram_type_);
    ImGui::Text("Type:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##Type", &selected_spectrogram_type, kSpectrogramTypes.data(),
                     static_cast<int>(kSpectrogramTypes.size())))
    {
        spectrogram_type_ = static_cast<SpectrogramType>(selected_spectrogram_type);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
        rir_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }
    constexpr std::array kFFTSizeOptions = {"512", "1024", "2048", "4096", "8192"};
    constexpr std::array kWindowSizeOptions = {"256", "512", "1024", "2048", "4096", "8192"};
    static int selected_fft_size_index = 1;    // Default to 1024
    static int selected_window_size_index = 2; // Default to 512
    static float overlap = 0.5f;

    ImGui::Text("FFT Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##FFTSize", &selected_fft_size_index, kFFTSizeOptions.data(),
                     static_cast<int>(kFFTSizeOptions.size())))
    {
        stft_options_.fft_size = 512 * (1 << selected_fft_size_index);
        if (stft_options_.fft_size < stft_options_.window_size)
        {
            stft_options_.window_size = stft_options_.fft_size;
            stft_options_.overlap = static_cast<uint32_t>(overlap * stft_options_.window_size);
            selected_window_size_index = selected_fft_size_index + 1;
        }

        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Window Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowSize", &selected_window_size_index, kWindowSizeOptions.data(),
                     static_cast<int>(kWindowSizeOptions.size())))
    {
        stft_options_.window_size = 256 * (1 << selected_window_size_index);
        stft_options_.overlap = static_cast<uint32_t>(overlap * stft_options_.window_size);
        if (stft_options_.window_size > stft_options_.fft_size)
        {
            stft_options_.fft_size = stft_options_.window_size;
            selected_fft_size_index = selected_window_size_index - 1;
        }
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Overlap:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderFloat("##Overlap", &overlap, 0.01f, 0.95f, "%.2f"))
    {
        stft_options_.overlap = static_cast<uint32_t>(overlap * stft_options_.window_size);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::Text("Window Type:");
    ImGui::SameLine(kColOffset);
    static int selected_window_type = 2; // Default to Hann
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowType", &selected_window_type, "Rectangular\0Hamming\0Hann\0Blackman\0"))
    {
        stft_options_.window_type = static_cast<audio_utils::FFTWindowType>(selected_window_type);
        fdn_analyzer_.RequestAnalysis(fdn_analysis::AnalysisType::Spectrogram);
    }

    ImGui::SeparatorText("Style Settings");

    // ImPlotStyle& style = ImPlot::GetStyle();

    // float line_weight = style.LineWeight;
    // ImGui::SetNextItemWidth(200);
    // if (ImGui::SliderFloat("Line Weight", &line_weight, 1.0f, 5.0f))
    // {
    //     style.LineWeight = line_weight;
    // }

    ImGui::SetNextItemWidth(200);
    ImGui::ShowFontSelector("ImGui Font");

    ImGui::SetNextItemWidth(200);
    ImGui::ShowStyleSelector("ImGui Style");

    ImGui::SetNextItemWidth(200);
    ImPlot::ShowStyleSelector("ImPlot Style");

    ImGui::SetNextItemWidth(200);
    ImPlot::ShowColormapSelector("ImPlot Colormap");

    ImGui::End(); // End the Settings window
}
void FDNToolboxApp::DrawOptimizationWindow()
{
    if (!ImGui::Begin("Optimization"))
    {
        ImGui::End();
        return;
    }

    if (optimization_gui_.Draw(fdn_config_, rir_analyzer_.GetImpulseResponse()))
    {
        UpdateFDN();
    }

    ImGui::End();
}
void FDNToolboxApp::DrawFDNInfoWindow()
{
    if (!ImGui::Begin("FDN Info"))
    {
        ImGui::End();
        return;
    }

    const std::string matrix_name =
        std::visit([](const auto& matrix_options) { return utils::GetMatrixName(matrix_options.type); },
                   fdn_config_.feedback_matrix_config);

    const auto& delays = fdn_config_.delay_bank_config.delays;
    float min_delay = 0.0f;
    float max_delay = 0.0f;
    if (!delays.empty())
    {
        const auto [min_it, max_it] = std::ranges::minmax_element(delays);
        min_delay = *min_it;
        max_delay = *max_it;
    }

    const auto t60_data = fdn_analyzer_.GetT60Data(-5.0f, -50.0f);
    const auto echo_density_data = fdn_analyzer_.GetEchoDensityData(25, 10);
    const float sample_rate = static_cast<float>(Settings::Instance().SampleRate());

    if (ImGui::BeginTable("##FDNSummary", 2,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp))
    {
        const auto draw_row = [](const char* label, const std::string& value) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("%s", label);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(value.c_str());
        };

        draw_row("FDN Size", std::to_string(fdn_config_.fdn_size));
        draw_row("Feedback Matrix", matrix_name);
        draw_row("Delay Count", std::to_string(delays.size()));
        draw_row("Delay Range (samples)", delays.empty() ? "--" : std::format("{:.0f} - {:.0f}", min_delay, max_delay));
        draw_row("Delay Range (ms)", delays.empty() ? "--"
                                                    : std::format("{:.2f} - {:.2f}", min_delay / sample_rate * 1000.0f,
                                                                  max_delay / sample_rate * 1000.0f));
        draw_row("Wideband T60", std::isfinite(t60_data.overall_t60.t60) && t60_data.overall_t60.t60 > 0.0f
                                     ? std::format("{:.2f} s", t60_data.overall_t60.t60)
                                     : "--");
        draw_row("Mixing Time", std::isfinite(echo_density_data.mixing_time) && echo_density_data.mixing_time > 0.0f
                                    ? std::format("{:.1f} ms", echo_density_data.mixing_time * 1000.0f)
                                    : "--");
        draw_row("Sample Rate", std::format("{} Hz", Settings::Instance().SampleRate()));
        draw_row("CPU", std::format("{:.1f}%", displayed_cpu_usage_ * 100.0f));
        ImGui::EndTable();
    }

    const nlohmann::json json_config = fdn_config_;
    std::string json_str = json_config.dump(4); // Pretty print with 4
    if (ImGui::CollapsingHeader("Raw Configuration (JSON)"))
    {
        if (ImGui::Button(fdn_sandbox::icons::CopyJson))
        {
            ImGui::SetClipboardText(json_str.c_str());
        }
        ImGui::InputTextMultiline("##FDNConfig", json_str.data(), json_str.size() + 1, ImVec2(-1, -1),
                                  ImGuiInputTextFlags_ReadOnly);
    }

    ImGui::End();
}
