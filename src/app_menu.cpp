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
    ProcessFileDialogSelections();
}

void FDNToolboxApp::DrawFileMenu()
{
    if (!ImGui::BeginMenu("File"))
    {
        return;
    }

    if (ImGui::MenuItem("Save IR"))
    {
        file_dialogs_.Open(fdn_sandbox::files::FileDialogRequest::SaveImpulseResponse);
    }
    if (ImGui::MenuItem("Load Config"))
    {
        file_dialogs_.Open(fdn_sandbox::files::FileDialogRequest::LoadConfiguration);
    }
    if (ImGui::MenuItem("Save Config"))
    {
        file_dialogs_.Open(fdn_sandbox::files::FileDialogRequest::SaveConfiguration);
    }
    if (ImGui::MenuItem("Load RIR"))
    {
        file_dialogs_.Open(fdn_sandbox::files::FileDialogRequest::LoadRir);
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
        fdn_session_.SaveTo(fdn_sandbox::session::FdnSlot::B);
    }
    if (ImGui::MenuItem("Save current to A"))
    {
        fdn_session_.SaveTo(fdn_sandbox::session::FdnSlot::A);
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
    uint32_t selected_config_slot = fdn_session_.ActiveSlot() == fdn_sandbox::session::FdnSlot::A ? 0U : 1U;
    if (!FancyButtons("A", "B", selected_config_slot))
    {
        return;
    }

    const auto slot = selected_config_slot == 0 ? fdn_sandbox::session::FdnSlot::A : fdn_sandbox::session::FdnSlot::B;
    auto commit = fdn_session_.SwitchTo(slot, Settings::Instance().SampleRateAs<float>());
    if (!commit)
    {
        ReportFdnBuildError(commit.error());
        return;
    }
    PublishFdnCommit(std::move(*commit));
}

void FDNToolboxApp::DrawFrameRateStatus()
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
    }
    ImGui::End();
}

void FDNToolboxApp::ProcessFileDialogSelections()
{
    for (const auto& selection : file_dialogs_.Draw())
    {
        std::visit([this](const auto& value) { HandleFileDialogSelection(value); }, selection);
    }
}

void FDNToolboxApp::HandleFileDialogSelection(const fdn_sandbox::files::SaveImpulseResponseSelection& selection)
{
    const auto result = file_workflows_.SaveImpulseResponse(
        selection.path, analysis_workspace_.Generated().GetImpulseResponse(), Settings::Instance().SampleRate());
    if (result)
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "ir-saved", "Impulse response saved",
                                  selection.path.string());
        return;
    }

    LOG_ERROR(Settings::Instance().GetLogger(), "Failed to save impulse response to {}: {}",
              result.error().path.string(), result.error().message);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "ir-save-failed",
                              "Failed to save impulse response", result.error().message, 6.0);
}

void FDNToolboxApp::HandleFileDialogSelection(const fdn_sandbox::files::LoadConfigurationSelection& selection)
{
    static_cast<void>(selection);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "config-load-not-implemented",
                              "Configuration loading", "Loading configuration files is not implemented yet.");
}

void FDNToolboxApp::HandleFileDialogSelection(const fdn_sandbox::files::SaveConfigurationSelection& selection)
{
    static_cast<void>(selection);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "config-save-not-implemented",
                              "Configuration saving", "Saving configuration files is not implemented yet.");
}

void FDNToolboxApp::HandleFileDialogSelection(const fdn_sandbox::files::LoadRirSelection& selection)
{
    auto loaded = file_workflows_.LoadRir(selection.path, fdn_sandbox::audio::kSystemBlockSize);
    if (!loaded)
    {
        const auto& error = loaded.error();
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to load RIR file {}: {}", error.path.string(),
                  error.message);
        if (error.kind == fdn_sandbox::files::FileErrorKind::UnsupportedChannelCount)
        {
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "rir-invalid-channels", "Invalid RIR",
                                      error.message, 6.0);
        }
        else if (error.kind == fdn_sandbox::files::FileErrorKind::EmptyAudio)
        {
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "rir-empty", "Invalid RIR",
                                      error.message, 6.0);
        }
        else
        {
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "rir-load-failed", "Failed to load RIR",
                                      error.message, 6.0);
        }
        return;
    }
    if (!audio_engine_.TryInstallConvolver(std::move(loaded->convolver)))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Audio engine rejected the loaded RIR");
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "rir-install-failed",
                                  "Failed to activate RIR", "The audio engine rejected the convolver.", 6.0);
        return;
    }

    const std::size_t sample_count = loaded->mono_samples.size();
    const std::string filename = loaded->path.string();
    const std::uint32_t sample_rate = loaded->effective_sample_rate;
    analysis_workspace_.SetReferenceIr(std::move(loaded->mono_samples), fdn_sandbox::analysis::ReferenceIrMetadata{
                                                                            .filename = filename,
                                                                            .analysis_sample_rate = sample_rate,
                                                                        });
    LOG_INFO(Settings::Instance().GetLogger(), "Loaded RIR file: {} ({} Hz, 1 channel, {} samples)", filename,
             sample_rate, sample_count);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "rir-loaded", "RIR loaded",
                              std::format("{} ({} samples)", loaded->path.filename().string(), sample_count));
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
    auto spectrogram_settings = analysis_workspace_.GetSpectrogramSettings();
    static int selected_spectrogram_type = static_cast<int>(spectrogram_settings.scale);
    ImGui::Text("Type:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##Type", &selected_spectrogram_type, kSpectrogramTypes.data(),
                     static_cast<int>(kSpectrogramTypes.size())))
    {
        spectrogram_settings.scale = static_cast<fdn_sandbox::analysis::SpectrogramScale>(selected_spectrogram_type);
        analysis_workspace_.SetSpectrogramSettings(spectrogram_settings);
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
        spectrogram_settings.stft.fft_size = 512U * (1U << selected_fft_size_index);
        if (spectrogram_settings.stft.fft_size < spectrogram_settings.stft.window_size)
        {
            spectrogram_settings.stft.window_size = spectrogram_settings.stft.fft_size;
            spectrogram_settings.stft.overlap =
                static_cast<uint32_t>(static_cast<float>(spectrogram_settings.stft.window_size) * overlap);
            selected_window_size_index = selected_fft_size_index + 1;
        }
        analysis_workspace_.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("Window Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowSize", &selected_window_size_index, kWindowSizeOptions.data(),
                     static_cast<int>(kWindowSizeOptions.size())))
    {
        spectrogram_settings.stft.window_size = 256U * (1U << selected_window_size_index);
        spectrogram_settings.stft.overlap =
            static_cast<uint32_t>(static_cast<float>(spectrogram_settings.stft.window_size) * overlap);
        if (spectrogram_settings.stft.window_size > spectrogram_settings.stft.fft_size)
        {
            spectrogram_settings.stft.fft_size = spectrogram_settings.stft.window_size;
            selected_fft_size_index = selected_window_size_index - 1;
        }
        analysis_workspace_.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("Overlap:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderFloat("##Overlap", &overlap, 0.01f, 0.95f, "%.2f"))
    {
        spectrogram_settings.stft.overlap =
            static_cast<uint32_t>(static_cast<float>(spectrogram_settings.stft.window_size) * overlap);
        analysis_workspace_.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("Window Type:");
    ImGui::SameLine(kColOffset);
    static int selected_window_type = 2; // Default to Hann
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowType", &selected_window_type, "Rectangular\0Hamming\0Hann\0Blackman\0"))
    {
        spectrogram_settings.stft.window_type = static_cast<audio_utils::FFTWindowType>(selected_window_type);
        analysis_workspace_.SetSpectrogramSettings(spectrogram_settings);
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

    if (optimization_gui_.Draw(fdn_session_.Draft(), analysis_workspace_.Reference().GetImpulseResponse()))
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
                   fdn_session_.Draft().feedback_matrix_config);

    const auto& delays = fdn_session_.Draft().delay_bank_config.delays;
    float min_delay = 0.0f;
    float max_delay = 0.0f;
    if (!delays.empty())
    {
        const auto [min_it, max_it] = std::ranges::minmax_element(delays);
        min_delay = *min_it;
        max_delay = *max_it;
    }

    const auto t60_data = analysis_workspace_.Generated().GetT60Data(-5.0f, -50.0f);
    const auto echo_density_data = analysis_workspace_.Generated().GetEchoDensityData(25, 10);
    const auto sample_rate = Settings::Instance().SampleRateAs<float>();

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

        draw_row("FDN Size", std::to_string(fdn_session_.Draft().fdn_size));
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

    const nlohmann::json json_config = fdn_session_.Draft();
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
