#include "views/settings_view.h"

#include "analysis/analysis_workspace.h"
#include "audio_utils/fft_utils.h"
#include "settings.h"
#include "views/window_ids.h"

#include <audio_utils/audio_analysis.h>

#include <imgui.h>
#include <implot.h>

#include <array>
#include <cstdint>

namespace fdn_sandbox::views
{

void SettingsView::Draw(analysis::AnalysisWorkspace& workspace)
{
    if (!ImGui::Begin(ids::kSettings))
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
    constexpr std::array kFFTSizeOptions = {"512", "1024", "2048", "4096", "8192"};
    constexpr std::array kWindowSizeOptions = {"256", "512", "1024", "2048", "4096", "8192"};

    auto spectrogram_settings = workspace.GetSpectrogramSettings();
    if (!initialized_)
    {
        selected_spectrogram_type_ = static_cast<int>(spectrogram_settings.scale);
        initialized_ = true;
    }

    ImGui::Text("Type:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##Type", &selected_spectrogram_type_, kSpectrogramTypes.data(),
                     static_cast<int>(kSpectrogramTypes.size())))
    {
        spectrogram_settings.scale = static_cast<analysis::SpectrogramScale>(selected_spectrogram_type_);
        workspace.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("FFT Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##FFTSize", &selected_fft_size_index_, kFFTSizeOptions.data(),
                     static_cast<int>(kFFTSizeOptions.size())))
    {
        spectrogram_settings.stft.fft_size = 512U * (1U << selected_fft_size_index_);
        if (spectrogram_settings.stft.fft_size < spectrogram_settings.stft.window_size)
        {
            spectrogram_settings.stft.window_size = spectrogram_settings.stft.fft_size;
            spectrogram_settings.stft.overlap =
                static_cast<std::uint32_t>(static_cast<float>(spectrogram_settings.stft.window_size) * overlap_);
            selected_window_size_index_ = selected_fft_size_index_ + 1;
        }
        workspace.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("Window Size:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowSize", &selected_window_size_index_, kWindowSizeOptions.data(),
                     static_cast<int>(kWindowSizeOptions.size())))
    {
        spectrogram_settings.stft.window_size = 256U * (1U << selected_window_size_index_);
        spectrogram_settings.stft.overlap =
            static_cast<std::uint32_t>(static_cast<float>(spectrogram_settings.stft.window_size) * overlap_);
        if (spectrogram_settings.stft.window_size > spectrogram_settings.stft.fft_size)
        {
            spectrogram_settings.stft.fft_size = spectrogram_settings.stft.window_size;
            selected_fft_size_index_ = selected_window_size_index_ - 1;
        }
        workspace.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("Overlap:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderFloat("##Overlap", &overlap_, 0.01f, 0.95f, "%.2f"))
    {
        spectrogram_settings.stft.overlap =
            static_cast<std::uint32_t>(static_cast<float>(spectrogram_settings.stft.window_size) * overlap_);
        workspace.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::Text("Window Type:");
    ImGui::SameLine(kColOffset);
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("##WindowType", &selected_window_type_, "Rectangular\0Hamming\0Hann\0Blackman\0"))
    {
        spectrogram_settings.stft.window_type = static_cast<audio_utils::FFTWindowType>(selected_window_type_);
        workspace.SetSpectrogramSettings(spectrogram_settings);
    }

    ImGui::SeparatorText("Style Settings");
    ImGui::SetNextItemWidth(200);
    ImGui::ShowFontSelector("ImGui Font");
    ImGui::SetNextItemWidth(200);
    ImGui::ShowStyleSelector("ImGui Style");
    ImGui::SetNextItemWidth(200);
    ImPlot::ShowStyleSelector("ImPlot Style");
    ImGui::SetNextItemWidth(200);
    ImPlot::ShowColormapSelector("ImPlot Colormap");
    ImGui::End();
}

} // namespace fdn_sandbox::views
