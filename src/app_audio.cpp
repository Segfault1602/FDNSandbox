#include "app.h"

#include "audio/audio_clip.h"
#include "audio/audio_types.h"
#include "imgui.h"
#include "notifications.h"
#include "settings.h"

#include <boost/dll/runtime_symbol_info.hpp>
#include <quill/LogMacros.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

namespace
{
constexpr std::array<const char*, 4> kAudioFiles = {"drumloop.wav", "guitar.wav", "bleepsandbloops.wav",
                                                    "saxophone.wav"};
}

void FDNToolboxApp::PlaySelectedAudioFile(int selected_audio_file)
{
    const auto audio_path = boost::dll::program_location().parent_path() / "audio" /
                            kAudioFiles.at(static_cast<std::size_t>(selected_audio_file));
    LOG_INFO(Settings::Instance().GetLogger(), "Playing audio file: {}", audio_path.string());
    auto clip = file_workflows_.LoadClip(std::filesystem::path(audio_path.string()), Settings::Instance().SampleRate());
    if (clip && audio_engine_.TryInstallClip(std::make_unique<fdn_sandbox::audio::AudioClip>(std::move(*clip))) &&
        audio_engine_.PlayClip(true))
    {
        return;
    }
    const std::string error = clip ? "Failed to submit the decoded clip." : clip.error().message;
    LOG_ERROR(Settings::Instance().GetLogger(), "Failed to play audio file: {}", error);
    notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "audio-clip-load-failed",
                              "Failed to play audio clip", error, 6.0);
}

void FDNToolboxApp::UpdateUiTelemetry()
{
    const fdn_sandbox::audio::AudioTelemetry telemetry = audio_engine_.ReadTelemetry();
    fdn_session_.ObserveActiveGeneration(telemetry.active_fdn_generation);
    audio_view_.UpdateTelemetry(telemetry, audio_engine_.ConsumePeak(), audio_engine_.ConsumeClipped(),
                                ImGui::GetIO().DeltaTime);
}
