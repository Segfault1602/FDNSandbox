#include "app.h"

#include "settings.h"
#include <quill/LogMacros.h>

#include <cstdint>
#include <format>
#include <memory>
#include <string>

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
