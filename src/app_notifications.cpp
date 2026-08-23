#include "app.h"

#include "settings.h"

#include <quill/LogMacros.h>

#include <format>
#include <utility>

void FDNToolboxApp::ProcessNotifications()
{
    if (audio_view_.IsClippingWarningDisplayed() && !clipping_notification_active_)
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "audio-clipping", "Audio clipping",
                                  "The output exceeded 0 dBFS.");
        clipping_notification_active_ = true;
    }
    else if (!audio_view_.IsClippingWarningDisplayed())
    {
        clipping_notification_active_ = false;
    }

    for (const fdn_sandbox::audio::AudioEvent& audio_event : audio_engine_.PollEvents())
    {
        switch (audio_event.kind)
        {
        case fdn_sandbox::audio::AudioEventKind::FrameSizeMismatch:
            LOG_ERROR(Settings::Instance().GetLogger(), "Frame size mismatch: expected {}, got {}",
                      audio_event.expected, audio_event.actual);
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "audio-frame-size-mismatch",
                                      "Unsupported audio block size",
                                      std::format("Expected {} frames, but the audio device supplied {}.",
                                                  audio_event.expected, audio_event.actual),
                                      6.0);
            break;
        case fdn_sandbox::audio::AudioEventKind::FdnInstability:
            if (fdn_session_.ShouldReportInstability(audio_event.generation))
            {
                notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "fdn-instability",
                                          "FDN instability",
                                          "The FDN output exceeded the safety threshold and was cleared.", 6.0);
                notification_center_.SetCritical(fdn_sandbox::NotificationSeverity::Error, "fdn-instability",
                                                 "FDN instability",
                                                 "The network was cleared. Update the FDN configuration to dismiss.");
            }
            break;
        case fdn_sandbox::audio::AudioEventKind::ConvolverBlockSizeMismatch:
            LOG_ERROR(Settings::Instance().GetLogger(),
                      "RIR PartitionedConvolver block size {} does not match system block size {}", audio_event.actual,
                      audio_event.expected);
            break;
        case fdn_sandbox::audio::AudioEventKind::DirectDelayOutOfRange:
            LOG_ERROR(Settings::Instance().GetLogger(), "Pre-delay samples {} exceed maximum delay {}",
                      audio_event.actual, audio_event.expected);
            break;
        case fdn_sandbox::audio::AudioEventKind::CpuHighWater:
            LOG_WARNING(Settings::Instance().GetLogger(), "New CPU highwater mark: {:.2f}%",
                        audio_event.value * 100.0f);
            break;
        case fdn_sandbox::audio::AudioEventKind::ClipFinished:
            break;
        }
    }

    const fdn_sandbox::audio::AudioTelemetry telemetry = audio_engine_.ReadTelemetry();
    if (telemetry.event_overflow_count > reported_event_overflow_count_)
    {
        reported_event_overflow_count_ = telemetry.event_overflow_count;
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "audio-event-overflow",
                                  "Audio diagnostics dropped",
                                  "The audio event queue overflowed before the UI could drain it.");
    }

    if (auto event = optimization_view_.ConsumeEvent())
    {
        switch (event->type)
        {
        case fdn_sandbox::views::OptimizationView::EventType::Completed:
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "optimization-completed",
                                      "Optimization completed", std::move(event->message), 6.0);
            break;
        case fdn_sandbox::views::OptimizationView::EventType::Canceled:
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "optimization-canceled",
                                      "Optimization canceled", std::move(event->message));
            break;
        case fdn_sandbox::views::OptimizationView::EventType::Failed:
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "optimization-failed",
                                      "Optimization failed", std::move(event->message), 6.0);
            break;
        }
    }
}
