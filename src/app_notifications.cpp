#include "app.h"

#include <utility>

void FDNToolboxApp::ProcessNotifications()
{
    if (output_meter_display_.clipping_warning_displayed && !clipping_notification_active_)
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "audio-clipping", "Audio clipping",
                                  "The output exceeded 0 dBFS.");
        clipping_notification_active_ = true;
    }
    else if (!output_meter_display_.clipping_warning_displayed)
    {
        clipping_notification_active_ = false;
    }

    const uint64_t instability_generation = fdn_instability_generation_.exchange(0, std::memory_order_acquire);
    if (instability_generation >= submitted_fdn_generation_ && instability_generation != 0)
    {
        notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "fdn-instability", "FDN instability",
                                  "The FDN output exceeded the safety threshold and was cleared.", 6.0);
        notification_center_.SetCritical(fdn_sandbox::NotificationSeverity::Error, "fdn-instability", "FDN instability",
                                         "The network was cleared. Update the FDN configuration to dismiss.");
    }

    if (auto event = optimization_gui_.ConsumeEvent())
    {
        switch (event->type)
        {
        case OptimizationGUI::EventType::Completed:
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Success, "optimization-completed",
                                      "Optimization completed", std::move(event->message), 6.0);
            break;
        case OptimizationGUI::EventType::Canceled:
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Warning, "optimization-canceled",
                                      "Optimization canceled", std::move(event->message));
            break;
        case OptimizationGUI::EventType::Failed:
            notification_center_.Push(fdn_sandbox::NotificationSeverity::Error, "optimization-failed",
                                      "Optimization failed", std::move(event->message), 6.0);
            break;
        }
    }
}
