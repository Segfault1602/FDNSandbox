#include "notifications.h"

#include "icons.h"
#include "theme.h"

#include "imgui_internal.h"

#include <algorithm>
#include <ranges>
#include <utility>

namespace fdn_sandbox
{
namespace
{
ImVec4 SeverityColor(NotificationSeverity severity)
{
    switch (severity)
    {
    case NotificationSeverity::Success:
        return theme::Color(theme::ColorRole::StatusOk);
    case NotificationSeverity::Warning:
        return theme::Color(theme::ColorRole::StatusWarning);
    case NotificationSeverity::Error:
        return theme::Color(theme::ColorRole::StatusError);
    case NotificationSeverity::Info:
        return theme::Color(theme::ColorRole::Accent);
    }
    return theme::Color(theme::ColorRole::Accent);
}
} // namespace

void NotificationCenter::Push(NotificationSeverity severity, std::string id, std::string title, std::string message,
                              double lifetime_seconds)
{
    const double expires_at = ImGui::GetTime() + lifetime_seconds;
    const auto existing = std::ranges::find(notifications_, id, &Notification::id);
    if (existing != notifications_.end())
    {
        existing->severity = severity;
        existing->title = std::move(title);
        existing->message = std::move(message);
        existing->expires_at = expires_at;
        return;
    }
    notifications_.push_back(Notification{
        .severity = severity,
        .id = std::move(id),
        .title = std::move(title),
        .message = std::move(message),
        .expires_at = expires_at,
    });
}

void NotificationCenter::SetCritical(NotificationSeverity severity, std::string id, std::string title,
                                     std::string message)
{
    critical_notification_ = Notification{
        .severity = severity,
        .id = std::move(id),
        .title = std::move(title),
        .message = std::move(message),
        .expires_at = 0.0,
    };
}

void NotificationCenter::ClearCritical(std::string_view id)
{
    if (critical_notification_.has_value() && critical_notification_->id == id)
    {
        critical_notification_.reset();
    }
}

void NotificationCenter::DrawToasts()
{
    const double now = ImGui::GetTime();
    std::erase_if(notifications_, [now](const Notification& notification) { return notification.expires_at <= now; });

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    float y_offset = 12.0f;
    for (const auto& notification : std::views::reverse(notifications_))
    {
        const float alpha = static_cast<float>(std::clamp((notification.expires_at - now) / 0.3, 0.0, 1.0));
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 12.0f,
                                       viewport->WorkPos.y + viewport->WorkSize.y - y_offset),
                                ImGuiCond_Always, ImVec2(1.0f, 1.0f));
        ImGui::SetNextWindowSizeConstraints(ImVec2(280.0f, 0.0f), ImVec2(380.0f, 160.0f));
        ImGui::SetNextWindowBgAlpha(0.94f * alpha);

        const std::string window_id = "##Toast_" + notification.id;
        constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                           ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing;
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
        ImGui::Begin(window_id.c_str(), nullptr, flags);
        const ImVec4 color = SeverityColor(notification.severity);
        const ImVec2 window_pos = ImGui::GetWindowPos();
        ImGui::GetWindowDrawList()->AddRectFilled(window_pos,
                                                  ImVec2(window_pos.x + 4.0f, window_pos.y + ImGui::GetWindowHeight()),
                                                  ImGui::ColorConvertFloat4ToU32(color));
        ImGui::TextColored(color, "%s %s", notification.severity == NotificationSeverity::Error ? icons::Warning : "",
                           notification.title.c_str());
        ImGui::TextWrapped("%s", notification.message.c_str());
        y_offset += ImGui::GetWindowHeight() + 8.0f;
        ImGui::End();
        ImGui::PopStyleVar();
    }
}

void NotificationCenter::DrawCriticalBanner()
{
    if (!critical_notification_.has_value())
    {
        return;
    }

    const ImGuiStyle& style = ImGui::GetStyle();
    const ImVec2 padding(style.WindowPadding.x, style.FramePadding.y);
    const float height = ImGui::GetFrameHeight() + (padding.y * 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, padding);
    const bool open =
        ImGui::BeginViewportSideBar("##CriticalNotification", ImGui::GetMainViewport(), ImGuiDir_Up, height,
                                    ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
                                        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    if (!open)
    {
        ImGui::End();
        return;
    }

    const ImVec4 color = SeverityColor(critical_notification_->severity);
    ImGui::TextColored(color, "%s %s:", icons::Warning, critical_notification_->title.c_str());
    ImGui::SameLine();
    ImGui::TextUnformatted(critical_notification_->message.c_str());
    ImGui::End();
}
} // namespace fdn_sandbox
