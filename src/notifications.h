#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace fdn_sandbox
{
enum class NotificationSeverity
{
    Info,
    Success,
    Warning,
    Error,
};

struct Notification
{
    NotificationSeverity severity = NotificationSeverity::Info;
    std::string id;
    std::string title;
    std::string message;
    double expires_at = 0.0;
};

class NotificationCenter
{
  public:
    void Push(NotificationSeverity severity, std::string id, std::string title, std::string message,
              double lifetime_seconds = 4.0);
    void SetCritical(NotificationSeverity severity, std::string id, std::string title, std::string message);
    void ClearCritical(std::string_view id);

    void DrawToasts();
    void DrawCriticalBanner();

  private:
    std::vector<Notification> notifications_;
    std::optional<Notification> critical_notification_;
};
} // namespace fdn_sandbox
