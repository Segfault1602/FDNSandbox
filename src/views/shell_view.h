#pragma once

#include "files/file_dialogs.h"
#include "session/fdn_session.h"

#include <imgui.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace fdn_sandbox::audio
{
class AudioEngine;
}

namespace fdn_sandbox::views
{

class AudioView;

struct ShellActions
{
    std::optional<files::FileDialogRequest> file_dialog;
    std::optional<session::FdnSlot> save_slot;
    std::optional<session::FdnSlot> switch_slot;
    std::optional<int> play_audio_file;
    bool reset_layout = false;
};

struct StatusBarData
{
    bool stream_running = false;
    std::uint32_t sample_rate = 0;
    std::string_view reference_name;
};

/**
 * @brief Renders the application shell and returns typed user actions.
 *
 * Menus, toolbar, status bar, audio settings, and default docking are kept
 * separate from the app's workflow and publication logic.
 */
class ShellView final
{
  public:
    [[nodiscard]] ShellActions DrawMainMenuBar(AudioView& audio_view, audio::AudioEngine& engine);
    [[nodiscard]] ShellActions DrawToolbar(AudioView& audio_view, audio::AudioEngine& engine,
                                           session::FdnSlot active_slot, bool optimization_active);
    void DrawStatusBar(const StatusBarData& data, const AudioView& audio_view);
    static void BuildDefaultDockLayout(ImGuiID dockspace_id);

  private:
    static void DrawFrameRateStatus();
    static std::optional<files::FileDialogRequest> DrawFileMenu();
    std::optional<session::FdnSlot> DrawOptionsMenu();
    static bool DrawViewMenu();
    static std::optional<session::FdnSlot> DrawConfigurationSwitcher(session::FdnSlot active_slot);
    static std::optional<files::FileDialogRequest> DrawQuickFileActions();
    void DrawAudioConfigurationWindow(AudioView& audio_view, audio::AudioEngine& engine);

    bool show_audio_config_window_ = false;
#ifndef NDEBUG
    bool fontaudio_verified_ = false;
#endif
};

} // namespace fdn_sandbox::views
