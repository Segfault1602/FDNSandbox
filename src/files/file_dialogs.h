#pragma once

#include <imgui.h>

#include <cassert>

#include <imfilebrowser.h>

#include <cstdint>
#include <filesystem>
#include <variant>
#include <vector>

namespace fdn_sandbox::files
{

enum class FileDialogRequest : std::uint8_t
{
    SaveImpulseResponse,
    LoadConfiguration,
    SaveConfiguration,
    LoadRir
};

struct SaveImpulseResponseSelection
{
    std::filesystem::path path;
};

struct LoadConfigurationSelection
{
    std::filesystem::path path;
};

struct SaveConfigurationSelection
{
    std::filesystem::path path;
};

struct LoadRirSelection
{
    std::filesystem::path path;
};

/// Typed path selections returned by the application-level file dialogs.
using FileDialogSelection = std::variant<SaveImpulseResponseSelection, LoadConfigurationSelection,
                                         SaveConfigurationSelection, LoadRirSelection>;

/**
 * @brief Owns and renders the application-level ImGui file browsers.
 *
 * Dialog configuration, extension normalization, and selection clearing stay
 * inside this UI-only component.
 */
class FileDialogs final
{
  public:
    FileDialogs();

    void Open(FileDialogRequest request);
    std::vector<FileDialogSelection> Draw();

  private:
    ImGui::FileBrowser save_ir_;
    ImGui::FileBrowser load_config_;
    ImGui::FileBrowser save_config_;
    ImGui::FileBrowser load_rir_;
};

} // namespace fdn_sandbox::files
