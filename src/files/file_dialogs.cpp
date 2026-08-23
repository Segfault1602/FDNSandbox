#include "files/file_dialogs.h"

#include <filesystem>
#include <string>
#include <vector>

namespace fdn_sandbox::files
{
namespace
{

std::filesystem::path EnsureExtension(const std::filesystem::path& path, const std::string& extension)
{
    std::string value = path.string();
    if (!value.ends_with(extension))
    {
        value += extension;
    }
    return value;
}

} // namespace

FileDialogs::FileDialogs()
    : save_ir_(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_config_(0)
    , save_config_(ImGuiFileBrowserFlags_EnterNewFilename | ImGuiFileBrowserFlags_CreateNewDir)
    , load_rir_(0)
{
    save_ir_.SetTitle("Save Impulse Response");
    save_ir_.SetTypeFilters({".wav"});

    load_config_.SetTitle("Load FDN Configuration");
    load_config_.SetTypeFilters({".json"});

    save_config_.SetTitle("Save FDN Configuration");
    save_config_.SetTypeFilters({".json"});

    load_rir_.SetTitle("Load RIR File");
    load_rir_.SetTypeFilters({".wav"});
}

void FileDialogs::Open(FileDialogRequest request)
{
    switch (request)
    {
    case FileDialogRequest::SaveImpulseResponse:
        save_ir_.Open();
        break;
    case FileDialogRequest::LoadConfiguration:
        load_config_.Open();
        break;
    case FileDialogRequest::SaveConfiguration:
        save_config_.Open();
        break;
    case FileDialogRequest::LoadRir:
        load_rir_.Open();
        break;
    }
}

std::vector<FileDialogSelection> FileDialogs::Draw()
{
    save_ir_.Display();
    load_config_.Display();
    save_config_.Display();
    load_rir_.Display();

    std::vector<FileDialogSelection> selections;
    selections.reserve(4);
    if (save_ir_.HasSelected())
    {
        selections.emplace_back(SaveImpulseResponseSelection{.path = EnsureExtension(save_ir_.GetSelected(), ".wav")});
        save_ir_.ClearSelected();
    }
    if (load_config_.HasSelected())
    {
        selections.emplace_back(LoadConfigurationSelection{.path = load_config_.GetSelected()});
        load_config_.ClearSelected();
    }
    if (save_config_.HasSelected())
    {
        selections.emplace_back(
            SaveConfigurationSelection{.path = EnsureExtension(save_config_.GetSelected(), ".json")});
        save_config_.ClearSelected();
    }
    if (load_rir_.HasSelected())
    {
        selections.emplace_back(LoadRirSelection{.path = load_rir_.GetSelected()});
        load_rir_.ClearSelected();
    }
    return selections;
}

} // namespace fdn_sandbox::files
