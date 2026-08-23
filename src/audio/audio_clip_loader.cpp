#include "audio/audio_clip_loader.h"

#include <samplerate.h>
#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace fdn_sandbox::audio
{
namespace
{

struct SndFileCloser
{
    void operator()(SNDFILE* file) const noexcept
    {
        if (file != nullptr)
        {
            sf_close(file);
        }
    }
};

} // namespace

std::expected<std::unique_ptr<AudioClip>, std::string> LoadAudioClip(const std::filesystem::path& path,
                                                                    std::uint32_t target_sample_rate)
{
    if (target_sample_rate == 0)
    {
        return std::unexpected("Target sample rate must be greater than zero.");
    }

    SF_INFO file_info{};
    const std::string path_string = path.string();
    std::unique_ptr<SNDFILE, SndFileCloser> file(sf_open(path_string.c_str(), SFM_READ, &file_info));
    if (file == nullptr)
    {
        return std::unexpected("Failed to open audio file: " + path_string);
    }
    if (file_info.frames <= 0 || file_info.channels <= 0 || file_info.samplerate <= 0)
    {
        return std::unexpected("Audio file has invalid metadata: frames=" + std::to_string(file_info.frames) +
                               ", channels=" + std::to_string(file_info.channels) +
                               ", sample_rate=" + std::to_string(file_info.samplerate));
    }

    const std::size_t frame_count = static_cast<std::size_t>(file_info.frames);
    const std::size_t channel_count = static_cast<std::size_t>(file_info.channels);
    std::vector<float> interleaved(frame_count * channel_count);
    const sf_count_t frames_read = sf_readf_float(file.get(), interleaved.data(), file_info.frames);
    if (frames_read != file_info.frames)
    {
        return std::unexpected("Failed to decode every audio frame.");
    }

    std::vector<float> mono(frame_count, 0.0f);
    for (std::size_t frame = 0; frame < frame_count; ++frame)
    {
        float sum = 0.0f;
        const std::size_t offset = frame * channel_count;
        for (std::size_t channel = 0; channel < channel_count; ++channel)
        {
            sum += interleaved[offset + channel];
        }
        mono[frame] = sum / static_cast<float>(channel_count);
    }

    if (static_cast<std::uint32_t>(file_info.samplerate) != target_sample_rate)
    {
        const double ratio = static_cast<double>(target_sample_rate) / static_cast<double>(file_info.samplerate);
        const std::size_t output_capacity =
            static_cast<std::size_t>(std::ceil(static_cast<double>(frame_count) * ratio)) + 1;
        std::vector<float> resampled(output_capacity);
        SRC_DATA resample_data{};
        resample_data.data_in = mono.data();
        resample_data.data_out = resampled.data();
        resample_data.input_frames = static_cast<long>(frame_count);
        resample_data.output_frames = static_cast<long>(output_capacity);
        resample_data.end_of_input = 1;
        resample_data.src_ratio = ratio;
        if (const int error = src_simple(&resample_data, SRC_SINC_BEST_QUALITY, 1); error != 0)
        {
            return std::unexpected(std::string("Failed to resample audio clip: ") + src_strerror(error));
        }
        if (resample_data.input_frames_used != static_cast<long>(frame_count))
        {
            return std::unexpected("Resampler did not consume every input frame.");
        }
        resampled.resize(static_cast<std::size_t>(resample_data.output_frames_gen));
        mono = std::move(resampled);
    }
    if (mono.empty())
    {
        return std::unexpected("Decoded audio clip contains no samples.");
    }

    return std::make_unique<AudioClip>(AudioClip{
        .samples = std::move(mono),
        .sample_rate = target_sample_rate,
    });
}

} // namespace fdn_sandbox::audio
