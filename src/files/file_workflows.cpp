#include "files/file_workflows.h"

#include <samplerate.h>
#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fdn_sandbox::files
{
namespace
{

class SndFileHandle final
{
  public:
    explicit SndFileHandle(SNDFILE* file) noexcept
        : file_(file)
    {
    }

    ~SndFileHandle()
    {
        if (file_ != nullptr)
        {
            sf_close(file_);
        }
    }

    SndFileHandle(const SndFileHandle&) = delete;
    SndFileHandle& operator=(const SndFileHandle&) = delete;

    SNDFILE* Get() const noexcept
    {
        return file_;
    }

    int Close() noexcept
    {
        if (file_ == nullptr)
        {
            return 0;
        }
        SNDFILE* file = std::exchange(file_, nullptr);
        return sf_close(file);
    }

  private:
    SNDFILE* file_;
};

struct DecodedAudio
{
    std::uint32_t sample_rate;
    std::size_t channels;
    std::size_t frames;
    std::vector<float> interleaved;
};

FileError MakeError(FileErrorKind kind, const std::filesystem::path& path, std::string message)
{
    return FileError{
        .kind = kind,
        .path = path,
        .message = std::move(message),
    };
}

std::expected<DecodedAudio, FileError> DecodeAudioFile(const std::filesystem::path& path)
{
    SF_INFO file_info{};
    const std::string path_string = path.string();
    SndFileHandle file(sf_open(path_string.c_str(), SFM_READ, &file_info));
    if (file.Get() == nullptr)
    {
        return std::unexpected(MakeError(FileErrorKind::OpenFailed, path,
                                         "Failed to open audio file: " + std::string(sf_strerror(nullptr))));
    }
    if (file_info.frames == 0)
    {
        return std::unexpected(MakeError(FileErrorKind::EmptyAudio, path, "Audio file contains no samples."));
    }
    if (file_info.frames < 0 || file_info.channels <= 0 || file_info.samplerate <= 0)
    {
        return std::unexpected(MakeError(FileErrorKind::InvalidMetadata, path,
                                         "Audio file has invalid metadata: frames=" + std::to_string(file_info.frames) +
                                             ", channels=" + std::to_string(file_info.channels) +
                                             ", sample_rate=" + std::to_string(file_info.samplerate)));
    }

    const auto frame_count = static_cast<std::uintmax_t>(file_info.frames);
    const auto channel_count = static_cast<std::uintmax_t>(file_info.channels);
    if (frame_count > std::numeric_limits<std::size_t>::max() ||
        channel_count > std::numeric_limits<std::size_t>::max() ||
        frame_count > std::numeric_limits<std::size_t>::max() / channel_count)
    {
        return std::unexpected(
            MakeError(FileErrorKind::InvalidMetadata, path, "Audio file dimensions exceed addressable memory."));
    }

    const std::size_t frames = static_cast<std::size_t>(frame_count);
    const std::size_t channels = static_cast<std::size_t>(channel_count);
    std::vector<float> interleaved(frames * channels);
    const sf_count_t frames_read = sf_readf_float(file.Get(), interleaved.data(), file_info.frames);
    if (frames_read != file_info.frames)
    {
        return std::unexpected(
            MakeError(FileErrorKind::DecodeFailed, path,
                      "Failed to decode every audio frame: " + std::string(sf_strerror(file.Get()))));
    }

    return DecodedAudio{
        .sample_rate = static_cast<std::uint32_t>(file_info.samplerate),
        .channels = channels,
        .frames = frames,
        .interleaved = std::move(interleaved),
    };
}

} // namespace

std::expected<LoadedRir, FileError> FileWorkflows::LoadRir(const std::filesystem::path& path,
                                                           std::size_t convolution_block_size) const
{
    if (convolution_block_size == 0)
    {
        return std::unexpected(
            MakeError(FileErrorKind::InvalidArgument, path, "Convolution block size must be greater than zero."));
    }

    auto decoded = DecodeAudioFile(path);
    if (!decoded)
    {
        return std::unexpected(std::move(decoded.error()));
    }
    if (decoded->channels != 1)
    {
        return std::unexpected(
            MakeError(FileErrorKind::UnsupportedChannelCount, path,
                      "RIR file must be mono; found " + std::to_string(decoded->channels) + " channels."));
    }

    std::vector<float> mono_samples = std::move(decoded->interleaved);
    std::unique_ptr<sfFDN::PartitionedConvolver> convolver;
    try
    {
        convolver = std::make_unique<sfFDN::PartitionedConvolver>(convolution_block_size, mono_samples);
    }
    catch (const std::exception& error)
    {
        return std::unexpected(MakeError(FileErrorKind::ConvolverBuildFailed, path,
                                         "Failed to build partitioned convolver: " + std::string(error.what())));
    }

    return LoadedRir{
        .path = path,
        .effective_sample_rate = decoded->sample_rate,
        .mono_samples = std::move(mono_samples),
        .convolver = std::move(convolver),
    };
}

std::expected<audio::AudioClip, FileError> FileWorkflows::LoadClip(const std::filesystem::path& path,
                                                                   std::uint32_t engine_sample_rate) const
{
    if (engine_sample_rate == 0)
    {
        return std::unexpected(
            MakeError(FileErrorKind::InvalidArgument, path, "Target sample rate must be greater than zero."));
    }

    auto decoded = DecodeAudioFile(path);
    if (!decoded)
    {
        return std::unexpected(std::move(decoded.error()));
    }

    std::vector<float> mono(decoded->frames, 0.0f);
    for (std::size_t frame = 0; frame < decoded->frames; ++frame)
    {
        float sum = 0.0f;
        const std::size_t offset = frame * decoded->channels;
        for (std::size_t channel = 0; channel < decoded->channels; ++channel)
        {
            sum += decoded->interleaved[offset + channel];
        }
        mono[frame] = sum / static_cast<float>(decoded->channels);
    }

    if (decoded->sample_rate != engine_sample_rate)
    {
        if (mono.size() > static_cast<std::size_t>(std::numeric_limits<long>::max()))
        {
            return std::unexpected(
                MakeError(FileErrorKind::ResampleFailed, path, "Audio clip is too large for the resampler."));
        }

        const double ratio = static_cast<double>(engine_sample_rate) / static_cast<double>(decoded->sample_rate);
        const double required_capacity = std::ceil(static_cast<double>(mono.size()) * ratio) + 1.0;
        if (!std::isfinite(required_capacity) ||
            required_capacity > static_cast<double>(std::numeric_limits<long>::max()) ||
            required_capacity > static_cast<double>(std::numeric_limits<std::size_t>::max()))
        {
            return std::unexpected(
                MakeError(FileErrorKind::ResampleFailed, path, "Resampled clip size is not representable."));
        }

        std::vector<float> resampled(static_cast<std::size_t>(required_capacity));
        SRC_DATA resample_data{};
        resample_data.data_in = mono.data();
        resample_data.data_out = resampled.data();
        resample_data.input_frames = static_cast<long>(mono.size());
        resample_data.output_frames = static_cast<long>(resampled.size());
        resample_data.end_of_input = 1;
        resample_data.src_ratio = ratio;
        if (const int error = src_simple(&resample_data, SRC_SINC_BEST_QUALITY, 1); error != 0)
        {
            return std::unexpected(MakeError(FileErrorKind::ResampleFailed, path,
                                             "Failed to resample audio clip: " + std::string(src_strerror(error))));
        }
        if (resample_data.input_frames_used != static_cast<long>(mono.size()))
        {
            return std::unexpected(
                MakeError(FileErrorKind::ResampleFailed, path, "Resampler did not consume every input frame."));
        }
        resampled.resize(static_cast<std::size_t>(resample_data.output_frames_gen));
        mono = std::move(resampled);
    }
    if (mono.empty())
    {
        return std::unexpected(MakeError(FileErrorKind::EmptyAudio, path, "Decoded audio clip contains no samples."));
    }

    return audio::AudioClip{
        .samples = std::move(mono),
        .sample_rate = engine_sample_rate,
    };
}

std::expected<void, FileError> FileWorkflows::SaveImpulseResponse(const std::filesystem::path& path,
                                                                  std::span<const float> samples,
                                                                  std::uint32_t sample_rate) const
{
    if (sample_rate == 0 || sample_rate > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        return std::unexpected(
            MakeError(FileErrorKind::InvalidArgument, path, "Sample rate must fit in a positive integer."));
    }
    if (samples.empty())
    {
        return std::unexpected(
            MakeError(FileErrorKind::InvalidArgument, path, "Impulse response must contain at least one sample."));
    }
    if (samples.size() > static_cast<std::size_t>(std::numeric_limits<sf_count_t>::max()))
    {
        return std::unexpected(
            MakeError(FileErrorKind::InvalidArgument, path, "Impulse response is too large to write."));
    }

    SF_INFO file_info{};
    file_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    file_info.samplerate = static_cast<int>(sample_rate);
    file_info.channels = 1;

    const std::string path_string = path.string();
    SndFileHandle file(sf_open(path_string.c_str(), SFM_WRITE, &file_info));
    if (file.Get() == nullptr)
    {
        return std::unexpected(
            MakeError(FileErrorKind::OpenFailed, path,
                      "Failed to open audio file for writing: " + std::string(sf_strerror(nullptr))));
    }

    const sf_count_t frame_count = static_cast<sf_count_t>(samples.size());
    const sf_count_t write_count = sf_writef_float(file.Get(), samples.data(), frame_count);
    if (write_count != frame_count)
    {
        const std::string detail = sf_strerror(file.Get());
        file.Close();
        return std::unexpected(
            MakeError(FileErrorKind::WriteFailed, path, "Failed to write impulse response: " + detail));
    }

    if (const int close_error = file.Close(); close_error != 0)
    {
        return std::unexpected(
            MakeError(FileErrorKind::FinalizeFailed, path,
                      "Failed to finalize audio file: " + std::string(sf_error_number(close_error))));
    }
    return {};
}

} // namespace fdn_sandbox::files
