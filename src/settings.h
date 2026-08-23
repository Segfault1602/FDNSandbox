#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "quill/Backend.h"
#include "quill/Frontend.h"
#include "quill/Logger.h"
#include "quill/sinks/ConsoleSink.h"

class Settings
{
  public:
    static Settings& Instance()
    {
        static Settings instance;
        return instance;
    }

    [[nodiscard]] uint32_t SampleRate() const noexcept
    {
        return sample_rate_;
    }

    template <std::floating_point T>
    [[nodiscard]] T SampleRateAs() const noexcept
    {
        return static_cast<T>(sample_rate_);
    }

    [[nodiscard]] uint32_t BlockSize() const noexcept
    {
        return block_size_; // Fixed block size for processing
    }

    [[nodiscard]] quill::Logger* GetLogger() const noexcept
    {
        return logger_;
    }

  private:
    Settings()
    {
        quill::Backend::start();
        logger_ = quill::Frontend::create_or_get_logger(
            "root", quill::Frontend::create_or_get_sink<quill::ConsoleSink>("sink_id_1"));
    }

    uint32_t sample_rate_ = 48000; // Default sample rate
    uint32_t block_size_ = 64;     // Fixed block size for processing
    quill::Logger* logger_;
};