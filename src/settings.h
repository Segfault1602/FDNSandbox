#pragma once

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

    uint32_t SampleRate() const
    {
        return sample_rate_;
    }

    uint32_t IRDuration() const
    {
        return ir_duration_;
    }

    void SetIRDuration(uint32_t duration)
    {
        if (duration > 0)
        {
            ir_duration_ = duration;
        }
    }

    uint32_t BlockSize() const
    {
        return block_size_; // Fixed block size for processing
    }

    void SetBlockSize(uint32_t block_size)
    {
        block_size_ = block_size;
    }

    quill::Logger* GetLogger() const
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

    size_t sample_rate_ = 48000; // Default sample rate
    size_t ir_duration_ = 1;     // Default impulse response duration in seconds

    uint32_t block_size_ = 64; // Fixed block size for processing
    quill::Logger* logger_;
};