#pragma once

#include <cstddef>

class Settings
{
  public:
    static Settings& Instance()
    {
        static Settings instance;
        return instance;
    }

    Settings(const Settings&) = delete;
    Settings& operator=(const Settings&) = delete;

    size_t SampleRate() const
    {
        return sample_rate_;
    }

    size_t IRDuration() const
    {
        return ir_duration_;
    }

    void SetIRDuration(size_t duration)
    {
        if (duration > 0)
        {
            ir_duration_ = duration;
        }
    }

  private:
    Settings() = default;
    ~Settings() = default;

    size_t sample_rate_ = 48000; // Default sample rate
    size_t ir_duration_ = 1;     // Default impulse response duration in seconds
};