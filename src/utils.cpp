#include "utils.h"

#include <Eigen/Core>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/math/statistics/linear_regression.hpp>
#include <quill/LogMacros.h>
#include <sndfile.h>

#include <cassert>
#include <complex>
#include <mdspan>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#include "settings.h"
#include "sffdn/delay_utils.h"
#include <audio_utils/fft_utils.h>

namespace
{
Eigen::ArrayXcf Polyval(const Eigen::ArrayXf& p, const Eigen::ArrayXcf& x)
{
    Eigen::ArrayXcf result = Eigen::ArrayXcf::Zero(x.size());
    result += p[0];

    for (size_t i = 1; i < p.size(); ++i)
    {
        result = x * result + p[i];
    }

    return result;
}

// Helper function to check if a number is prime
bool isPrime(uint32_t n)
{
    if (n <= 1)
    {
        return false;
    }
    for (uint32_t i = 2; i * i <= n; ++i)
    {
        if (n % i == 0)
        {
            return false;
        }
    }
    return true;
}

float ComputeRMSImpl(std::span<const float> buffer)
{
    Eigen::Map<const Eigen::ArrayXf> buffer_map(buffer.data(), buffer.size());
    float rms = std::sqrt(buffer_map.square().mean());
    return rms;
}

} // namespace

namespace utils
{
bool IsPowerOfTwo(size_t n)
{
    return (n != 0) && ((n & (n - 1)) == 0);
}

template <typename T>
std::vector<T> LogSpace(T start, T stop, size_t num)
{
    std::vector<T> result(num);
    if (num == 0)
    {
        return result;
    }

    Eigen::Map<Eigen::ArrayX<T>> result_map(result.data(), num);

    result_map = Eigen::ArrayX<T>::LinSpaced(num, start, stop);
    result_map = Eigen::pow(10, result_map);

    return result;
}

template <typename T>
std::vector<T> Linspace(T start, T stop, size_t num)
{
    std::vector<T> result(num);
    if (num == 0)
    {
        return result;
    }

    Eigen::Map<Eigen::ArrayX<T>> result_map(result.data(), num);

    result_map = Eigen::ArrayX<T>::LinSpaced(num, start, stop);

    return result;
}

std::vector<float> pchip(std::span<const float> x, std::span<const float> y, std::span<const float> xq)
{
    std::vector<float> x_copy(x.begin(), x.end());
    std::vector<float> y_copy(y.begin(), y.end());
    auto spline = boost::math::interpolators::pchip(std::move(x_copy), std::move(y_copy));

    std::vector<float> yq;
    yq.reserve(xq.size());
    for (float i : xq)
    {
        yq.push_back(spline(i));
    }

    return yq;
}

std::vector<float> AbsFreqz(std::span<const float> sos, std::span<const float> w, size_t sr)
{
    // sos must be a multiple of 6 coefficients
    if (sos.size() % 6 != 0)
    {
        throw std::runtime_error("SOS coefficients must be a multiple of 6");
    }
    if (sos.size() < 6)
    {
        throw std::runtime_error("SOS coefficients must have at least 6 coefficients");
    }

    const size_t K = sos.size() / 6;

    Eigen::Map<const Eigen::ArrayXf> w_map(w.data(), w.size());
    Eigen::ArrayXcf dig_w(w.size());
    // if sample rate is specified, convert to rad/sample
    if (sr != 0.0f)
    {
        dig_w = Eigen::exp(std::complex(0.0f, 1.0f) * w_map * (-2.0f * std::numbers::pi_v<float> / sr));
    }
    else
    {
        dig_w = Eigen::exp(std::complex(0.0f, 1.0f) * w_map);
    }

    Eigen::Map<const Eigen::ArrayXf> b_map(sos.data(), 3);
    Eigen::Map<const Eigen::ArrayXf> a_map(sos.data() + 3, 3);

    Eigen::ArrayXcf h_complex = Polyval(b_map, dig_w) / Polyval(a_map, dig_w);

    for (size_t i = 1; i < K; ++i)
    {
        Eigen::Map<const Eigen::ArrayXf> b_map(sos.data() + (i * 6), 3);
        Eigen::Map<const Eigen::ArrayXf> a_map(sos.data() + (i * 6) + 3, 3);
        Eigen::ArrayXcf h = Polyval(b_map, dig_w) / Polyval(a_map, dig_w);

        h_complex = h_complex * h;
    }

    std::vector<float> h(w.size(), 0.0f);
    Eigen::Map<Eigen::ArrayXf> h_map(h.data(), h.size());
    h_map = h_complex.abs();

    return h;
}

void WriteAudioFile(const std::string& filename, std::span<const float> audio_data, int sample_rate)
{
    SF_INFO sf_info{};
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT; // Use WAV format with float samples
    sf_info.samplerate = sample_rate;
    sf_info.channels = 1; // Mono audio

    SNDFILE* sndfile = sf_open(filename.c_str(), SFM_WRITE, &sf_info);
    if (sndfile == nullptr)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file for writing: {}", sf_strerror(nullptr));
        return;
    }

    sf_count_t write_count = sf_writef_float(sndfile, audio_data.data(), audio_data.size());
    if (write_count != static_cast<sf_count_t>(audio_data.size()))
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to write audio file: {}", sf_strerror(sndfile));
    }

    sf_close(sndfile);
}

uint32_t GetChannelCountFromAudioFile(const std::string_view filename)
{
    SF_INFO sf_info{};
    SNDFILE* sndfile = sf_open(filename.data(), SFM_READ, &sf_info);
    if (sndfile == nullptr)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file for reading: {}", sf_strerror(nullptr));
        return {};
    }
    return sf_info.channels;
}

std::vector<float> ReadAudioFile(const std::string_view filename, uint32_t channel)
{
    SF_INFO sf_info{};
    SNDFILE* sndfile = sf_open(filename.data(), SFM_READ, &sf_info);
    if (sndfile == nullptr)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file for reading: {}", sf_strerror(nullptr));
        return {};
    }

    channel = std::min(channel, static_cast<uint32_t>(sf_info.channels - 1));

    std::vector<float> frame(sf_info.channels, 0.0f);

    std::vector<float> audio_data(sf_info.frames, 0.0f);
    for (size_t i = 0; i < static_cast<size_t>(sf_info.frames); ++i)
    {
        sf_readf_float(sndfile, frame.data(), 1);
        audio_data[i] = frame[channel];
    }

    sf_close(sndfile);
    return audio_data;
}

uint32_t GetClosestPrime(uint32_t n)
{
    if (n < 2)
    {
        return 2; // The smallest prime number
    }

    // Check if n is prime
    if (isPrime(n))
    {
        return n;
    }

    // Search for the closest prime number
    uint32_t lower = n;
    uint32_t upper = n;

    while (true)
    {
        if (isPrime(lower))
        {
            return lower;
        }
        if (isPrime(upper))
        {
            return upper;
        }
        --lower;
        ++upper;
    }
}

std::string GetMatrixName(sfFDN::ScalarMatrixType type)
{
    switch (type)
    {
    case sfFDN::ScalarMatrixType::Identity:
        return "Identity";
    case sfFDN::ScalarMatrixType::Random:
        return "Random Orthogonal";
    case sfFDN::ScalarMatrixType::Householder:
        return "Householder";
    case sfFDN::ScalarMatrixType::RandomHouseholder:
        return "Random Householder";
    case sfFDN::ScalarMatrixType::Hadamard:
        return "Hadamard";
    case sfFDN::ScalarMatrixType::Circulant:
        return "Circulant";
    case sfFDN::ScalarMatrixType::Allpass:
        return "Allpass";
    case sfFDN::ScalarMatrixType::NestedAllpass:
        return "Nested Allpass";
    case sfFDN::ScalarMatrixType::VariableDiffusion:
        return "Variable Diffusion";
    default:
        return "Unknown";
    }
}

std::string GetDelayLengthTypeName(int type)
{
    if (type < static_cast<int>(sfFDN::DelayLengthType::Count))
    {
        switch (static_cast<sfFDN::DelayLengthType>(type))
        {
        case sfFDN::DelayLengthType::Random:
            return "Random";
        case sfFDN::DelayLengthType::Gaussian:
            return "Gaussian";
        case sfFDN::DelayLengthType::Primes:
            return "Primes";
        case sfFDN::DelayLengthType::Uniform:
            return "Uniform";
        case sfFDN::DelayLengthType::PrimePower:
            return "Prime Power";
        case sfFDN::DelayLengthType::SteamAudio:
            return "Steam Audio";
        default:
            return "Unknown";
        }
    }

    if (type == static_cast<int>(sfFDN::DelayLengthType::Count))
    {
        return "Mean Delay";
    }
    return "Unknown";
}

std::vector<float> T60ToGainsDb(std::span<const float> t60s, uint32_t delay, size_t sample_rate)
{
    std::vector<float> gains(t60s.size(), 0.0f);
    for (size_t i = 0; i < t60s.size(); ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], static_cast<float>(delay) / sample_rate);
        gains[i] = 20.f * std::log10(gains[i]);
    }

    return gains;
}

std::vector<float> ComputeRMS(std::span<const float> buffer, uint32_t block_size, uint32_t hop_size)
{
    std::vector<float> rms_values;
    for (size_t i = 0; i + block_size <= buffer.size(); i += hop_size)
    {
        auto block = buffer.subspan(i, block_size);
        float rms = ComputeRMSImpl(block);
        rms_values.push_back(rms);
    }
    return rms_values;
}

template std::vector<double> LogSpace(double start, double stop, size_t num);
template std::vector<float> LogSpace(float start, float stop, size_t num);
template std::vector<double> Linspace(double start, double stop, size_t num);
template std::vector<float> Linspace(float start, float stop, size_t num);
} // namespace utils