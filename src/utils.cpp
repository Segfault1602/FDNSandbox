#include "utils.h"

#include <Eigen/Core>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/math/statistics/linear_regression.hpp>
#include <quill/LogMacros.h>
#include <sndfile.h>

#include <cassert>
#include <complex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

#include "settings.h"
#include "sffdn/delay_utils.h"
#include <audio_utils/fft_utils.h>

namespace
{
// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

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

std::vector<float> AbsFreqz(std::span<const sfFDN::FilterCoefficients> sos, std::span<const float> w, size_t sr)
{
    const size_t K = sos.size();

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

    Eigen::Array3f b_coeffs = {sos[0].b0, sos[0].b1, sos[0].b2};
    Eigen::Array3f a_coeffs = {sos[0].a0, sos[0].a1, sos[0].a2};

    Eigen::ArrayXcf h_complex = Polyval(b_coeffs, dig_w) / Polyval(a_coeffs, dig_w);

    for (size_t i = 1; i < K; ++i)
    {
        Eigen::Array3f b_coeffs = {sos[i].b0, sos[i].b1, sos[i].b2};
        Eigen::Array3f a_coeffs = {sos[i].a0, sos[i].a1, sos[i].a2};
        // Eigen::Map<const Eigen::ArrayXf> b_map(sos.data() + (i * 6), 3);
        // Eigen::Map<const Eigen::ArrayXf> a_map(sos.data() + (i * 6) + 3, 3);
        Eigen::ArrayXcf h = Polyval(b_coeffs, dig_w) / Polyval(a_coeffs, dig_w);

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

std::string GetDelayInterpolationTypeName(int type)
{
    sfFDN::DelayInterpolationType interp_type = static_cast<sfFDN::DelayInterpolationType>(type);
    switch (interp_type)
    {
    case sfFDN::DelayInterpolationType::None:
        return "None";
    case sfFDN::DelayInterpolationType::Linear:
        return "Linear";
    case sfFDN::DelayInterpolationType::Allpass:
        return "Allpass";
    case sfFDN::DelayInterpolationType::Lagrange:
        return "Lagrange";
    default:
        return "Unknown";
    }
}

sfFDN::AttenuationFilterBankOptions FindAttenuationFilterBankOptions(sfFDN::FDNConfig2& config)
{
    for (auto& config_variant : config.loop_filter_configs)
    {
        if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(config_variant))
        {
            return std::get<sfFDN::AttenuationFilterBankOptions>(config_variant);
        }
    }

    assert(false);
    // Create a default one if not found
    sfFDN::AttenuationFilterBankOptions default_config;
    for (auto i = 0; i < config.fdn_size; ++i)
    {
        default_config.filter_configs.emplace_back(sfFDN::ProportionalAttenuationOptions{
            .t60 = 1.f, .delay = -1.f, .sample_rate = static_cast<float>(config.sample_rate)});
    }

    config.loop_filter_configs.emplace_back(default_config);
    return default_config;
}

void ReplaceAttenuationFilterBankOptions(sfFDN::FDNConfig2& config,
                                         const sfFDN::AttenuationFilterBankOptions& new_options)
{
    for (auto i = 0u; i < config.loop_filter_configs.size(); ++i)
    {
        if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(config.loop_filter_configs[i]))
        {
            config.loop_filter_configs[i] = new_options;
            return;
        }
    }
}

std::vector<float> T60ToGainsDb(std::span<const float> t60s, uint32_t delay, size_t sample_rate)
{
    std::vector<float> gains(t60s.size(), 0.0f);
    for (size_t i = 0; i < t60s.size(); ++i)
    {
        // gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        // gains[i] = std::pow(gains[i], static_cast<float>(delay) / sample_rate);
        // gains[i] = 20.f * std::log10(gains[i]);
        gains[i] = -60.f / (t60s[i] * sample_rate);
        gains[i] *= static_cast<float>(delay);
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

void ResizeMultichannelProcessorConfigs(sfFDN::multi_channel_processor_variant_t& config_variant, uint32_t new_size)
{
    std::visit(
        overloaded{
            [new_size](sfFDN::ParallelGainsOptions& config) { config.gains.resize(new_size, 0.5f); },
            [new_size](sfFDN::ParallelSchroederAllpassSectionOptions& config) { config.sections.resize(new_size); },
            [new_size](sfFDN::AttenuationFilterBankOptions& config) {
                auto previous_size = config.filter_configs.size();
                config.filter_configs.resize(new_size);

                auto last_config = config.filter_configs.back();
                for (size_t i = previous_size; i < new_size; ++i)
                {
                    config.filter_configs[i] = last_config;
                }
            },
            [new_size](sfFDN::DelayBankOptions& config) { config.delays.resize(new_size, 512.f); },
            [new_size](sfFDN::DelayBankTimeVaryingOptions& config) { config.delays.resize(new_size, 512.f); },
            [new_size](sfFDN::CascadedFeedbackMatrixOptions& config) { config.matrix_size = new_size; },
            [new_size](sfFDN::ScalarFeedbackMatrixOptions& config) {
                config.matrix_size = new_size;
                if (config.custom_matrix.has_value())
                {
                    config.custom_matrix->resize(new_size * new_size, 0.f);
                }
            },
        },
        config_variant);
}

void ResizeFDNConfig(sfFDN::FDNConfig2& config, uint32_t new_size)
{
    config.fdn_size = new_size;

    config.delay_bank_config.delays.resize(new_size, 512.f);

    config.input_block_config.parallel_gains_config.gains.resize(new_size, 0.5f);

    for (auto& processor_variant : config.input_block_config.multichannel_processors)
    {
        ResizeMultichannelProcessorConfigs(processor_variant, new_size);
    }

    config.output_block_config.parallel_gains_config.gains.resize(new_size, 0.5f);
    for (auto& processor_variant : config.output_block_config.multichannel_processors)
    {
        ResizeMultichannelProcessorConfigs(processor_variant, new_size);
    }

    for (auto& processor_variant : config.loop_filter_configs)
    {
        ResizeMultichannelProcessorConfigs(processor_variant, new_size);
    }

    std::visit([new_size](auto&& feedback_matrix_config) { feedback_matrix_config.matrix_size = new_size; },
               config.feedback_matrix_config);
}

std::string GetProcessorName(const sfFDN::single_channel_processor_variant_t& processor_variant)
{
    return std::visit(overloaded{
                          [](const sfFDN::SchroederAllpassSectionOptions&) { return "Schroeder Allpass"; },
                          [](const sfFDN::AllpassFilterOptions&) { return "Allpass Filter"; },
                          [](const sfFDN::CascadedBiquadsOptions&) { return "Cascaded Biquads"; },
                          [](const sfFDN::FirOptions&) { return "FIR Filter"; },
                          [](const sfFDN::DelayOptions&) { return "Delay"; },
                          [](const sfFDN::GraphicEQOptions&) { return "Graphic EQ"; },
                      },
                      processor_variant);
}

std::string GetProcessorName(const sfFDN::multi_channel_processor_variant_t& processor_variant)
{
    return std::visit(
        overloaded{
            [](const sfFDN::ParallelGainsOptions&) { return "Parallel Gains"; },
            [](const sfFDN::ParallelSchroederAllpassSectionOptions&) { return "Parallel Schroeder Allpass"; },
            [](const sfFDN::AttenuationFilterBankOptions&) { return "Attenuation Filter Bank"; },
            [](const sfFDN::DelayBankOptions&) { return "Delay Bank"; },
            [](const sfFDN::DelayBankTimeVaryingOptions&) { return "Time-Varying Delay Bank"; },
            [](const sfFDN::CascadedFeedbackMatrixOptions&) { return "Cascaded Feedback Matrix"; },
            [](const sfFDN::ScalarFeedbackMatrixOptions&) { return "Scalar Feedback Matrix"; },
        },
        processor_variant);
}

template std::vector<double> LogSpace(double start, double stop, size_t num);
template std::vector<float> LogSpace(float start, float stop, size_t num);
template std::vector<double> Linspace(double start, double stop, size_t num);
template std::vector<float> Linspace(float start, float stop, size_t num);
} // namespace utils