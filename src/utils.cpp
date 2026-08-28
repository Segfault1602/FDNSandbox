#include "utils.h"

#include <Eigen/Core>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/math/statistics/linear_regression.hpp>
#include <quill/LogMacros.h>
#include <sndfile.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "settings.h"
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

    for (Eigen::Index i = 1; i < p.size(); ++i)
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

std::vector<float> pchip(PchipInput input)
{
    std::vector<float> x_copy(input.x.begin(), input.x.end());
    std::vector<float> y_copy(input.y.begin(), input.y.end());
    auto spline = boost::math::interpolators::pchip(std::move(x_copy), std::move(y_copy));

    std::vector<float> yq;
    yq.reserve(input.xq.size());
    for (const float i : input.xq)
    {
        yq.push_back(spline(i));
    }

    return yq;
}

std::vector<float> AbsFreqz(std::span<const sfFDN::FilterCoefficients> sos, std::span<const float> w, size_t sr)
{
    const size_t K = sos.size();

    const auto w_length = static_cast<Eigen::Index>(w.size());
    const Eigen::Map<const Eigen::ArrayXf> w_map(w.data(), w_length);
    Eigen::ArrayXcf dig_w(static_cast<Eigen::Index>(w.size()));
    // if sample rate is specified, convert to rad/sample
    if (sr != 0U)
    {
        const auto sample_rate = static_cast<float>(sr);
        dig_w = Eigen::exp(std::complex(0.0f, 1.0f) * w_map * (-2.0f * std::numbers::pi_v<float> / sample_rate));
    }
    else
    {
        dig_w = Eigen::exp(std::complex(0.0f, 1.0f) * w_map);
    }

    const Eigen::Array3f b_coeffs = {sos[0].b0, sos[0].b1, sos[0].b2};
    const Eigen::Array3f a_coeffs = {sos[0].a0, sos[0].a1, sos[0].a2};

    Eigen::ArrayXcf h_complex = Polyval(b_coeffs, dig_w) / Polyval(a_coeffs, dig_w);

    for (size_t i = 1; i < K; ++i)
    {
        const Eigen::Array3f b_coeffs = {sos[i].b0, sos[i].b1, sos[i].b2};
        const Eigen::Array3f a_coeffs = {sos[i].a0, sos[i].a1, sos[i].a2};
        // Eigen::Map<const Eigen::ArrayXf> b_map(sos.data() + (i * 6), 3);
        // Eigen::Map<const Eigen::ArrayXf> a_map(sos.data() + (i * 6) + 3, 3);
        const Eigen::ArrayXcf h = Polyval(b_coeffs, dig_w) / Polyval(a_coeffs, dig_w);

        h_complex = h_complex * h;
    }

    std::vector<float> h(w.size(), 0.0f);
    const auto h_length = static_cast<Eigen::Index>(h.size());
    Eigen::Map<Eigen::ArrayXf> h_map(h.data(), h_length);
    h_map = h_complex.abs();

    return h;
}

uint32_t GetChannelCountFromAudioFile(std::string_view filename)
{
    const std::string filename_string(filename);
    SF_INFO sf_info{};
    SNDFILE* sndfile = sf_open(filename_string.c_str(), SFM_READ, &sf_info);
    if (sndfile == nullptr)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file for reading: {}", sf_strerror(nullptr));
        return {};
    }

    sf_close(sndfile);

    return sf_info.channels;
}

std::vector<float> ReadAudioFile(std::string_view filename, uint32_t channel)
{
    const std::string filename_string(filename);
    SF_INFO sf_info{};
    SNDFILE* sndfile = sf_open(filename_string.c_str(), SFM_READ, &sf_info);
    if (sndfile == nullptr)
    {
        LOG_ERROR(Settings::Instance().GetLogger(), "Failed to open audio file for reading: {}", sf_strerror(nullptr));
        return {};
    }

    channel = std::min(channel, static_cast<uint32_t>(sf_info.channels - 1));

    std::vector<float> frame(sf_info.channels, 0.0f);

    std::vector<float> audio_data(sf_info.frames, 0.0f);
    for (sf_count_t i = 0; i < sf_info.frames; ++i)
    {
        sf_readf_float(sndfile, frame.data(), 1);
        audio_data[static_cast<size_t>(i)] = frame[channel];
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
    const auto interp_type = static_cast<sfFDN::DelayInterpolationType>(type);
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

sfFDN::AttenuationFilterBankOptions FindAttenuationFilterBankOptions(sfFDN::FDNConfig& config)
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
    for (uint32_t i = 0; i < config.fdn_size; ++i)
    {
        default_config.filter_configs.emplace_back(
            sfFDN::HomogenousFilterOptions{.t60 = 1.f, .delay = -1.f, .sample_rate = config.sample_rate});
    }

    config.loop_filter_configs.emplace_back(default_config);
    return default_config;
}

void ReplaceAttenuationFilterBankOptions(sfFDN::FDNConfig& config,
                                         const sfFDN::AttenuationFilterBankOptions& new_options)
{
    for (auto& loop_filter_config : config.loop_filter_configs)
    {
        if (std::holds_alternative<sfFDN::AttenuationFilterBankOptions>(loop_filter_config))
        {
            loop_filter_config = new_options;
            return;
        }
    }
}

std::vector<float> T60ToGainsDb(T60ToGainsDbInput input)
{
    std::vector<float> gains(input.t60s.size(), 0.0f);
    for (size_t i = 0; i < input.t60s.size(); ++i)
    {
        const auto sample_rate_float = static_cast<float>(input.sample_rate);
        gains[i] = -60.f / (input.t60s[i] * sample_rate_float);
        gains[i] *= static_cast<float>(input.delay);
    }

    return gains;
}

namespace
{
sfFDN::ModulationOptions MakeDefaultGainModulation(uint32_t channel_index, uint32_t channel_count, float sample_rate)
{
    constexpr float kDefaultFrequencyHz = 0.25f;
    constexpr float kDefaultAmplitude = 0.1f;

    return sfFDN::ModulationOptions{
        .frequency = sample_rate > 0.0f ? kDefaultFrequencyHz / sample_rate : 0.0f,
        .amplitude = kDefaultAmplitude,
        .initial_phase =
            channel_count > 0 ? static_cast<float>(channel_index) / static_cast<float>(channel_count) : 0.0f,
    };
}

sfFDN::ModulationOptions MakeDefaultDelayModulation(uint32_t channel_index, uint32_t channel_count, float sample_rate,
                                                    float base_delay)
{
    constexpr float kDefaultFrequencyHz = 0.25f;
    constexpr float kDefaultAmplitude = 10.0f;

    return sfFDN::ModulationOptions{
        .frequency = sample_rate > 0.0f ? kDefaultFrequencyHz / sample_rate : 0.0f,
        .amplitude = std::min(kDefaultAmplitude, std::max(0.0f, base_delay - 1.0f)),
        .initial_phase =
            channel_count > 0 ? static_cast<float>(channel_index) / static_cast<float>(channel_count) : 0.0f,
    };
}

// Schlecht and Habets (2015) recommend roughly 1 Hz per rotation block, an amplitude near 0.7, random initial
// phases, and per-block frequencies spread by about +/-50%. Modulating every block synchronously produces easily
// perceivable beating, so the spread and the phase scatter are load-bearing rather than cosmetic.
constexpr float kMatrixModulationBaseFrequencyHz = 1.0f;
constexpr float kMatrixModulationFrequencySpread = 0.5f;
constexpr float kMatrixModulationAmplitude = 0.7f;
// Above roughly 4 Hz the modulation stops sounding smooth and turns into audible detuning.
constexpr float kMatrixModulationMaximumFrequencyHz = 4.0f;

sfFDN::ModulationOptions ClampMatrixModulation(const sfFDN::ModulationOptions& modulation, float sample_rate)
{
    const float maximum_frequency = sample_rate > 0.0f ? kMatrixModulationMaximumFrequencyHz / sample_rate : 0.0f;

    sfFDN::ModulationOptions normalized{};
    normalized.frequency =
        std::isfinite(modulation.frequency) ? std::clamp(modulation.frequency, 0.0f, maximum_frequency) : 0.0f;
    // sfFDN multiplies the amplitude by pi exactly once, and rejects anything outside [-1, 1].
    normalized.amplitude = std::isfinite(modulation.amplitude) ? std::clamp(modulation.amplitude, -1.0f, 1.0f) : 0.0f;
    normalized.initial_phase =
        std::isfinite(modulation.initial_phase) ? std::clamp(modulation.initial_phase, 0.0f, 1.0f) : 0.0f;
    return normalized;
}

std::vector<sfFDN::ModulationOptions> MakeDefaultMatrixModulation(uint32_t block_count, float sample_rate)
{
    std::vector<sfFDN::ModulationOptions> config(block_count);
    if (block_count == 0)
    {
        return config;
    }

    // A fixed seed keeps a given FDN size reproducible across sessions and saved configurations.
    std::mt19937 generator(0x7A11CE5Du);
    std::uniform_real_distribution<float> phase_distribution(0.0f, 1.0f);

    for (uint32_t block = 0; block < block_count; ++block)
    {
        const float position = block_count > 1 ? static_cast<float>(block) / static_cast<float>(block_count - 1) : 0.5f;
        const float spread = ((2.0f * position) - 1.0f) * kMatrixModulationFrequencySpread;
        const float frequency_hz = kMatrixModulationBaseFrequencyHz * (1.0f + spread);

        config[block] = ClampMatrixModulation(
            sfFDN::ModulationOptions{
                .frequency = sample_rate > 0.0f ? frequency_hz / sample_rate : 0.0f,
                .amplitude = kMatrixModulationAmplitude,
                .initial_phase = phase_distribution(generator),
            },
            sample_rate);
    }

    return config;
}

uint32_t RequiredMaximumDelay(const sfFDN::DelayBankTimeVaryingOptions& config)
{
    constexpr uint32_t kAlignment = 64;
    constexpr uint32_t kSafetyMargin = 64;
    double maximum_delay = 1.0;
    for (const auto& [delay, modulation] : std::views::zip(config.delays, config.time_varying_config))
    {
        maximum_delay = std::max(maximum_delay, static_cast<double>(delay + modulation.amplitude));
    }

    const auto maximum_supported =
        static_cast<double>(std::numeric_limits<uint32_t>::max() - kSafetyMargin - (kAlignment - 1));
    const auto required = static_cast<uint32_t>(std::ceil(std::min(maximum_delay, maximum_supported))) + kSafetyMargin;
    return ((required + kAlignment - 1) / kAlignment) * kAlignment;
}
} // namespace

void ResizeParallelGainsOptions(sfFDN::ParallelGainsOptions& config, ParallelGainsResizeOptions options)
{
    config.gains.resize(options.channel_count, options.new_gain);
    if (config.time_varying_config.empty())
    {
        return;
    }

    const size_t previous_size = config.time_varying_config.size();
    config.time_varying_config.resize(options.channel_count);
    for (size_t index = previous_size; index < options.channel_count; ++index)
    {
        config.time_varying_config[index] =
            MakeDefaultGainModulation(static_cast<uint32_t>(index), options.channel_count, options.sample_rate);
    }
}

void SetTimeVaryingGainsEnabled(sfFDN::ParallelGainsOptions& config, bool enabled, uint32_t channel_count,
                                float sample_rate)
{
    if (!enabled)
    {
        config.time_varying_config.clear();
        return;
    }

    if (!config.time_varying_config.empty())
    {
        ResizeParallelGainsOptions(config,
                                   {.channel_count = channel_count, .sample_rate = sample_rate, .new_gain = 0.5f});
        return;
    }

    config.time_varying_config.reserve(channel_count);
    for (uint32_t index = 0; index < channel_count; ++index)
    {
        config.time_varying_config.push_back(MakeDefaultGainModulation(index, channel_count, sample_rate));
    }
}

bool NormalizeTimeVaryingDelayBank(sfFDN::DelayBankTimeVaryingOptions& config,
                                   TimeVaryingDelayBankNormalizeOptions options)
{
    bool changed = false;
    const float new_delay = std::isfinite(options.new_delay) ? std::max(1.0f, options.new_delay) : 512.0f;

    const size_t previous_delay_count = config.delays.size();
    if (previous_delay_count != options.channel_count)
    {
        config.delays.resize(options.channel_count, new_delay);
        changed = true;
    }

    for (float& delay : config.delays)
    {
        const float normalized = std::isfinite(delay) ? std::max(1.0f, delay) : new_delay;
        changed |= normalized != delay;
        delay = normalized;
    }

    const size_t previous_modulation_count = config.time_varying_config.size();
    if (previous_modulation_count != options.channel_count)
    {
        config.time_varying_config.resize(options.channel_count);
        changed = true;
    }

    for (size_t index = 0; index < config.time_varying_config.size(); ++index)
    {
        auto& modulation = config.time_varying_config[index];
        if (index >= previous_modulation_count)
        {
            modulation = MakeDefaultDelayModulation(static_cast<uint32_t>(index), options.channel_count,
                                                    options.sample_rate, config.delays[index]);
            continue;
        }

        const float maximum_frequency = options.sample_rate > 0.0f ? 20.0f / options.sample_rate : 0.0f;
        const float normalized_frequency =
            std::isfinite(modulation.frequency) ? std::clamp(modulation.frequency, 0.0f, maximum_frequency) : 0.0f;
        const float normalized_amplitude =
            std::isfinite(modulation.amplitude)
                ? std::clamp(modulation.amplitude, 0.0f, std::max(0.0f, config.delays[index] - 1.0f))
                : 0.0f;
        const float normalized_phase =
            std::isfinite(modulation.initial_phase) ? std::clamp(modulation.initial_phase, 0.0f, 1.0f) : 0.0f;

        changed |= normalized_frequency != modulation.frequency;
        changed |= normalized_amplitude != modulation.amplitude;
        changed |= normalized_phase != modulation.initial_phase;
        modulation.frequency = normalized_frequency;
        modulation.amplitude = normalized_amplitude;
        modulation.initial_phase = normalized_phase;
    }

    if (config.interpolation_type != sfFDN::DelayInterpolationType::Linear &&
        config.interpolation_type != sfFDN::DelayInterpolationType::Allpass &&
        config.interpolation_type != sfFDN::DelayInterpolationType::Lagrange)
    {
        config.interpolation_type = sfFDN::DelayInterpolationType::Linear;
        changed = true;
    }

    const uint32_t required_maximum_delay = RequiredMaximumDelay(config);
    if (config.max_delay != required_maximum_delay)
    {
        config.max_delay = required_maximum_delay;
        changed = true;
    }

    return changed;
}

sfFDN::DelayBankTimeVaryingOptions MakeTimeVaryingDelayBank(uint32_t channel_count, float sample_rate, float base_delay)
{
    sfFDN::DelayBankTimeVaryingOptions config{
        .delays = std::vector<float>(channel_count, base_delay),
        .max_delay = 0,
        .interpolation_type = sfFDN::DelayInterpolationType::Linear,
        .time_varying_config = {},
    };
    NormalizeTimeVaryingDelayBank(
        config, {.channel_count = channel_count, .sample_rate = sample_rate, .new_delay = base_delay});
    return config;
}

uint32_t GetTimeVaryingRotationBlockCount(const sfFDN::TimeVaryingFeedbackMatrixOptions& config)
{
    if (config.mode == sfFDN::TimeVaryingMatrixMode::Hadamard)
    {
        return IsPowerOfTwo(config.matrix_size) ? config.matrix_size / 2 : 0;
    }

    if (config.matrix_size < 2 || (config.matrix_size % 2) != 0)
    {
        return 0;
    }

    // RealSchur derives its rotation blocks from a basis built at construction time, so the count cannot be computed
    // from the options alone. Probe with modulation disabled to avoid the per-block validation.
    try
    {
        const sfFDN::TimeVaryingFeedbackMatrix probe(
            sfFDN::TimeVaryingFeedbackMatrixOptions{.matrix_size = config.matrix_size,
                                                    .mode = config.mode,
                                                    .time_varying_config = {},
                                                    .rng_seed = config.rng_seed});
        return probe.RotationBlockCount();
    }
    catch (const std::exception& error)
    {
        LOG_WARNING(Settings::Instance().GetLogger(), "Failed to probe time-varying matrix rotation blocks: {}",
                    error.what());
        return 0;
    }
}

bool NormalizeTimeVaryingMatrixOptions(sfFDN::TimeVaryingFeedbackMatrixOptions& config, uint32_t fdn_size,
                                       float sample_rate, std::optional<uint32_t> known_block_count)
{
    bool changed = false;

    if (config.matrix_size != fdn_size)
    {
        config.matrix_size = fdn_size;
        changed = true;
        known_block_count.reset();
    }

    // Hadamard construction is only defined for power-of-two orders; RealSchur accepts any even order.
    if (config.mode == sfFDN::TimeVaryingMatrixMode::Hadamard && !IsPowerOfTwo(config.matrix_size))
    {
        config.mode = sfFDN::TimeVaryingMatrixMode::RealSchur;
        changed = true;
        known_block_count.reset();
    }

    const uint32_t block_count =
        known_block_count.has_value() ? *known_block_count : GetTimeVaryingRotationBlockCount(config);
    if (block_count == 0)
    {
        changed |= !config.time_varying_config.empty();
        config.time_varying_config.clear();
        return changed;
    }

    // An empty config is the library's "modulation off" state, so leave it alone rather than switching it on.
    if (config.time_varying_config.empty())
    {
        return changed;
    }

    const size_t previous_count = config.time_varying_config.size();
    if (previous_count != block_count)
    {
        config.time_varying_config.resize(block_count);
        changed = true;
    }

    const auto defaults = MakeDefaultMatrixModulation(block_count, sample_rate);
    for (size_t index = 0; index < config.time_varying_config.size(); ++index)
    {
        auto& modulation = config.time_varying_config[index];
        if (index >= previous_count)
        {
            modulation = defaults[index];
            continue;
        }

        const sfFDN::ModulationOptions normalized = ClampMatrixModulation(modulation, sample_rate);
        changed |= normalized.frequency != modulation.frequency || normalized.amplitude != modulation.amplitude ||
                   normalized.initial_phase != modulation.initial_phase;
        modulation = normalized;
    }

    return changed;
}

sfFDN::TimeVaryingFeedbackMatrixOptions MakeTimeVaryingFeedbackMatrix(uint32_t fdn_size, float sample_rate)
{
    sfFDN::TimeVaryingFeedbackMatrixOptions config{
        .matrix_size = fdn_size,
        .mode =
            IsPowerOfTwo(fdn_size) ? sfFDN::TimeVaryingMatrixMode::Hadamard : sfFDN::TimeVaryingMatrixMode::RealSchur,
        .time_varying_config = {},
        .rng_seed = 0,
    };

    const uint32_t block_count = GetTimeVaryingRotationBlockCount(config);
    config.time_varying_config = MakeDefaultMatrixModulation(block_count, sample_rate);
    return config;
}

std::string GetFeedbackMatrixName(const sfFDN::feedback_matrix_variant_t& matrix_variant)
{
    return std::visit(
        overloaded{
            [](const sfFDN::CascadedFeedbackMatrixOptions& config) { return "Cascaded " + GetMatrixName(config.type); },
            [](const sfFDN::ScalarFeedbackMatrixOptions& config) { return GetMatrixName(config.type); },
            [](const sfFDN::TimeVaryingFeedbackMatrixOptions& config) {
                return std::string("Time-Varying (") +
                       (config.mode == sfFDN::TimeVaryingMatrixMode::Hadamard ? "Hadamard" : "RealSchur") + ")";
            },
        },
        matrix_variant);
}

bool NormalizeGraphicEq(sfFDN::GraphicEQOptions& config, float sample_rate)
{
    static constexpr std::array<float, 10> kDefaultFrequencies = {31.25f,  62.5f,   125.0f,  250.0f,  500.0f,
                                                                  1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f};

    bool changed = false;
    const float nyquist_frequency = sample_rate * 0.5f;

    for (size_t index = 0; index < config.freqs.size(); ++index)
    {
        const float default_frequency = std::min(kDefaultFrequencies.at(index), nyquist_frequency * 0.99f);
        float frequency = config.freqs.at(index);
        if (!std::isfinite(frequency) || frequency <= 0.0f || frequency >= nyquist_frequency)
        {
            frequency = default_frequency;
        }
        changed |= frequency != config.freqs.at(index);
        config.freqs.at(index) = frequency;
    }

    for (float& gain_db : config.gains_db)
    {
        const float normalized =
            std::isfinite(gain_db) ? std::clamp(gain_db, kGraphicEqMinimumGainDb, kGraphicEqMaximumGainDb) : 0.0f;
        changed |= normalized != gain_db;
        gain_db = normalized;
    }

    if (config.sample_rate != sample_rate)
    {
        config.sample_rate = sample_rate;
        changed = true;
    }

    return changed;
}

sfFDN::GraphicEQOptions MakeGraphicEq(float sample_rate)
{
    // Value-initialization zeroes the band arrays; normalization then fills in the default octave
    // band frequencies and a flat response.
    sfFDN::GraphicEQOptions config{};
    NormalizeGraphicEq(config, sample_rate);
    return config;
}

namespace
{
void ResizeMultichannelProcessorConfigs(sfFDN::multi_channel_processor_variant_t& config_variant, uint32_t new_size,
                                        float sample_rate)
{
    std::visit(
        overloaded{
            [new_size, sample_rate](sfFDN::ParallelGainsOptions& config) {
                ResizeParallelGainsOptions(config,
                                           {.channel_count = new_size, .sample_rate = sample_rate, .new_gain = 0.5f});
            },
            [new_size](sfFDN::MultichannelSchroederAllpassSectionOptions& config) { config.sections.resize(new_size); },
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
            [new_size, sample_rate](sfFDN::DelayBankTimeVaryingOptions& config) {
                NormalizeTimeVaryingDelayBank(
                    config, {.channel_count = new_size, .sample_rate = sample_rate, .new_delay = 512.0f});
            },
            [new_size](sfFDN::CascadedFeedbackMatrixOptions& config) { config.matrix_size = new_size; },
            [new_size](sfFDN::ScalarFeedbackMatrixOptions& config) {
                config.matrix_size = new_size;
                if (config.custom_matrix.has_value())
                {
                    const size_t matrix_size = static_cast<size_t>(new_size) * static_cast<size_t>(new_size);
                    config.custom_matrix->resize(matrix_size, 0.f);
                }
            },
            [new_size](sfFDN::MultichannelFirOptions& config) {
                auto previous_size = config.coeffs.size();
                config.coeffs.resize(new_size);

                auto last_config = config.coeffs.back();
                for (size_t i = previous_size; i < new_size; ++i)
                {
                    config.coeffs[i] = last_config;
                }
            },
        },
        config_variant);
}
} // namespace

bool NormalizeAttenuationFilterBank(sfFDN::FDNConfig& config)
{
    if (!config.attenuation_filter_bank_config.has_value())
    {
        return false;
    }

    auto& filter_configs = config.attenuation_filter_bank_config->filter_configs;
    const bool changed = filter_configs.size() != config.fdn_size;

    if (filter_configs.empty())
    {
        filter_configs.emplace_back(sfFDN::HomogenousFilterOptions{});
    }

    const auto fill_filter = filter_configs.back();
    filter_configs.resize(config.fdn_size, fill_filter);

    for (size_t i = 0; i < filter_configs.size(); ++i)
    {
        const float delay = i < config.delay_bank_config.delays.size() ? config.delay_bank_config.delays[i] : -1.0f;
        std::visit(
            [delay, sample_rate = config.sample_rate](auto& filter) {
                filter.delay = delay;
                filter.sample_rate = sample_rate;
            },
            filter_configs[i]);
    }

    return changed;
}

bool IsAttenuationFilterBankHomogeneous(const sfFDN::AttenuationFilterBankOptions& filter_bank)
{
    const auto& filter_configs = filter_bank.filter_configs;
    if (filter_configs.size() < 2)
    {
        return true;
    }

    // sfFDN's option structs have no equality operators, and a member-wise comparison would be
    // wrong anyway: `delay` is assigned per channel from the delay bank, so comparing whole structs
    // would report every bank as heterogeneous. Only the decay-shaping fields are compared here.
    const auto same_response = [](const sfFDN::attenuation_filter_variant_t& lhs,
                                  const sfFDN::attenuation_filter_variant_t& rhs) {
        if (lhs.index() != rhs.index())
        {
            return false;
        }
        return std::visit(
            [&rhs](const auto& left) {
                using Filter = std::decay_t<decltype(left)>;
                const auto& right = std::get<Filter>(rhs);
                if constexpr (std::is_same_v<Filter, sfFDN::HomogenousFilterOptions>)
                {
                    return left.t60 == right.t60;
                }
                else if constexpr (std::is_same_v<Filter, sfFDN::TwoBandFilterOptions>)
                {
                    return left.t60s == right.t60s;
                }
                else if constexpr (std::is_same_v<Filter, sfFDN::ThreeBandFilterOptions>)
                {
                    return left.t60s == right.t60s && left.freqs == right.freqs && left.q == right.q;
                }
                else
                {
                    return left.t60s == right.t60s && left.shelf_cutoff == right.shelf_cutoff;
                }
            },
            lhs);
    };

    return std::ranges::all_of(filter_configs, [&](const sfFDN::attenuation_filter_variant_t& filter) {
        return same_response(filter_configs.front(), filter);
    });
}

void ResizeFDNConfig(sfFDN::FDNConfig& config, uint32_t new_size)
{
    config.fdn_size = new_size;

    config.delay_bank_config.delays.resize(new_size, 512.f);

    ResizeParallelGainsOptions(config.input_block_config.parallel_gains_config,
                               {.channel_count = new_size, .sample_rate = config.sample_rate, .new_gain = 0.5f});

    for (auto& processor_variant : config.input_block_config.multichannel_processors)
    {
        ResizeMultichannelProcessorConfigs(processor_variant, new_size, config.sample_rate);
    }

    ResizeParallelGainsOptions(config.output_block_config.parallel_gains_config,
                               {.channel_count = new_size, .sample_rate = config.sample_rate, .new_gain = 0.5f});
    for (auto& processor_variant : config.output_block_config.multichannel_processors)
    {
        ResizeMultichannelProcessorConfigs(processor_variant, new_size, config.sample_rate);
    }

    for (auto& processor_variant : config.loop_filter_configs)
    {
        ResizeMultichannelProcessorConfigs(processor_variant, new_size, config.sample_rate);
    }

    const auto demote_hadamard = [new_size](sfFDN::ScalarMatrixType& type) {
        if (type == sfFDN::ScalarMatrixType::Hadamard && !IsPowerOfTwo(new_size))
        {
            type = sfFDN::ScalarMatrixType::Random;
        }
    };

    std::visit(overloaded{
                   [&](sfFDN::CascadedFeedbackMatrixOptions& matrix_config) {
                       matrix_config.matrix_size = new_size;
                       demote_hadamard(matrix_config.type);
                   },
                   [&](sfFDN::ScalarFeedbackMatrixOptions& matrix_config) {
                       matrix_config.matrix_size = new_size;
                       demote_hadamard(matrix_config.type);
                       if (matrix_config.custom_matrix.has_value())
                       {
                           matrix_config.custom_matrix = sfFDN::GenerateMatrix(new_size, matrix_config.type);
                       }
                   },
                   [&](sfFDN::TimeVaryingFeedbackMatrixOptions& matrix_config) {
                       // A time-varying matrix cannot be built at an odd order at all: an odd-dimensional orthogonal
                       // matrix always has a static real eigenvalue, so sfFDN rejects it. Fall back to a scalar
                       // matrix rather than leaving behind a configuration that throws at build time.
                       if (new_size < 2 || (new_size % 2) != 0)
                       {
                           sfFDN::ScalarFeedbackMatrixOptions replacement{.matrix_size = new_size};
                           demote_hadamard(replacement.type);
                           config.feedback_matrix_config = replacement;
                           return;
                       }
                       NormalizeTimeVaryingMatrixOptions(matrix_config, new_size, config.sample_rate);
                   },
               },
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
            [](const sfFDN::MultichannelSchroederAllpassSectionOptions&) { return "Parallel Schroeder Allpass"; },
            [](const sfFDN::AttenuationFilterBankOptions&) { return "Attenuation Filter Bank"; },
            [](const sfFDN::DelayBankOptions&) { return "Delay Bank"; },
            [](const sfFDN::DelayBankTimeVaryingOptions&) { return "Time-Varying Delay Bank"; },
            [](const sfFDN::CascadedFeedbackMatrixOptions&) { return "Cascaded Feedback Matrix"; },
            [](const sfFDN::ScalarFeedbackMatrixOptions&) { return "Scalar Feedback Matrix"; },
            [](const sfFDN::MultichannelFirOptions&) { return "Multichannel FIR Filter"; },
        },
        processor_variant);
}

template std::vector<double> LogSpace(double start, double stop, size_t num);
template std::vector<float> LogSpace(float start, float stop, size_t num);
} // namespace utils