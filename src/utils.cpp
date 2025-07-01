#include "utils.h"

#include <Eigen/Core>
#include <boost/math/interpolators/pchip.hpp>
#include <sndfile.h>

#include <complex>
#include <mdspan>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

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
        return false;
    for (uint32_t i = 2; i * i <= n; ++i)
    {
        if (n % i == 0)
            return false;
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
        return result;

    T step = (stop - start) / (num - 1);

    Eigen::Map<Eigen::ArrayX<T>> result_map(result.data(), num);

    result_map = Eigen::ArrayX<T>::LinSpaced(num, start, stop);
    result_map = Eigen::pow(10, result_map);

    return result;
}

std::vector<float> pchip(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& xq)
{
    auto x_copy = x;
    auto y_copy = y;
    auto spline = boost::math::interpolators::pchip(std::move(x_copy), std::move(y_copy));

    std::vector<float> yq;
    yq.reserve(xq.size());
    for (size_t i = 0; i < xq.size(); ++i)
    {
        yq.push_back(spline(xq[i]));
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
        Eigen::Map<const Eigen::ArrayXf> b_map(sos.data() + i * 6, 3);
        Eigen::Map<const Eigen::ArrayXf> a_map(sos.data() + i * 6 + 3, 3);
        Eigen::ArrayXcf h = Polyval(b_map, dig_w) / Polyval(a_map, dig_w);

        h_complex = h_complex * h;
    }

    std::vector<float> h(w.size(), 0.0f);
    Eigen::Map<Eigen::ArrayXf> h_map(h.data(), h.size());
    h_map = h_complex.abs();

    return h;
}

std::vector<float> ReadAudioFile(const std::string& filename)
{
    // Preload the drum loop
    SF_INFO sf_info{0};
    SNDFILE* sndfile = sf_open(filename.c_str(), SFM_READ, &sf_info);
    if (!sndfile)
    {
        std::cerr << "Failed to open audio file: " << sf_strerror(nullptr) << std::endl;
        return {};
    }

    if (sf_info.channels != 1)
    {
        std::cerr << "Audio file must be mono." << std::endl;
        sf_close(sndfile);
        return {};
    }

    std::cout << "Audio file format: " << std::hex << sf_info.format << std::dec << std::endl;

    std::vector<float> audio_data(sf_info.frames);
    sf_count_t read_count = sf_readf_float(sndfile, audio_data.data(), sf_info.frames);
    if (read_count != sf_info.frames)
    {
        std::cerr << "Failed to read audio file: " << sf_strerror(sndfile) << std::endl;
        sf_close(sndfile);
        return {};
    }

    sf_close(sndfile);
    return audio_data;
}

void WriteAudioFile(const std::string& filename, std::span<const float> audio_data, int sample_rate)
{
    SF_INFO sf_info{};
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT; // Use WAV format with float samples
    sf_info.samplerate = sample_rate;
    sf_info.channels = 1; // Mono audio

    SNDFILE* sndfile = sf_open(filename.c_str(), SFM_WRITE, &sf_info);
    if (!sndfile)
    {
        std::cerr << "Failed to open audio file for writing: " << sf_strerror(nullptr) << std::endl;
        return;
    }

    sf_count_t write_count = sf_writef_float(sndfile, audio_data.data(), audio_data.size());
    if (write_count != static_cast<sf_count_t>(audio_data.size()))
    {
        std::cerr << "Failed to write audio file: " << sf_strerror(sndfile) << std::endl;
    }

    sf_close(sndfile);
}

std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db)
{
    if (signal.empty())
    {
        return {};
    }

    // Calculate the energy decay curve
    std::vector<float> decay_curve(signal.size());
    float cumulative_energy = 0.0f;

    for (int i = signal.size() - 1; i > 0; --i)
    {
        cumulative_energy += signal[i] * signal[i];
        decay_curve[i] = std::sqrt(cumulative_energy);

        if (to_db)
        {
            decay_curve[i] = 20.0f * std::log10(decay_curve[i]);
        }
    }

    return decay_curve;
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

std::array<std::array<float, 6>, 10> GetOctaveBandsSOS()
{
    constexpr std::array<std::array<float, 6>, 10> sos = {{
        {0.0015030849515687771, 0.0, -0.0015030849515687771, 1.0, -1.9969768922228524, 0.9969938300968622},
        {0.002871531785966629, 0.0, -0.002871531785966629, 1.0, -1.9941885090948224, 0.9942569364280671},
        {0.0057913669578533045, 0.0, -0.0057913669578533045, 1.0, -1.9881473928034248, 0.9884172660842931},
        {0.011452462772237581, 0.0, -0.011452462772237581, 1.0, -1.9760247796495045, 0.9770950744555245},
        {0.022585992641400612, 0.0, -0.022585992641400612, 1.0, -1.950619403188065, 0.954828014717199},
        {0.044136899380977195, 0.0, -0.044136899380977195, 1.0, -1.8953529021139157, 0.9117262012380455},
        {0.08443111970068386, 0.0, -0.08443111970068386, 1.0, -1.7688495350676474, 0.8311377605986322},
        {0.15660038007809052, 0.0, -0.15660038007809052, 1.0, -1.4604043277147636, 0.686799239843819},
        {0.2772680112435692, 0.0, -0.2772680112435692, 1.0, -0.6989762067733558, 0.4454639775128615},
        {0.5255476381877647, -0.5255476381877647, 0.0, 1.0, -0.051095276375529505, 0.0},
    }};

    return sos;
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
    default:
        return "Unknown";
    }
}

std::vector<float> T60ToGainsDb(std::span<const float> t60s, size_t sample_rate)
{
    std::vector<float> gains(t60s.size(), 0.0f);
    for (size_t i = 0; i < t60s.size(); ++i)
    {
        gains[i] = std::pow(10.0, -3.0 / t60s[i]);
        gains[i] = std::pow(gains[i], 1000.f / sample_rate);
        gains[i] = 20.f * std::log10(gains[i]);
    }

    return gains;
}

template std::vector<double> LogSpace(double start, double stop, size_t num);
template std::vector<float> LogSpace(float start, float stop, size_t num);
} // namespace utils