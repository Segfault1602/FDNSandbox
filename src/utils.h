#pragma once

#include <span>
#include <string>
#include <vector>

#include <sffdn/sffdn.h>

namespace utils
{

struct PchipInput
{
    std::span<const float> x;
    std::span<const float> y;
    std::span<const float> xq;
};

struct T60ToGainsDbInput
{
    std::span<const float> t60s;
    uint32_t delay;
    size_t sample_rate;
};

/**
 * @brief Check if a number is a power of two.
 *
 * @param n The number to check.
 * @return true if the number is a power of two, false otherwise.
 */
bool IsPowerOfTwo(size_t n);

/**
 * @brief Get the closest prime number to a given number.
 *
 * @param n The number to find the closest prime for.
 * @return The closest prime number.
 */
uint32_t GetClosestPrime(uint32_t n);

/**
 * @brief Generate a logarithmically spaced vector.
 *
 * @tparam T The type of the elements in the vector.
 * @param start The starting value of the range.
 * @param stop The ending value of the range.
 * @param num The number of points to generate.
 * @return A vector containing logarithmically spaced values.
 */
template <typename T>
std::vector<T> LogSpace(T start, T stop, size_t num);

/**
 * @brief Perform piecewise cubic Hermite interpolation.
 *
 * @param x The x-coordinates of the data points.
 * @param y The y-coordinates of the data points.
 * @param xq The x-coordinates of the query points.
 * @return A vector containing the interpolated values.
 */
std::vector<float> pchip(PchipInput input);

/**
 * @brief Compute the absolute value of the frequency response.
 *
 * @param sos The second-order sections coefficients.
 * @param w The frequencies at which to evaluate the response.
 * @param sr The sample rate.
 * @return A vector containing the absolute frequency response.
 */
std::vector<float> AbsFreqz(std::span<const sfFDN::FilterCoefficients> sos, std::span<const float> w, size_t sr);

uint32_t GetChannelCountFromAudioFile(std::string_view filename);

std::vector<float> ReadAudioFile(std::string_view filename, uint32_t channel);

std::string GetMatrixName(sfFDN::ScalarMatrixType type);

std::string GetDelayLengthTypeName(int type);

std::string GetDelayInterpolationTypeName(int type);

sfFDN::AttenuationFilterBankOptions FindAttenuationFilterBankOptions(sfFDN::FDNConfig& config);

void ReplaceAttenuationFilterBankOptions(sfFDN::FDNConfig& config,
                                         const sfFDN::AttenuationFilterBankOptions& new_options);

std::vector<float> T60ToGainsDb(T60ToGainsDbInput input);

struct ParallelGainsResizeOptions
{
    uint32_t channel_count;
    float sample_rate;
    float new_gain;
};

void ResizeParallelGainsOptions(sfFDN::ParallelGainsOptions& config, ParallelGainsResizeOptions options);
void SetTimeVaryingGainsEnabled(sfFDN::ParallelGainsOptions& config, bool enabled, uint32_t channel_count,
                                float sample_rate);

struct TimeVaryingDelayBankNormalizeOptions
{
    uint32_t channel_count;
    float sample_rate;
    float new_delay = 512.0f;
};

bool NormalizeTimeVaryingDelayBank(sfFDN::DelayBankTimeVaryingOptions& config,
                                   TimeVaryingDelayBankNormalizeOptions options);
sfFDN::DelayBankTimeVaryingOptions MakeTimeVaryingDelayBank(uint32_t channel_count, float sample_rate,
                                                            float base_delay = 512.0f);

struct Span2D
{
    std::span<float> data;
    size_t rows;
    size_t cols;

    float& operator()(size_t i, size_t j) const
    {
        return data[(i * cols) + j];
    }
};

void ResizeFDNConfig(sfFDN::FDNConfig& config, uint32_t new_size);
bool NormalizeAttenuationFilterBank(sfFDN::FDNConfig& config);

std::string GetProcessorName(const sfFDN::single_channel_processor_variant_t& processor_variant);
std::string GetProcessorName(const sfFDN::multi_channel_processor_variant_t& processor_variant);

} // namespace utils