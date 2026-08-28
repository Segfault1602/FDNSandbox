#pragma once

#include <optional>
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

/// Reports how many independently modulatable 2x2 rotation blocks a time-varying feedback matrix has.
///
/// Hadamard mode always has `matrix_size / 2`. RealSchur mode depends on the basis generated from `rng_seed` and is
/// only knowable after construction, so this probe-constructs a matrix and is far too expensive to call per frame.
/// Returns 0 when the options are invalid.
uint32_t GetTimeVaryingRotationBlockCount(const sfFDN::TimeVaryingFeedbackMatrixOptions& config);

/// Brings a time-varying feedback matrix in line with `fdn_size` and its own rotation block count.
///
/// Sets `matrix_size` to `fdn_size`, demotes Hadamard to RealSchur when `fdn_size` is not a power of two, and grows
/// or shrinks `time_varying_config` to one entry per rotation block. Returns whether anything changed.
///
/// Pass `known_block_count` from a cache on per-frame paths: without it this probe-constructs a matrix, which costs
/// a Schur decomposition in RealSchur mode. The hint is ignored if normalization had to change `matrix_size` or
/// `mode`, since that would invalidate it.
bool NormalizeTimeVaryingMatrixOptions(sfFDN::TimeVaryingFeedbackMatrixOptions& config, uint32_t fdn_size,
                                       float sample_rate, std::optional<uint32_t> known_block_count = std::nullopt);

/// Builds a time-varying feedback matrix using the modulation settings recommended by Schlecht and Habets (2015).
///
/// Roughly 1 Hz per block spread over +/-50%, normalized amplitude 0.7 (that is, a peak deviation of 0.7 pi rad),
/// and deterministic pseudo-random initial phases. The spread and phase scatter are deliberate: the papers report
/// easily perceivable beating when every block modulates synchronously.
sfFDN::TimeVaryingFeedbackMatrixOptions MakeTimeVaryingFeedbackMatrix(uint32_t fdn_size, float sample_rate);

/// Returns a display name for any feedback matrix option, including the time-varying modes.
std::string GetFeedbackMatrixName(const sfFDN::feedback_matrix_variant_t& matrix_variant);

inline constexpr float kGraphicEqMinimumGainDb = -30.0f;
inline constexpr float kGraphicEqMaximumGainDb = 30.0f;

bool NormalizeGraphicEq(sfFDN::GraphicEQOptions& config, float sample_rate);
sfFDN::GraphicEQOptions MakeGraphicEq(float sample_rate);

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

/// Reports whether every channel in the bank shares the same attenuation response.
///
/// Only the decay-shaping parameters are compared. Each channel's `delay` is set from the delay
/// bank and so legitimately differs, and `sample_rate` is uniform by construction; including either
/// would report every bank as heterogeneous.
bool IsAttenuationFilterBankHomogeneous(const sfFDN::AttenuationFilterBankOptions& filter_bank);

std::string GetProcessorName(const sfFDN::single_channel_processor_variant_t& processor_variant);
std::string GetProcessorName(const sfFDN::multi_channel_processor_variant_t& processor_variant);

} // namespace utils