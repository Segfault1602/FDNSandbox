#pragma once

#include <span>
#include <string>
#include <vector>

#include <sffdn/sffdn.h>

namespace utils
{

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
 * @brief Generate a linearly spaced vector.
 *
 * @tparam T The type of the elements in the vector.
 * @param start The starting value of the range.
 * @param stop The ending value of the range.
 * @param num The number of points to generate.
 * @return A vector containing linearly spaced values.
 */
template <typename T>
std::vector<T> Linspace(T start, T stop, size_t num);

/**
 * @brief Perform piecewise cubic Hermite interpolation.
 *
 * @param x The x-coordinates of the data points.
 * @param y The y-coordinates of the data points.
 * @param xq The x-coordinates of the query points.
 * @return A vector containing the interpolated values.
 */
std::vector<float> pchip(std::span<const float> x, std::span<const float> y, std::span<const float> xq);

/**
 * @brief Compute the absolute value of the frequency response.
 *
 * @param sos The second-order sections coefficients.
 * @param w The frequencies at which to evaluate the response.
 * @param sr The sample rate.
 * @return A vector containing the absolute frequency response.
 */
std::vector<float> AbsFreqz(std::span<const float> sos, std::span<const float> w, size_t sr);

/**
 * @brief Write audio samples to a file.
 *
 * @param filename The path to the audio file.
 * @param audio_data The audio samples to write.
 * @param sample_rate The sample rate of the audio.
 */
void WriteAudioFile(const std::string& filename, std::span<const float> audio_data, int sample_rate);

std::array<std::array<float, 6>, 10> GetOctaveBandsSOS();

std::string GetMatrixName(sfFDN::ScalarMatrixType type);

std::string GetDelayLengthTypeName(int type);

std::vector<float> T60ToGainsDb(std::span<const float> t60s, uint32_t delay, size_t sample_rate);

} // namespace utils