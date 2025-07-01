#pragma once

#include <span>
#include <vector>

#include <matrix_gallery.h>

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
 * @brief Perform piecewise cubic Hermite interpolation.
 *
 * @param x The x-coordinates of the data points.
 * @param y The y-coordinates of the data points.
 * @param xq The x-coordinates of the query points.
 * @return A vector containing the interpolated values.
 */
std::vector<float> pchip(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& xq);

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
 * @brief Read an audio file and return its samples.
 *
 * @param filename The path to the audio file.
 * @return A vector containing the audio samples.
 */
std::vector<float> ReadAudioFile(const std::string& filename);

/**
 * @brief Write audio samples to a file.
 *
 * @param filename The path to the audio file.
 * @param audio_data The audio samples to write.
 * @param sample_rate The sample rate of the audio.
 */
void WriteAudioFile(const std::string& filename, std::span<const float> audio_data, int sample_rate);

/**
 * @brief Compute the energy decay curve of a signal.
 *
 * @param signal The input signal.
 * @param to_db If true, convert the energy values to decibels.
 * @return std::vector<float> The short-time energy of the signal.
 */
std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db = false);

std::array<std::array<float, 6>, 10> GetOctaveBandsSOS();

std::string GetMatrixName(sfFDN::ScalarMatrixType type);

std::vector<float> T60ToGainsDb(std::span<const float> t60s, size_t sample_rate);

} // namespace utils