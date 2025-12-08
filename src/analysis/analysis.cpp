#include "analysis.h"

#include <Eigen/Core>
#include <boost/math/statistics/linear_regression.hpp>
#include <sndfile.h>

#include <algorithm>
#include <limits>
#include <numbers>

#include "audio_utils/fft.h"
#include "octave_band_coeff.h"
#include "octave_band_filters_fir.h"
#include <audio_utils/fft_utils.h>

#include <sffdn/sffdn.h>

namespace fdn_analysis
{
std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db)
{
    if (signal.empty())
    {
        return {};
    }

    // Calculate the energy decay curve
    std::vector<float> decay_curve(signal.size());
    float cumulative_energy = 0.0f;

    for (int i = signal.size() - 1; i >= 0; --i)
    {
        cumulative_energy += signal[i] * signal[i];
        decay_curve[i] = cumulative_energy;
    }

    // Normalize energy
    float max_energy = *std::ranges::max_element(decay_curve);
    for (auto& energy : decay_curve)
    {
        if (max_energy != 0.0f)
        {
            energy /= max_energy;
        }
    }

    if (to_db)
    {
        for (auto& energy : decay_curve)
        {
            energy = 10.0f * std::log10(energy + 1e-10f); // Add small value to avoid log(0)
        }
    }
    return decay_curve;
}

std::array<float, 10> GetOctaveBandFrequencies()
{
    return kOctaveBandFrequencies;
}

EstimateT60Results EstimateT60(std::span<const float> decay_curve, std::span<const float> time, float decay_start_db,
                               float decay_end_db)
{
    if (decay_start_db <= decay_end_db)
    {
        throw std::invalid_argument("decay_start_db must be less than decay_end_db");
    }

    if (decay_curve.size() != time.size())
    {
        throw std::invalid_argument("decay_curve and time must have the same size");
    }

    auto it_start = std::ranges::lower_bound(decay_curve, decay_start_db, [](float value, float threshold) {
        return std::abs(value) < std::abs(threshold);
    });
    auto it_end = std::ranges::lower_bound(
        decay_curve, decay_end_db, [](float value, float threshold) { return std::abs(value) < std::abs(threshold); });

    auto start_index = std::distance(decay_curve.begin(), it_start);
    auto end_index = std::distance(decay_curve.begin(), it_end);

    auto decay_span = std::span(decay_curve).subspan(start_index, end_index - start_index);
    auto time_span = std::span(time).subspan(start_index, end_index - start_index);

    if (time_span.empty() || decay_span.empty())
    {
        // This can happen if the decay curve is not within the specified dB range
        return {.t60 = 0.0f, .decay_start_time = 0.0f, .decay_end_time = 0.0f, .intercept = 0.0f, .slope = 0.0f};
    }

    auto [c0, c1] = boost::math::statistics::simple_ordinary_least_squares(time_span, decay_span);

    EstimateT60Results results;
    results.t60 = -60.0f / c1;
    results.decay_start_time = time_span.front();
    results.decay_end_time = time_span.back();
    results.intercept = c0;
    results.slope = c1;

    return results;
}

std::array<std::vector<float>, 10> EnergyDecayRelief(std::span<const float> signal, bool to_db)
{
    std::array<std::vector<float>, 10> edc_octaves;

#pragma omp parallel for firstprivate(signal, to_db)
    for (auto i = 0; i < kOctaveBandFirFilters.size(); ++i)
    {
        const auto& fir_coeffs = kOctaveBandFirFilters[i];

        auto conv_size = signal.size() + fir_coeffs.size() - 1;
        conv_size = audio_utils::FFT::NextSupportedFFTSize(static_cast<uint32_t>(conv_size));
        audio_utils::FFT fft(conv_size);

        std::vector<float> result(conv_size, 0.0f);
        fft.Convolve(signal, fir_coeffs, result);

        // The filters introduce a delay of roughly half the filter length
        const size_t delay = (fir_coeffs.size() / 2) * 0.9;
        std::span<float> result_span = std::span(result).subspan(delay, signal.size());
        edc_octaves[i] = EnergyDecayCurve(result_span, to_db);
    }

    return edc_octaves;
}

EchoDensityResults EchoDensity(std::span<const float> signal, uint32_t window_size, uint32_t sample_rate,
                               uint32_t hop_size)
{
    if (signal.empty() || window_size == 0 || sample_rate == 0)
    {
        return {};
    }

    EchoDensityResults results;

    std::vector<float> win(window_size, 0.0f);
    GetWindow(audio_utils::FFTWindowType::Hann, win);
    float win_sum = std::accumulate(win.begin(), win.end(), 0.0f);
    for (auto& w : win)
    {
        w /= win_sum;
    }

    const int half_win = window_size / 2;
    results.echo_densities.reserve((signal.size() + hop_size - 1) / hop_size);
    results.sparse_indices.reserve((signal.size() + hop_size - 1) / hop_size);

    results.mixing_time = std::numeric_limits<float>::infinity();

    for (int n = 0; n < signal.size(); n += hop_size)
    {
        std::span<const float> hTau;
        std::span<const float> wT;

        if (n < half_win)
        {
            hTau = signal.subspan(0, n + half_win);
            wT = std::span(win).subspan(win.size() - n - half_win, n + half_win);
        }
        else if (n > signal.size() - half_win)
        {
            hTau = signal.subspan(n - half_win, signal.size() - n + half_win);
            wT = std::span(win).subspan(0, signal.size() - n + half_win);
        }
        else
        {
            hTau = signal.subspan(n - half_win, window_size);
            wT = win;
        }

        assert(hTau.size() == wT.size());

        Eigen::Map<const Eigen::ArrayXf> hTau_map(hTau.data(), hTau.size());
        Eigen::Map<const Eigen::ArrayXf> wT_map(wT.data(), wT.size());

        float std = std::sqrt((hTau_map.square() * wT_map).sum());

        // Use Eigen for vectorized computation of echo_density
        Eigen::ArrayXf abs_hTau = hTau_map.abs();
        Eigen::ArrayXf mask = (abs_hTau > std).cast<float>();
        float echo_density = (mask * wT_map).sum();

        // normalize
        const float kErfc = std::erfc(1.0f / std::numbers::sqrt2_v<float>);
        echo_density /= kErfc;

        results.sparse_indices.push_back(n);
        results.echo_densities.push_back(echo_density);

        // Estimate mixing time as the time when echo density first exceeds 0.9
        if (results.mixing_time == std::numeric_limits<float>::infinity() && echo_density >= 0.9f)
        {
            results.mixing_time = static_cast<float>(n) / static_cast<float>(sample_rate);
        }
    }

    return results;
}

} // namespace fdn_analysis