#include "analysis.h"

#include <Eigen/Core>
#include <boost/math/statistics/linear_regression.hpp>
#include <sndfile.h>

#include "audio_utils/fft.h"
#include "octave_band_coeff.h"
#include "octave_band_filters_fir.h"
#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft_utils.h>

#include <sffdn/sffdn.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numbers>

namespace fdn_analysis
{
std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db, bool normalize)
{
    if (signal.empty())
    {
        return {};
    }

    // Discard silence at the beginning of impulse response
    const float max_val = *std::ranges::max_element(signal, [](float a, float b) { return std::abs(a) < std::abs(b); });
    constexpr float kDirectImpulseThreshold = 0.5f;
    auto it_start = std::ranges::find_if(signal, [threshold = kDirectImpulseThreshold * std::abs(max_val)](
                                                     float sample) { return std::abs(sample) >= threshold; });

    std::span<const float> trimmed_signal;
    if (it_start != signal.end())
    {
        trimmed_signal = std::span(it_start, signal.end());
    }
    else
    {
        trimmed_signal = signal;
    }

    std::ranges::reverse_view trimmed_signal_reversed{trimmed_signal};
    auto s = trimmed_signal_reversed | std::views::transform([](float v) { return v * v; });

    // Calculate the energy decay curve
    std::vector<float> decay_curve(signal.size(), 0.0f);
    std::ranges::reverse_view decay_curve_reversed{std::span(decay_curve).subspan(0, trimmed_signal.size())};

    std::partial_sum(s.begin(), s.end(), decay_curve_reversed.begin());

    // Normalize energy
    if (normalize)
    {
        float max_energy = *std::ranges::max_element(decay_curve);
        if (max_energy != 0.0f)
        {
            for (auto& energy : decay_curve)
            {
                energy /= max_energy;
            }
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

    float start_db_value = decay_curve[0];
    decay_start_db += start_db_value;
    decay_end_db += start_db_value;

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

std::array<std::vector<float>, 10> EnergyDecayRelief(std::span<const float> signal, bool to_db, bool normalize)
{
    std::array<std::vector<float>, 10> edc_octaves;

    std::vector<float> nc_signal;
    std::ranges::copy(signal, std::back_inserter(nc_signal));

#pragma omp parallel for
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
        edc_octaves[i] = EnergyDecayCurve(result_span, to_db, normalize);
    }

    return edc_octaves;
}

EnergyDecayReliefResult EnergyDecayReliefSTFT(std::span<const float> signal, const EnergyDecayReliefOptions& options)
{
    if (signal.empty())
    {
        return {};
    }

    if (options.fft_length < options.window_size)
    {
        throw std::invalid_argument("FFT length must be greater than or equal to window size");
    }

    if (options.hop_size > options.window_size)
    {
        throw std::invalid_argument("Hop size must be less than or equal to window size");
    }

    audio_utils::analysis::SpectrogramInfo spec_info{
        .fft_size = options.fft_length,
        .overlap = options.fft_length - options.hop_size,
        .window_size = options.window_size,
        .samplerate = 48000,
        .window_type = options.window_type,
    };

    auto spectrogram = audio_utils::analysis::MelSpectrogram(signal, spec_info, options.n_mels);

    Eigen::Map<const Eigen::MatrixXf> spec_map(spectrogram.data.data(), spectrogram.num_bins, spectrogram.num_frames);

    std::vector<float> edr_data(spectrogram.num_frames * spectrogram.num_bins, 0);
    Eigen::Map<Eigen::MatrixXf> edr_map(edr_data.data(), spectrogram.num_bins, spectrogram.num_frames);

    Eigen::ArrayXf cumulative_energy = Eigen::ArrayXf::Zero(spectrogram.num_bins);
    for (int i = spec_map.cols() - 1; i >= 0; --i)
    {
        cumulative_energy += spec_map.col(i).array().square();
        edr_map.col(i) = cumulative_energy;
    }

    edr_map = 10.0f * Eigen::log10(edr_map.array() + 1e-10f);

    EnergyDecayReliefResult edr_data_struct{
        .data = std::move(edr_data), .num_bins = spectrogram.num_bins, .num_frames = spectrogram.num_frames};

    return edr_data_struct;
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