#include "optimizer.h"

#include <sffdn/sffdn.h>

#include <audio_utils/audio_file_manager.h>

#include "quill/Logger.h"
#include "quill/sinks/ConsoleSink.h"
#include <armadillo>
#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>

#include <iostream>
#include <ostream>
#include <random>
#include <thread>
#include <vector>

constexpr uint32_t kFDNOrder = 8;
constexpr uint32_t kSampleRate = 48000;

void WriteConfigToFile(const sfFDN::FDNConfig& config, const std::string_view filename, quill::Logger* logger);
void SaveImpulseResponse(const sfFDN::FDNConfig& config, uint32_t ir_length, const std::string_view filename,
                         quill::Logger* logger);
void WriteLossHistoryToFile(const std::vector<std::vector<double>>& loss_history,
                            const std::vector<std::string>& loss_names, const std::string_view filename,
                            quill::Logger* logger);

int main()
{
    quill::Backend::start();
    quill::Logger* logger = quill::Frontend::create_or_get_logger(
        "root", quill::Frontend::create_or_get_sink<quill::ConsoleSink>("sink_id_1"));

    std::cout << "FDN Optimization Tool" << std::endl;

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::Gains,
                                      fdn_optimization::OptimizationParamType::Matrix_Householder};

    sfFDN::FDNConfig initial_fdn_config{};
    initial_fdn_config.N = kFDNOrder;
    initial_fdn_config.transposed = false;
    initial_fdn_config.input_gains = std::vector<float>(kFDNOrder, 0.5f);
    initial_fdn_config.output_gains = std::vector<float>(kFDNOrder, 0.5f);
    initial_fdn_config.attenuation_t60s = {10.f};

    if (kFDNOrder == 4)
    {
        initial_fdn_config.delays = {1499, 1889, 2381, 2999};
    }
    else if (kFDNOrder == 6)
    {
        initial_fdn_config.delays = {997, 1153, 1327, 1559, 1801, 2099};
    }
    else if (kFDNOrder == 8)
    {
        initial_fdn_config.delays = {809, 877, 937, 1049, 1151, 1249, 1373, 1499};
    }
    else
    {
        initial_fdn_config.delays = sfFDN::GetDelayLengths(kFDNOrder, 512, 3000, sfFDN::DelayLengthType::Random, 42);
    }

    // ADAM paramaters
    fdn_optimization::AdamParameters adam_params{.step_size = 0.6, .learning_rate_decay = 0.99};

    std::random_device rd;
    auto seed = rd();
    LOG_INFO(logger, "Using random seed: {}", seed);
    // arma::arma_rng::set_seed(seed);

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = kSampleRate,
                                                .optimizer_params = adam_params};

    fdn_optimization::FDNOptimizer optimizer(logger);

    optimizer.StartOptimization(opt_info);

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto progress = optimizer.GetProgress();
        LOG_INFO(logger, "Elapsed Time: {:.2f} s, Evaluations: {}", progress.elapsed_time.count(),
                 progress.evaluation_count);
    }

    auto result = optimizer.GetResult();
    LOG_INFO(logger, "Optimization completed in {:.2f} s with {} evaluations.", result.total_time.count(),
             result.total_evaluations);

    WriteConfigToFile(result.initial_fdn_config, "optim_output/initial_fdn_config.txt", logger);
    WriteConfigToFile(result.optimized_fdn_config, "optim_output/optimized_fdn_config.txt", logger);
    SaveImpulseResponse(result.initial_fdn_config, kSampleRate, "optim_output/initial_ir.wav", logger);
    SaveImpulseResponse(result.optimized_fdn_config, kSampleRate, "optim_output/optimized_ir.wav", logger);
    WriteLossHistoryToFile(result.loss_history, result.loss_names, "optim_output/loss_history.txt", logger);

    return 0;
}

void WriteConfigToFile(const sfFDN::FDNConfig& config, const std::string_view filename, quill::Logger* logger)
{
    std::ofstream file(filename.data(), std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing FDNConfig.", filename);
        return;
    }

    // Format is
    // Row 1: input_gains
    // Row 2: output_gains
    // Row 3: delays
    // Next N rows: feedback matrix
    for (const auto& gain : config.input_gains)
    {
        file << gain << " ";
    }
    file << std::endl;

    for (const auto& gain : config.output_gains)
    {
        file << gain << " ";
    }
    file << std::endl;

    for (const auto& delay : config.delays)
    {
        file << delay << " ";
    }
    file << std::endl;

    if (std::holds_alternative<std::vector<float>>(config.matrix_info))
    {
        const auto& matrix_coeffs = std::get<std::vector<float>>(config.matrix_info);
        const uint32_t N = config.N;
        for (uint32_t r = 0; r < N; ++r)
        {
            for (uint32_t c = 0; c < N; ++c)
            {
                file << matrix_coeffs[r * N + c] << " ";
            }
            file << std::endl;
        }
    }
    else
    {
        LOG_ERROR(logger, "Feedback matrix is not in expected format for writing to file.");
    }
}

void SaveImpulseResponse(const sfFDN::FDNConfig& config, uint32_t ir_length, const std::string_view filename,
                         quill::Logger* logger)
{
    auto config_copy = config;
    config_copy.attenuation_t60s = {1.f};

    auto fdn = sfFDN::CreateFDNFromConfig(config_copy, kSampleRate);
    fdn->SetDirectGain(0.0f);

    std::vector<float> input_data(ir_length, 0.0f);
    input_data[0] = 1.0f; // Delta impulse

    std::vector<float> impulse_response(ir_length, 0.0f);
    sfFDN::AudioBuffer impulse_buffer(impulse_response);

    sfFDN::AudioBuffer in_buffer(input_data);
    fdn->Process(in_buffer, impulse_buffer);

    LOG_INFO(logger, "Writing impulse response to file: {}", filename);
    audio_utils::audio_file::WriteWavFile(filename.data(), impulse_response, kSampleRate);
}

void WriteLossHistoryToFile(const std::vector<std::vector<double>>& loss_history,
                            const std::vector<std::string>& loss_names, const std::string_view filename,
                            quill::Logger* logger)
{
    std::ofstream file(filename.data(), std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger, "Failed to open file {} for writing loss history.", filename);
        return;
    }

    // Check that all loss vectors have the same length
    size_t history_length = loss_history[0].size();
    for (const auto& losses : loss_history)
    {
        if (losses.size() != history_length)
        {
            LOG_ERROR(logger, "Inconsistent loss history lengths when writing to file {}.", filename);
            return;
        }
    }

    // Write header
    file << "Total, ";
    for (size_t i = 0; i < loss_names.size(); ++i)
    {
        file << loss_names[i];
        if (i < loss_names.size() - 1)
        {
            file << ", ";
        }
    }
    file << std::endl;

    for (size_t i = 0; i < history_length; ++i)
    {
        for (size_t j = 0; j < loss_history.size(); ++j)
        {
            file << loss_history[j][i];
            if (j < loss_history.size() - 1)
            {
                file << ", ";
            }
        }
        file << std::endl;
    }
}