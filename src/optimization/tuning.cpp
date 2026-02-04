#include "optimizer.h"

#include <sffdn/sffdn.h>

#include <audio_utils/audio_analysis.h>
#include <audio_utils/audio_file_manager.h>

#include "quill/Logger.h"
#include "quill/sinks/ConsoleSink.h"
#include <armadillo>
#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>

#include <iostream>
#include <ostream>
#include <print>
#include <random>
#include <thread>
#include <vector>

constexpr uint32_t kFDNOrder = 8;
constexpr uint32_t kSampleRate = 48000;

fdn_optimization::OptimizationResult DoOptimization(quill::Logger* logger, fdn_optimization::OptimizationInfo& opt_info)
{
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
        float last_loss = 0.0f;
        if (!progress.loss_history.empty() && !progress.loss_history[0].empty())
        {
            last_loss = static_cast<float>(progress.loss_history[0].back());
        }
        LOG_INFO(logger, "Elapsed Time: {:.2f} s, Evaluations: {}, Last Loss: {:.6f}", progress.elapsed_time.count(),
                 progress.evaluation_count, last_loss);
    }

    auto result = optimizer.GetResult();
    return result;
}

fdn_optimization::AdamParameters GetAdamParametersFromArgs(std::span<const std::string> args)
{
    fdn_optimization::AdamParameters params;
    if (args.size() < 6)
    {
        std::println("Not enough arguments to parse ADAM parameters.");
        throw std::invalid_argument("Not enough arguments to parse ADAM parameters.");
    }

    params.step_size = std::stod(args[0]);
    params.learning_rate_decay = std::stod(args[1]);
    params.decay_step_size = static_cast<size_t>(std::stoull(args[2]));
    params.epoch_restarts = static_cast<size_t>(std::stoull(args[3]));
    params.max_restarts = static_cast<size_t>(std::stoull(args[4]));
    params.tolerance = std::stod(args[5]);

    std::println("Using ADAM optimizer with step size {}, learning rate decay {}, tolerance {}, epoch restarts {}, "
                 "decay step size {}",
                 params.step_size, params.learning_rate_decay, params.tolerance, params.epoch_restarts,
                 params.decay_step_size);

    return params;
}

fdn_optimization::SPSAParameters GetSPSAParametersFromArgs(std::span<const std::string> args)
{
    fdn_optimization::SPSAParameters params;
    if (args.size() < 6)
    {
        std::println("Not enough arguments to parse SPSA parameters.");
        throw std::invalid_argument("Not enough arguments to parse SPSA parameters.");
    }

    params.alpha = std::stod(args[0]);
    params.gamma = std::stod(args[1]);
    params.step_size = std::stod(args[2]);
    params.evaluationStepSize = std::stod(args[3]);
    params.max_iterations = static_cast<size_t>(std::stoull(args[4]));
    params.tolerance = std::stod(args[5]);

    std::println(
        "Using SPSA optimizer with alpha {}, gamma {}, step size {}, evaluation step size {}, max iterations {}, "
        "tolerance {} ",
        params.alpha, params.gamma, params.step_size, params.evaluationStepSize, params.max_iterations,
        params.tolerance);

    return params;
}

fdn_optimization::CMAESParameters GetCMAESParametersFromArgs(std::span<const std::string> args)
{
    fdn_optimization::CMAESParameters params;
    if (args.size() < 4)
    {
        std::println("Not enough arguments to parse CMAES parameters.");
        throw std::invalid_argument("Not enough arguments to parse CMAES parameters.");
    }

    params.population_size = static_cast<size_t>(std::stoull(args[0]));
    params.max_iterations = static_cast<size_t>(std::stoull(args[1]));
    params.tolerance = std::stod(args[2]);
    params.step_size = std::stod(args[3]);

    std::println("Using CMAES optimizer with population size {}, max iterations {}, tolerance {}, step size {}",
                 params.population_size, params.max_iterations, params.tolerance, params.step_size);

    return params;
}

int main(int argc, char** argv)
{
    quill::Backend::start();
    quill::Logger* logger = quill::Frontend::create_or_get_logger(
        "root", quill::Frontend::create_or_get_sink<quill::ConsoleSink>("sink_id_1"));

    std::cout << "FDN Optimization Tool" << std::endl;

    std::vector<std::string> args(argv + 1, argv + argc);

    std::vector<float> target_rir;
    int num_channels = 0;
    int sample_rate = kSampleRate;
    if (audio_utils::audio_file::ReadWavFile("../rirs/py_rirs/rir_dining_room.wav", target_rir, sample_rate,
                                             num_channels))
    {
        LOG_INFO(logger, "Loaded target RIR with {} samples at {} Hz.", target_rir.size(), sample_rate);
    }
    else
    {
        LOG_ERROR(logger, "Failed to load target RIR.");
        return -1;
    }

    sfFDN::FDNConfig initial_fdn_config{};
    initial_fdn_config.N = kFDNOrder;
    initial_fdn_config.transposed = false;
    initial_fdn_config.input_gains = std::vector<float>(kFDNOrder, 1.0f / std::sqrt(kFDNOrder));
    initial_fdn_config.output_gains = std::vector<float>(kFDNOrder, 1.0f / std::sqrt(kFDNOrder));
    initial_fdn_config.attenuation_t60s = {10.f};
    initial_fdn_config.tc_frequencies = {31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};

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

    initial_fdn_config.matrix_info = sfFDN::GenerateMatrix(kFDNOrder, sfFDN::ScalarMatrixType::Random, 42);

    // std::random_device rd;
    // auto seed = rd();
    // LOG_INFO(logger, "Using random seed: {}", seed);
    // arma::arma_rng::set_seed(seed);

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::Gains,
                                      fdn_optimization::OptimizationParamType::Matrix};
    // std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::AttenuationFilters,
    //                                   fdn_optimization::OptimizationParamType::TonecorrectionFilters};

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = static_cast<uint32_t>(target_rir.size()),
                                                .gradient_method = fdn_optimization::GradientMethod::CentralDifferences,
                                                .gradient_delta = 1e-2,
                                                .target_rir = target_rir,
                                                .optimizer_params = {}};
    const std::string optim_type = args[0];
    try
    {
        if (optim_type == "ADAM")
        {
            opt_info.optimizer_params = GetAdamParametersFromArgs(std::span<const std::string>(args).subspan(1));
        }
        else if (optim_type == "SPSA")
        {
            opt_info.optimizer_params = GetSPSAParametersFromArgs(std::span<const std::string>(args).subspan(1));
        }
        else if (optim_type == "CMAES")
        {
            opt_info.optimizer_params = GetCMAESParametersFromArgs(std::span<const std::string>(args).subspan(1));
        }
        else
        {
            LOG_ERROR(logger, "Unknown optimizer type: {}. Supported types are ADAM, SPSA, CMAES.", optim_type);
            return -1;
        }
    }
    catch (const std::exception& e)
    {
        std::print("Error parsing optimizer parameters: {}\n", e.what());
        return -1;
    }

    // auto result = DoOptimization(logger, opt_info);
    // LOG_INFO(logger, "Best loss: {}", result.best_loss);
    // LOG_INFO(logger, "Final time: {} seconds", result.total_time.count());
    // LOG_INFO(logger, "Colorless optimization completed in {:.2f} s with {} evaluations.", result.total_time.count(),
    //          result.total_evaluations);

    // return 0;

    opt_info.parameters_to_optimize = {fdn_optimization::OptimizationParamType::AttenuationFilters,
                                       fdn_optimization::OptimizationParamType::TonecorrectionFilters,
                                       fdn_optimization::OptimizationParamType::OverallGain};

    // fdn_optimization::AdamParameters filter_opt_params{
    //     .step_size = 0.6, .learning_rate_decay = 0.95, .tolerance = 1e-4};
    // fdn_optimization::CMAESParameters filter_opt_params{
    //     .population_size = 200, .max_iterations = 100000, .tolerance = 1e-4, .step_size = 0.5};
    // fdn_optimization::SimulatedAnnealingParameters filter_opt_params{.initial_temperature = 100.0};
    // opt_info.optimizer_params = filter_opt_params;
    auto result = DoOptimization(logger, opt_info);
    LOG_INFO(logger, "Best loss: {}", result.best_loss);
    LOG_INFO(logger, "Final time: {} seconds", result.total_time.count());

    LOG_INFO(logger, "Filter optimization completed in {:.2f} s with {} evaluations.", result.total_time.count(),
             result.total_evaluations);

    return 0;
}