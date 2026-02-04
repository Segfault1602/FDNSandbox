#pragma once

#include <sffdn/sffdn.h>

#include <quill/LogMacros.h>
#include <quill/Logger.h>

#include "optim_types.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

namespace fdn_optimization
{

enum class OptimizationStatus : uint8_t
{
    Ready,
    StartRequested,
    Running,
    CancelRequested,
    Completed,
    Canceled,
    Failed
};

struct AdamParameters
{
    float step_size = 0.4253;
    float learning_rate_decay = 0.613;
    int decay_step_size = 50;
    int epoch_restarts = 250;
    int max_restarts = 1;
    float tolerance = 1e-3;
};

struct SPSAParameters
{
    double alpha = 0.01;
    double gamma = 0.101;
    double step_size = 0.9;
    double evaluationStepSize = 0.9;
    size_t max_iterations = 1e6;
    double tolerance = 1e-5;
};

struct SimulatedAnnealingParameters
{
    size_t max_iterations = 1e6;
    double initial_temperature = 10000.0;
    size_t init_moves = 1000;
    size_t move_ctrl_sweep = 100;
    size_t max_tolerance_sweep = 3;
    double max_move_coef = 20.0;
    double init_move_coef = 0.3;
    double gain = 0.3;
};

struct DifferentialEvolutionParameters
{
    size_t population_size = 100;
    size_t max_generation = 2000;
    double crossover_rate = 0.6;
    double differential_weight = 0.8;
};

struct PSOParameters
{
    size_t num_particles = 64;
    size_t max_iterations = 3000;
    size_t horizon_size = 350;
    double exploitation_factor = 2.05;
    double exploration_factor = 2.05;
};

struct RandomSearchParameters
{
};

struct L_BFGSParameters
{
    size_t num_basis = 10;
    size_t max_iterations = 1000;
    double wolfe = 0.9;
    double min_gradient_norm = 1e-6;
    double factor = 1e-15;
    size_t max_line_search_trials = 50;
    double min_step = 1e-20;
    double max_step = 1e20;
};

struct GradientDescentParameters
{
    double step_size = 0.01;
    size_t max_iterations = 1e6;
    double tolerance = 1e-5;
};

struct CMAESParameters
{
    size_t population_size = 0;
    size_t max_iterations = 1000;
    double tolerance = 1e-5;
    double step_size = 0;
};

enum class OptimizationAlgoType : uint8_t
{
    SPSA,
    SimulatedAnnealing,
    DifferentialEvolution,
    PSO,
    RandomSearch,
    CMAES,
    // Below here use gradient information
    Adam,
    L_BFGS,
    GradientDescent,
    Count,
};

constexpr const char* OptimizationAlgoTypeToString(OptimizationAlgoType type)
{
    switch (type)
    {
    case OptimizationAlgoType::Adam:
        return "Adam";
    case OptimizationAlgoType::SPSA:
        return "SPSA";
    case OptimizationAlgoType::SimulatedAnnealing:
        return "Simulated Annealing";
    case OptimizationAlgoType::DifferentialEvolution:
        return "Differential Evolution";
    case OptimizationAlgoType::PSO:
        return "Particle Swarm Optimization";
    case OptimizationAlgoType::RandomSearch:
        return "Random Search";
    case OptimizationAlgoType::L_BFGS:
        return "L-BFGS";
    case OptimizationAlgoType::GradientDescent:
        return "Gradient Descent";
    case OptimizationAlgoType::CMAES:
        return "CMA-ES";
    default:
        return "Unknown";
    }
}

using OptimizationAlgoParams =
    std::variant<AdamParameters, SPSAParameters, SimulatedAnnealingParameters, DifferentialEvolutionParameters,
                 PSOParameters, RandomSearchParameters, L_BFGSParameters, GradientDescentParameters, CMAESParameters>;

struct OptimizationInfo
{
    std::vector<OptimizationParamType> parameters_to_optimize;
    sfFDN::FDNConfig initial_fdn_config;
    uint32_t ir_size;
    fdn_optimization::GradientMethod gradient_method = fdn_optimization::GradientMethod::CentralDifferences;
    double gradient_delta = 1e-4;

    // Colorless loss weights
    double spectral_flatness_weight = 1.0;
    double sparsity_weight = 1.0;
    double power_envelope_weight = 1.0;

    // RIR match loss weights
    double edc_weight = 0.0;
    double mel_edr_weight = 1.0;

    uint32_t mel_edr_fft_length = 4096;
    uint32_t mel_edr_hop_size = 128;
    uint32_t mel_edr_window_size = 1024;
    uint32_t mel_edr_num_bands = 32;

    std::vector<float> target_rir;

    OptimizationAlgoParams optimizer_params;
};

struct OptimizationProgressInfo
{
    std::chrono::duration<double> elapsed_time;
    uint32_t evaluation_count;
    std::vector<std::vector<double>> loss_history;
};

struct OptimizationResult
{
    sfFDN::FDNConfig initial_fdn_config;
    sfFDN::FDNConfig optimized_fdn_config;
    std::chrono::duration<double> total_time;
    uint32_t total_evaluations;
    std::vector<std::vector<double>> loss_history;
    std::vector<std::string> loss_names;
    double best_loss;
};

class OptimCallback;

class FDNOptimizer
{
  public:
    FDNOptimizer(quill::Logger* logger);
    ~FDNOptimizer();

    void StartOptimization(OptimizationInfo& info);
    void CancelOptimization();
    void ResetStatus();

    OptimizationStatus GetStatus() const;
    OptimizationProgressInfo GetProgress();

    OptimizationResult GetResult();

  private:
    void ThreadProc(std::stop_token stop_token, OptimizationInfo info);

    quill::Logger* logger_;
    std::atomic<OptimizationStatus> status_;

    std::chrono::steady_clock::time_point start_time_;
    std::jthread thread_;
    std::mutex mutex_;

    sfFDN::FDNConfig optimized_config_;
    OptimizationResult optimization_result_;

    std::unique_ptr<OptimCallback> optim_callback_;
};
} // namespace fdn_optimization