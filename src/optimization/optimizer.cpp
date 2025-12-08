#include "optimizer.h"

#include "model.h"
#include "random_searcher.h"

#include <armadillo>
#include <ensmallen.hpp>

namespace
{

} // namespace

namespace fdn_optimization
{
class OptimCallback
{
  public:
    OptimCallback(std::stop_token stop_token, double decay_rate = 0.99)
        : stop_token_(stop_token)
        , evaluation_count_(0)
        , decay_rate_(decay_rate)
    {
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    void BeginOptimization(OptimizerType&, FunctionType& function, MatType&)
    {
        auto loss_functions = function.GetLossFunctions();
        {
            std::scoped_lock lock(mutex_);
            individual_losses_.resize(loss_functions.size());
        }
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool Evaluate(OptimizerType&, FunctionType& function, const MatType& iterate, const double objective)
    {
        ++evaluation_count_;

        if constexpr (std::is_same_v<OptimizerType, ens::SPSA>)
        {
            if (step_was_taken_)
            {
                step_was_taken_ = false;
                SaveLossHistory(function, objective);
            }
        }

        if constexpr (std::is_same_v<OptimizerType, ens::DE>)
        {
            if (de_pop_size_ > 0)
            {
                if (objective < de_best_objective_)
                {
                    de_best_objective_ = objective;
                    de_best_params_ = iterate;
                }
                // For DE, there is no easy way to get the best objective per generation so we have to keep track of it
                // manually.
                ++de_pop_evals_;
                if (de_pop_evals_ == de_pop_size_ * 2) // Each generation evaluates 2 * population size
                {
                    de_pop_evals_ = 0;
                    SaveLossHistory(function, de_best_objective_);
                }
            }
        }

        return stop_token_.stop_requested();
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool EndEpoch(OptimizerType& optimizer, FunctionType& function, const MatType&, const size_t epoch,
                  const double objective)
    {
        SaveLossHistory(function, objective);
        static_assert(std::is_same_v<OptimizerType, ens::SGD<ens::AdamUpdate, ens::NoDecay>>);

        if constexpr (std::is_same_v<OptimizerType, ens::SGD<ens::AdamUpdate, ens::NoDecay>>)
        {
            if (epoch % 10 == 0)
            {
                optimizer.StepSize() = optimizer.StepSize() * decay_rate_;
            }
        }

        return stop_token_.stop_requested();
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool StepTaken(OptimizerType&, FunctionType& function, MatType& iterate)
    {
        step_was_taken_ = true;

        if constexpr (std::is_same_v<OptimizerType, ens::SA<ens::ExponentialSchedule>>)
        {
            // For Simulated Annealing, we only have access to StepTaken.
            // So we save the loss history here.
            // Note that this may include steps that were not accepted.
            SaveLossHistory(function, function.Evaluate(iterate));
        }
        if constexpr (std::is_same_v<OptimizerType, ens::LBestPSO>)
        {
            // For PSO, we only have access to StepTaken.
            // So we save the loss history here.
            // Note that this may include steps that were not accepted.
            SaveLossHistory(function, function.Evaluate(iterate));
        }
        if constexpr (std::is_same_v<OptimizerType, ens::L_BFGS>)
        {
            // For DE, we only have access to StepTaken.
            // So we save the loss history here.
            // Note that this may include steps that were not accepted.
            SaveLossHistory(function, function.Evaluate(iterate));
        }
        if constexpr (std::is_same_v<OptimizerType, ens::GradientDescent>)
        {
            // For DE, we only have access to StepTaken.
            // So we save the loss history here.
            // Note that this may include steps that were not accepted.
            SaveLossHistory(function, function.Evaluate(iterate));
        }
        if constexpr (std::is_same_v<OptimizerType,
                                     ens::CMAES<ens::FullSelection, ens::EmptyTransformation<arma::mat>>>)
        {
            // For DE, we only have access to StepTaken.
            // So we save the loss history here.
            // Note that this may include steps that were not accepted.
            SaveLossHistory(function, function.Evaluate(iterate));
        }

        return false;
    }

    template <typename FunctionType>
    void SaveLossHistory(FunctionType& function, double objective)
    {
        {
            std::scoped_lock lock(mutex_);
            loss_history_.push_back(objective);
        }

        assert(individual_losses_.size() == function.last_losses_.size());
        for (size_t i = 0; i < function.last_losses_.size(); ++i)
        {
            individual_losses_[i].push_back(function.last_losses_[i]);
        }
    }

    std::vector<std::vector<double>> GetLossHistory()
    {
        std::scoped_lock lock(mutex_);
        std::vector<std::vector<double>> all_losses;
        all_losses.push_back(loss_history_);
        for (const auto& losses : individual_losses_)
        {
            all_losses.push_back(losses);
        }
        return all_losses;
    }

    std::stop_token stop_token_;
    std::atomic<uint32_t> evaluation_count_;
    double decay_rate_ = 0.99;

    int de_pop_size_ = 0;
    int de_pop_evals_ = 0;
    double de_best_objective_ = std::numeric_limits<double>::max();
    arma::mat de_best_params_;

  private:
    std::mutex mutex_;
    std::vector<double> loss_history_;
    bool step_was_taken_ = false;

    std::vector<std::vector<double>> individual_losses_;
};

struct OptimizationVisitor
{
    arma::mat& params;
    FDNModel& model;
    OptimCallback* optim_callback;
    const OptimizationInfo& info;

    void operator()(AdamParameters& adam_params)
    {
        optim_callback->decay_rate_ = adam_params.learning_rate_decay;
        ens::Adam optimizer(adam_params.step_size, 1, 0.9, 0.999, 1e-8, 1e6, 1e-5, false, true, true);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(SPSAParameters& spsa_params)
    {
        ens::SPSA optimizer(spsa_params.alpha, spsa_params.gamma, spsa_params.step_size, spsa_params.evaluationStepSize,
                            spsa_params.max_iterations, 1e-7);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(SimulatedAnnealingParameters& p)
    {
        ens::SA optimizer(ens::ExponentialSchedule(), p.max_iterations, p.initial_temperature, p.init_moves,
                          p.move_ctrl_sweep, 1e-5, p.max_tolerance_sweep, p.max_move_coef, p.init_move_coef, p.gain);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(DifferentialEvolutionParameters& p)
    {

        optim_callback->de_pop_size_ = static_cast<int>(p.population_size);
        ens::DE optimizer(p.population_size, p.max_generation, p.crossover_rate, p.differential_weight, 1e-7);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(PSOParameters& p)
    {
        ens::LBestPSO optimizer(p.num_particles, 1.0, 1.0, p.max_iterations, p.horizon_size, 1e-7,
                                p.exploitation_factor, p.exploration_factor);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        // params = store_best.BestCoordinates();
        if (!arma::approx_equal(params, store_best.BestCoordinates(), "absdiff", 1e-5))
        {
            std::cout << "Mismatch in best coordinates!" << std::endl;
        }
    }

    void operator()(RandomSearchParameters&)
    {
        RandomSearcher optimizer;
        optim_callback->BeginOptimization(optimizer, model, params);
        std::stop_token stop_token = optim_callback->stop_token_;
        params = model.GetInitialParams();
        optimizer.StartSearch(model, stop_token);

        while (!stop_token.stop_requested())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            params = optimizer.GetBestParams();
            auto best_objective = optimizer.GetBestObjective();
            optim_callback->evaluation_count_ = optimizer.GetEvaluationCount();
            model.Evaluate(params);
            optim_callback->SaveLossHistory(model, best_objective);
        }

        params = optimizer.GetBestParams();
        optim_callback->evaluation_count_ = optimizer.GetEvaluationCount();
        model.Evaluate(params);
        optim_callback->SaveLossHistory(model, optimizer.GetBestObjective());
    }

    void operator()(L_BFGSParameters& p)
    {
        constexpr double kArmijoConstant = 1e-4;
        ens::L_BFGS optimizer(p.num_basis, p.max_iterations, kArmijoConstant, p.wolfe, p.min_gradient_norm, p.factor,
                              p.max_line_search_trials, p.min_step, p.max_step);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(GradientDescentParameters& p)
    {
        ens::GradientDescent optimizer(p.step_size, p.max_iterations, p.tolerance);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(CMAESParameters& p)
    {
        // ens::BoundaryBoxConstraint b(-1.0, 1.0);
        ens::CMAES optimizer(p.population_size, ens::EmptyTransformation(), 1, p.max_iterations, p.tolerance,
                             ens::FullSelection(), p.step_size);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }
};

FDNOptimizer::FDNOptimizer(quill::Logger* logger)
    : logger_(logger)
    , status_(OptimizationStatus::Ready)
{
}

FDNOptimizer::~FDNOptimizer()
{
    thread_.request_stop();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

void FDNOptimizer::StartOptimization(OptimizationInfo& info)
{
    auto current_status = status_.load();
    if (current_status == OptimizationStatus::Running || current_status == OptimizationStatus::StartRequested)
    {
        LOG_INFO(logger_, "Optimization is already running.");
        return; // Already running
    }

    status_.store(OptimizationStatus::StartRequested);

    // Copy the OptimizationInfo to send to the thread
    OptimizationInfo info_copy = info;

    LOG_INFO(logger_, "Starting optimization.");
    start_time_ = std::chrono::steady_clock::now();
    thread_ = std::jthread([this, info_copy](std::stop_token st) { ThreadProc(st, info_copy); });
    status_.store(OptimizationStatus::Running);
}

void FDNOptimizer::CancelOptimization()
{
    if (status_.load() != OptimizationStatus::Running)
    {
        LOG_INFO(logger_, "Optimization is not running.");
        return; // Not running
    }

    LOG_INFO(logger_, "Requesting optimization cancellation.");
    status_.store(OptimizationStatus::CancelRequested);
    thread_.request_stop();
}

void FDNOptimizer::ResetStatus()
{
    if (status_.load() == OptimizationStatus::Running)
    {
        LOG_WARNING(logger_, "Cannot reset status while optimization is running. Cancelling first.");
        CancelOptimization();
        thread_.join();
    }

    status_.store(OptimizationStatus::Ready);
}

OptimizationStatus FDNOptimizer::GetStatus() const
{
    return status_.load();
}

OptimizationProgressInfo FDNOptimizer::GetProgress()
{
    std::scoped_lock lock(mutex_);

    OptimizationProgressInfo progress;
    progress.elapsed_time = std::chrono::steady_clock::now() - start_time_;

    if (optim_callback_)
    {
        progress.evaluation_count = optim_callback_->evaluation_count_.load();
        progress.loss_history = optim_callback_->GetLossHistory();
    }
    else
    {
        progress.evaluation_count = 0;
    }

    return progress;
}

OptimizationResult FDNOptimizer::GetResult()
{
    std::scoped_lock lock(mutex_);
    return optimization_result_;
}

void FDNOptimizer::ThreadProc(std::stop_token stop_token, OptimizationInfo info)
{
    LOG_INFO(logger_, "Optimization thread started.");
    status_.store(OptimizationStatus::Running);

    uint32_t fdn_order = info.initial_fdn_config.N;
    FDNModel model(fdn_order, info.ir_size, info.initial_fdn_config.delays, info.parameters_to_optimize,
                   info.gradient_method);
    model.SetGradientDelta(info.gradient_delta);
    arma::mat params = model.GetInitialParams();

    double initial_loss = model.Evaluate(params);
    LOG_INFO(logger_, "Initial loss: {}", initial_loss);
    sfFDN::FDNConfig initial_config = model.GetFDNConfig(params);

    optim_callback_ = std::make_unique<OptimCallback>(stop_token);

    OptimizationVisitor visitor{params, model, optim_callback_.get(), info};
    std::visit(visitor, info.optimizer_params);

    {
        std::scoped_lock lock(mutex_);

        optimized_config_ = model.GetFDNConfig(params);

        optimization_result_.initial_fdn_config = initial_config;
        optimization_result_.optimized_fdn_config = optimized_config_;
        optimization_result_.total_time = std::chrono::steady_clock::now() - start_time_;
        optimization_result_.total_evaluations = optim_callback_->evaluation_count_.load();
        optimization_result_.loss_history = optim_callback_->GetLossHistory();

        auto loss_functions = model.GetLossFunctions();
        for (const auto& lf : loss_functions)
        {
            optimization_result_.loss_names.push_back(lf.name);
        }
    }

    auto current_status = status_.load();
    if (current_status == OptimizationStatus::CancelRequested)
    {
        status_.store(OptimizationStatus::Canceled);
        LOG_INFO(logger_, "Optimization was canceled.");
        return;
    }

    status_.store(OptimizationStatus::Completed);
    LOG_INFO(logger_, "Optimization thread completed.");
}

} // namespace fdn_optimization