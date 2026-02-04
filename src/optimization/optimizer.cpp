#include "optimizer.h"

#include "model.h"
#include "random_searcher.h"
#include <audio_utils/audio_analysis.h>

#include <armadillo>
#include <ensmallen.hpp>

#include <iostream>
#include <thread>

template <typename T>
concept HasStepSize = requires(T a) { a.StepSize(); };

template <class T, class U>
struct is_same_template : std::false_type
{
};

template <template <class...> class C, class... R1s, class... R2s>
struct is_same_template<C<R1s...>, C<R2s...>> : std::true_type
{
};

template <class T, class U>
inline constexpr bool is_same_template_v = is_same_template<T, U>::value;

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
    void BeginOptimization(OptimizerType& optimizer, FunctionType& function, MatType&)
    {
        if constexpr (HasStepSize<OptimizerType>)
        {
            starting_step_size_ = optimizer.StepSize();
        }
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
                // For DE, there is no easy way to get the best objective per generation so we have to keep track of
                // it manually.
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

        if constexpr (is_same_template_v<OptimizerType, ens::SGD<>>)
        {
            if (decay_step_size_ > 0 && epoch % decay_step_size_ == 0)
            {
                optimizer.StepSize() = optimizer.StepSize() * decay_rate_;
            }

            if (epoch_restarts_ > 0 && epoch % epoch_restarts_ == 0 && epoch != 0 && restart_count_ < max_restarts_)
            {
                optimizer.StepSize() = starting_step_size_;
                ++restart_count_;
            }
        }

        return stop_token_.stop_requested();
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool StepTaken(OptimizerType&, FunctionType& function, MatType& iterate)
    {
        step_was_taken_ = true;

        if constexpr (std::is_same_v<OptimizerType, ens::SA<ens::ExponentialSchedule>> ||
                      std::is_same_v<OptimizerType, ens::LBestPSO> || std::is_same_v<OptimizerType, ens::L_BFGS> ||
                      std::is_same_v<OptimizerType, ens::GradientDescent> ||
                      is_same_template_v<OptimizerType, ens::CMAES<>> ||
                      is_same_template_v<OptimizerType, ens::ActiveCMAES<>>)
        {
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
    size_t decay_step_size_ = 1;
    size_t epoch_restarts_ = 0;
    size_t restart_count_ = 0;
    size_t max_restarts_ = 0;

    int de_pop_size_ = 0;
    int de_pop_evals_ = 0;
    double de_best_objective_ = std::numeric_limits<double>::max();
    arma::mat de_best_params_;

  private:
    std::mutex mutex_;
    std::vector<double> loss_history_;
    bool step_was_taken_ = false;
    double starting_step_size_ = 0.01;

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
        optim_callback->epoch_restarts_ = adam_params.epoch_restarts;
        optim_callback->decay_step_size_ = adam_params.decay_step_size;
        optim_callback->max_restarts_ = adam_params.max_restarts;
        ens::Adam optimizer(adam_params.step_size, 1, 0.9, 0.999, 1e-8, 1e6, adam_params.tolerance, false, true, true);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(SPSAParameters& spsa_params)
    {
        ens::SPSA optimizer(spsa_params.alpha, spsa_params.gamma, spsa_params.step_size, spsa_params.evaluationStepSize,
                            spsa_params.max_iterations, spsa_params.tolerance);

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
        ens::SGD optimizer(p.step_size, 1, p.max_iterations, p.tolerance, false);

        ens::StoreBestCoordinates<arma::mat> store_best;
        optimizer.Optimize(model, params, store_best, ens::Report(), *optim_callback);

        params = store_best.BestCoordinates();
    }

    void operator()(CMAESParameters& p)
    {
        // ens::BoundaryBoxConstraint b(-1.0, 1.0);
        ens::ActiveCMAES optimizer(p.population_size, ens::EmptyTransformation(), 1, p.max_iterations, p.tolerance,
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

    const bool optimizing_filters =
        std::ranges::find(info.parameters_to_optimize, fdn_optimization::OptimizationParamType::AttenuationFilters) !=
        info.parameters_to_optimize.end();

    if (optimizing_filters && info.target_rir.empty())
    {
        LOG_ERROR(logger_, "Target RIR must be provided when optimizing filters. Cancelling optimization.");
        status_.store(OptimizationStatus::Failed);
        return;
    }

    if (optimizing_filters && info.ir_size != info.target_rir.size())
    {
        LOG_WARNING(logger_,
                    "IR size ({}) does not match target RIR size ({}). Adjusting IR size to match target RIR size.",
                    info.ir_size, info.target_rir.size());
        info.ir_size = static_cast<uint32_t>(info.target_rir.size());
    }

    FDNModel model(info.initial_fdn_config, info.ir_size, info.parameters_to_optimize, info.gradient_method);
    model.SetGradientDelta(info.gradient_delta);

    std::vector<LossFunction> loss_functions;

    if (optimizing_filters)
    {
        // std::vector<float> time_data(target_edc_octaves[0].size());
        // for (size_t i = 0; i < time_data.size(); ++i)
        // {
        //     time_data[i] = static_cast<float>(i) * (1.0f / 48000.0f); // assuming 48kHz sample rate
        // }

        // std::vector<float> estimated_t60s;
        // for (const auto& edc_octave : target_edc_octaves)
        // {
        //     auto t60_results = fdn_analysis::EstimateT60(edc_octave, time_data, -5.0f, -15.0f);
        //     estimated_t60s.push_back(t60_results.t60);
        //     LOG_INFO(logger_, "Estimated T60: {:.2f} s", t60_results.t60);
        // }

        if (info.edc_weight > 0.0)
        {
            auto target_edc_octaves = audio_utils::analysis::EnergyDecayCurve_FilterBank(info.target_rir, true);
            // use shared_ptr to capture in lambda
            auto target_edc_octaves_ptr =
                std::make_shared<std::array<std::vector<float>, audio_utils::analysis::kNumOctaveBands>>(
                    std::move(target_edc_octaves));

            LossFunction edc_loss;
            edc_loss.func = [target_edc_octaves_ptr](std::span<const float> signal) -> double {
                return EDCLoss(signal, *target_edc_octaves_ptr);
            };
            edc_loss.weight = 1.0;
            edc_loss.name = "EDC Relief Loss";
            loss_functions.push_back(edc_loss);
        }

        if (info.mel_edr_weight > 0.0)
        {
            audio_utils::analysis::EnergyDecayReliefOptions edr_options;
            edr_options.fft_length = info.mel_edr_fft_length;
            edr_options.hop_size = info.mel_edr_hop_size;
            edr_options.window_size = info.mel_edr_window_size;
            edr_options.window_type = audio_utils::FFTWindowType::Hann;
            edr_options.n_mels = info.mel_edr_num_bands;

            auto target_edr_result = audio_utils::analysis::EnergyDecayRelief(info.target_rir, edr_options);

            auto target_edr_result_ptr =
                std::make_shared<audio_utils::analysis::EnergyDecayReliefResult>(std::move(target_edr_result));

            LossFunction edr_loss;
            edr_loss.func = [target_edr_result_ptr, edr_options](std::span<const float> signal) -> double {
                return EDRLoss(signal, *target_edr_result_ptr, edr_options);
            };
            edr_loss.weight = 1.0;
            edr_loss.name = "EDR Loss";
            loss_functions.push_back(edr_loss);
        }
    }
    else
    {
        if (info.spectral_flatness_weight > 0.0)
        {
            LossFunction spectral_flatness_loss;
            spectral_flatness_loss.func = [&](std::span<const float> signal) -> double {
                double spectral_flatness = SpectralFlatnessLoss(signal);
                return std::abs(0.5575f - spectral_flatness);
            };
            spectral_flatness_loss.weight = info.spectral_flatness_weight;
            spectral_flatness_loss.name = "Spectral Flatness Loss";
            loss_functions.push_back(spectral_flatness_loss);
        }

        if (info.power_envelope_weight > 0.0)
        {
            LossFunction power_env_loss;
            power_env_loss.func = [&](std::span<const float> signal) -> double {
                constexpr uint32_t kSampleRate = 48000;
                return PowerEnvelopeLoss(signal, 1024, 128, kSampleRate);
            };
            power_env_loss.weight = info.power_envelope_weight;
            power_env_loss.name = "Power Envelope Loss";
            loss_functions.push_back(power_env_loss);
        }

        if (info.sparsity_weight > 0.0)
        {
            LossFunction sparsity_loss;
            sparsity_loss.func = [&](std::span<const float> signal) -> double {
                double sparsity = SparsityLoss(signal.subspan(0, 4096));
                return sparsity;
            };
            sparsity_loss.weight = info.sparsity_weight;
            sparsity_loss.name = "Sparsity Loss";
            loss_functions.push_back(sparsity_loss);
        }
    }

    model.SetLossFunctions(loss_functions);

    arma::mat params = model.GetInitialParams();

    double initial_loss = model.Evaluate(params);
    LOG_INFO(logger_, "Initial loss: {}", initial_loss);
    sfFDN::FDNConfig initial_config = model.GetFDNConfig(params);

    optim_callback_ = std::make_unique<OptimCallback>(stop_token);

    OptimizationVisitor visitor{params, model, optim_callback_.get(), info};
    std::visit(visitor, info.optimizer_params);

    double final_loss = model.Evaluate(params);
    LOG_INFO(logger_, "Final loss: {}", final_loss);

    {
        std::scoped_lock lock(mutex_);

        optimized_config_ = model.GetFDNConfig(params);

        optimization_result_.initial_fdn_config = initial_config;
        optimization_result_.optimized_fdn_config = optimized_config_;
        optimization_result_.total_time = std::chrono::steady_clock::now() - start_time_;
        optimization_result_.total_evaluations = optim_callback_->evaluation_count_.load();
        optimization_result_.loss_history = optim_callback_->GetLossHistory();
        optimization_result_.best_loss = final_loss;

        optimization_result_.loss_names.clear();
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