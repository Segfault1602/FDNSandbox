#include "random_searcher.h"

#include <armadillo>

#include <limits>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace fdn_optimization
{
RandomSearcher::RandomSearcher()
    : best_objective_(std::numeric_limits<double>::infinity())
{
}

void RandomSearcher::StartSearch(const FDNModel& model, std::stop_token stop_token)
{
    const size_t n_threads = std::thread::hardware_concurrency();
    threads_.clear();
    stop_token_ = stop_token;
    best_params_ = model.GetInitialParams();

    for (size_t t = 0; t < n_threads; ++t)
    {
        threads_.emplace_back([this, &model]() {
            FDNModel thread_model = model;

            arma::mat initial_params = thread_model.GetInitialParams();
            arma::mat params = arma::mat(initial_params.n_rows, initial_params.n_cols, arma::fill::randu) * 2.0 - 1.0;
            arma::mat gradient;
            double last_objective = std::numeric_limits<double>::infinity();
            while (!stop_token_.stop_requested())
            {

                double objective = thread_model.Evaluate(params);
                evaluation_count_++;

                {
                    std::scoped_lock lock(mutex_);
                    if (objective < best_objective_)
                    {
                        best_objective_ = objective;
                        best_params_ = params;
                    }
                }

                if (std::abs(objective - last_objective) < 1e-6)
                {
                    // If no improvement, randomize params
                    params = arma::mat(initial_params.n_rows, initial_params.n_cols, arma::fill::randu) * 2.0 - 1.0;
                }
                else
                {
                    thread_model.EvaluateWithGradient(params, gradient);
                    params -= 0.1 * gradient;
                }
                last_objective = objective;
            }
        });
    }

    return;
}
} // namespace fdn_optimization