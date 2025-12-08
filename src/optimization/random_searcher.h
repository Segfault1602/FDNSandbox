#pragma once

#include "model.h"

#include <armadillo>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace fdn_optimization
{

class RandomSearcher
{
  public:
    RandomSearcher();

    void StartSearch(const FDNModel& model, std::stop_token stop_token);

    double GetBestObjective()
    {
        std::scoped_lock lock(mutex_);
        return best_objective_;
    }

    arma::mat GetBestParams()
    {
        std::scoped_lock lock(mutex_);
        return best_params_;
    }

    uint32_t GetEvaluationCount() const
    {
        return evaluation_count_.load();
    }

  private:
    std::mutex mutex_;
    std::stop_token stop_token_;
    double best_objective_;
    arma::mat best_params_;
    std::vector<std::jthread> threads_;
    std::atomic<uint32_t> evaluation_count_{0};
};
} // namespace fdn_optimization