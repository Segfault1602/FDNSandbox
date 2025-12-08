#pragma once

#include <armadillo>
#include <audio_utils/fft.h>
#include <sffdn/sffdn.h>

#include "loss.h"

#include <cstdint>

namespace fdn_optimization
{

enum class OptimizationParamType : uint8_t
{
    Gains,
    Matrix,
    Matrix_Householder,
    Matrix_Circulant,
    Delays,
};

enum class GradientMethod : uint8_t
{
    CentralDifferences,
    ForwardDifferences,
};

class FDNModel
{
  public:
    FDNModel(uint32_t fdn_order, uint32_t ir_size, std::span<const uint32_t> delays,
             std::span<const OptimizationParamType> param_types,
             GradientMethod gradient_method = GradientMethod::CentralDifferences);

    FDNModel(const FDNModel&);
    FDNModel& operator=(const FDNModel&);

    void SetLossFunctions(const std::vector<LossFunction>& loss_functions);

    std::vector<LossFunction> GetLossFunctions() const
    {
        return loss_functions_;
    }

    uint32_t GetParamCount() const;

    void SetGradientDelta(double delta)
    {
        gradient_delta_ = delta;
    }

    arma::mat GetInitialParams() const;

    std::span<const float> GenerateIR();

    void Setup(const arma::mat& params);

    double Evaluate(const arma::mat& params);
    double Evaluate(const arma::mat& params, const size_t i, const size_t batch_size);

    double EvaluateWithGradient(const arma::mat& x, arma::mat& g);
    double EvaluateWithGradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize);

    sfFDN::FDNConfig GetFDNConfig(const arma::mat& params) const;

    void PrintFDNConfig(const arma::mat& params) const;

    size_t NumFunctions() const
    {
        return 1;
    }

    void Shuffle()
    {
        // No-op
    }

    std::vector<double> last_losses_;

  private:
    std::unique_ptr<sfFDN::FDN> fdn_;
    uint32_t ir_size_;
    std::vector<float> impulse_buffer_;
    std::vector<float> response_buffer_;
    std::vector<LossFunction> loss_functions_;

    std::vector<float> matrix_coeffs_;
    std::vector<OptimizationParamType> param_types_;
    std::vector<uint32_t> delays_;

    double gradient_delta_ = 1e-3;

    GradientMethod gradient_method_ = GradientMethod::CentralDifferences;

    void GradientCentralDifferences(const arma::mat& x, arma::mat& g);
    void GradientForwardDifferences(const arma::mat& x, arma::mat& g, double current_loss);
};
} // namespace fdn_optimization