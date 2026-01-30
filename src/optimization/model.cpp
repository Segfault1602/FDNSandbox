#include "model.h"

#include <armadillo>
#include <sffdn/sffdn.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <omp.h>

namespace
{
constexpr uint32_t kSampleRate = 48000;
constexpr uint32_t kNBands = 10;

// double Sigmoid(double x)
// {
//     return 1.0 / (1.0 + std::exp(-x));
// }

arma::mat ParamToGain(const arma::mat& params)
{
    arma::mat gains = params;
    gains /= arma::norm(gains, 2);
    return gains;
}

arma::mat ParamToGains(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.N;
    assert(params.n_cols >= 2 * fdn_order);

    arma::mat input_gains_arma = ParamToGain(params.cols(0, fdn_order - 1));
    arma::mat output_gains_arma = ParamToGain(params.cols(fdn_order, (2 * fdn_order) - 1));

    config.input_gains.resize(fdn_order);
    config.output_gains.resize(fdn_order);

    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        config.input_gains[i] = static_cast<float>(input_gains_arma(i));
        config.output_gains[i] = static_cast<float>(output_gains_arma(i));
    }

    const size_t start_offset = 2 * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(2 * fdn_order, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.N;
    assert(params.n_cols >= fdn_order * fdn_order);

    arma::mat M = params.cols(0, (fdn_order * fdn_order) - 1);
    M.reshape(fdn_order, fdn_order);

    arma::mat Q, R;
    arma::qr_econ(Q, R, M);
    Q = Q * arma::diagmat(arma::sign(R.diag()));

    // arma::mat test = Q.t() * Q;
    // test.print("Q^T * Q:");

    std::vector<float> matrix_coeffs(fdn_order * fdn_order);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(Q(r, c));
        }
    }

    config.matrix_info = std::move(matrix_coeffs);

    const size_t start_offset = fdn_order * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToHouseholderMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.N;
    assert(params.n_cols >= fdn_order);

    arma::vec u = params.cols(0, fdn_order - 1).as_col();
    u /= arma::norm(u, 2);

    arma::mat I = arma::eye<arma::mat>(fdn_order, fdn_order);
    arma::mat Q = I - 2.0 * (u * u.t());

    std::vector<float> matrix_coeffs(fdn_order * fdn_order);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(Q(r, c));
        }
    }

    config.matrix_info = std::move(matrix_coeffs);

    const size_t start_offset = fdn_order * fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamToCirculantMatrix(sfFDN::FDNConfig& config, const arma::mat& params)
{
    const uint32_t fdn_order = config.N;
    assert(params.n_cols >= fdn_order);

    arma::vec u = params.cols(0, fdn_order - 1).as_col();
    arma::cx_vec R = arma::fft(u);
    R = R / arma::abs(R);
    arma::rowvec v = arma::real(arma::ifft(R)).as_row();
    arma::rowvec v2 = arma::shift(arma::fliplr(v), 1);
    arma::mat C = arma::toeplitz(v2, v);

    std::vector<float> matrix_coeffs(fdn_order * fdn_order, 0.0f);
    for (uint32_t r = 0; r < fdn_order; ++r)
    {
        for (uint32_t c = 0; c < fdn_order; ++c)
        {
            matrix_coeffs[r * fdn_order + c] = static_cast<float>(C(r, c));
        }
    }

    config.matrix_info = std::move(matrix_coeffs);

    const size_t start_offset = fdn_order;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToDelays(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols == config.N);
    assert(config.delays.size() == config.N);

    constexpr uint32_t kDelayFactor = 250;

    for (size_t i = 0; i < params.n_cols; ++i)
    {
        double p = params(0, i);
        uint32_t delay_adjustement = std::tanh(p) * kDelayFactor;
        config.delays[i] += delay_adjustement;
    }

    const size_t start_offset = config.N;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(start_offset, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToAttenuationFilters(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols >= kNBands);

    arma::mat t60s = params.cols(0, kNBands - 1);

    t60s = arma::abs(t60s);
    t60s = arma::clamp(t60s, 0.1, 20.0);

    config.attenuation_t60s.resize(kNBands);
    for (uint32_t i = 0; i < kNBands; ++i)
    {
        config.attenuation_t60s[i] = static_cast<float>(t60s(0, i));
    }

    const size_t start_offset = kNBands;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(kNBands, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToTonecorrectionFilters(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols >= kNBands);

    arma::mat gains = params.cols(0, kNBands - 1);

    // gains = arma::clamp(gains, -12.0, 12.0);

    config.tc_gains.resize(kNBands);
    for (uint32_t i = 0; i < kNBands; ++i)
    {
        config.tc_gains[i] = static_cast<float>(gains(0, i));
    }

    config.tc_frequencies = {31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};

    const size_t start_offset = kNBands;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(kNBands, params.n_cols - 1);
    return leftover_params;
}

arma::mat ParamsToOverallGain(sfFDN::FDNConfig& config, const arma::mat& params)
{
    assert(params.n_cols >= 1);

    arma::mat gains = params.cols(0, 0);

    for (auto& g : config.output_gains)
    {
        g *= static_cast<float>(gains(0, 0));
    }

    const size_t start_offset = 1;
    if (params.n_cols <= start_offset)
    {
        return arma::mat(0, 0); // empty
    }

    arma::mat leftover_params = params.cols(1, params.n_cols - 1);
    return leftover_params;
}

#pragma clang diagnostic ignored "-Wunused-function"
arma::mat GetRandomInitialParams(uint32_t fdn_order,
                                 std::span<const fdn_optimization::OptimizationParamType> param_types)
{
    arma::mat params(0, 0);

    for (const auto& type : param_types)
    {
        switch (type)
        {
        case fdn_optimization::OptimizationParamType::Gains:
        {
            arma::mat input_gains(1, fdn_order, arma::fill::randn);
            arma::mat output_gains(1, fdn_order, arma::fill::randn);

            params = arma::join_horiz(params, input_gains);
            params = arma::join_horiz(params, output_gains);
        }
        break;
        case fdn_optimization::OptimizationParamType::Matrix:
        {
            arma::mat M(1, fdn_order * fdn_order, arma::fill::randn);
            params = arma::join_horiz(params, M);
        }
        break;
        case fdn_optimization::OptimizationParamType::Delays:
        {
            arma::mat delay_params(1, fdn_order, arma::fill::zeros);
            params = arma::join_horiz(params, delay_params);
        }
        break;
        case fdn_optimization::OptimizationParamType::Matrix_Householder:
            [[fallthrough]];
        case fdn_optimization::OptimizationParamType::Matrix_Circulant:
        {
            arma::mat vec_u(1, fdn_order, arma::fill::randn);
            params = arma::join_horiz(params, vec_u);
        }
        break;
        case fdn_optimization::OptimizationParamType::AttenuationFilters:
        {
            arma::mat t60s(1, kNBands, arma::fill::ones);
            params = arma::join_horiz(params, t60s);
        }
        break;
        case fdn_optimization::OptimizationParamType::TonecorrectionFilters:
        {
            arma::mat tc_gains(1, kNBands, arma::fill::zeros);
            params = arma::join_horiz(params, tc_gains);
        }
        break;
        case fdn_optimization::OptimizationParamType::OverallGain:
        {
            arma::mat overall_gain(1, 1, arma::fill::ones);
            params = arma::join_horiz(params, overall_gain);
        }
        break;
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return params;
}

arma::mat GetInitialParamsFromConfig(const sfFDN::FDNConfig& config,
                                     std::span<const fdn_optimization::OptimizationParamType> param_types)
{
    arma::mat params(0, 0);

    const uint32_t fdn_order = config.N;

    for (const auto& type : param_types)
    {
        switch (type)
        {
        case fdn_optimization::OptimizationParamType::Gains:
        {
            arma::mat input_gains(1, fdn_order);
            arma::mat output_gains(1, fdn_order);

            for (uint32_t i = 0; i < fdn_order; ++i)
            {
                input_gains(0, i) = static_cast<double>(config.input_gains[i]);
                output_gains(0, i) = static_cast<double>(config.output_gains[i]);
            }

            params = arma::join_horiz(params, input_gains);
            params = arma::join_horiz(params, output_gains);
        }
        break;
        case fdn_optimization::OptimizationParamType::Matrix:
        {
            arma::mat M(1, fdn_order * fdn_order, arma::fill::randn);

            if (std::holds_alternative<std::vector<float>>(config.matrix_info))
            {

                const auto& matrix_coeffs = std::get<std::vector<float>>(config.matrix_info);

                for (uint32_t r = 0; r < fdn_order; ++r)
                {
                    for (uint32_t c = 0; c < fdn_order; ++c)
                    {
                        M(0, r * fdn_order + c) = static_cast<double>(matrix_coeffs[r * fdn_order + c]);
                    }
                }
            }

            params = arma::join_horiz(params, M);
        }
        break;
        case fdn_optimization::OptimizationParamType::Delays:
        {
            arma::mat delay_params(1, fdn_order, arma::fill::zeros);
            for (uint32_t i = 0; i < fdn_order; ++i)
            {
                delay_params(0, i) = static_cast<double>(config.delays[i]);
            }
            params = arma::join_horiz(params, delay_params);
        }
        break;
        case fdn_optimization::OptimizationParamType::Matrix_Householder:
            [[fallthrough]];
        case fdn_optimization::OptimizationParamType::Matrix_Circulant:
        {
            // Using initial config is not supported for the moment
            arma::mat vec_u(1, fdn_order, arma::fill::randn);
            params = arma::join_horiz(params, vec_u);
        }
        break;
        case fdn_optimization::OptimizationParamType::AttenuationFilters:
        {
            arma::mat t60s(1, kNBands, arma::fill::ones);

            if (config.attenuation_t60s.size() == kNBands)
            {
                for (uint32_t i = 0; i < kNBands; ++i)
                {
                    t60s(0, i) = static_cast<double>(config.attenuation_t60s[i]);
                }
            }

            params = arma::join_horiz(params, t60s);
        }
        break;
        case fdn_optimization::OptimizationParamType::TonecorrectionFilters:
        {
            arma::mat tc_gains(1, kNBands, arma::fill::zeros);
            if (config.tc_gains.size() == kNBands)
            {
                for (uint32_t i = 0; i < kNBands; ++i)
                {
                    tc_gains(0, i) = static_cast<double>(config.tc_gains[i]);
                }
            }

            params = arma::join_horiz(params, tc_gains);
        }
        break;
        case fdn_optimization::OptimizationParamType::OverallGain:
        {
            arma::mat overall_gain(1, 1, arma::fill::ones);
            params = arma::join_horiz(params, overall_gain);
        }
        break;
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return params;
}

} // namespace

namespace fdn_optimization
{
FDNModel::FDNModel(sfFDN::FDNConfig initial_config, uint32_t ir_size,
                   std::span<const OptimizationParamType> param_types, GradientMethod gradient_method)
    : initial_config_(initial_config)
    , current_config_(initial_config)
    , ir_size_(ir_size)
    , param_types_(param_types.begin(), param_types.end())
    , gradient_method_(gradient_method)
{
    constexpr uint32_t kRandomSeed = 42;
    const uint32_t fdn_order = initial_config_.N;

    if (initial_config_.delays.size() != fdn_order)
    {
        // Following delays are from [1]
        // [1] G. D. Santo, K. Prawda, S. J. Schlecht, and V. Välimäki, “Efficient Optimization of Feedback Delay
        // Networks for Smooth Reverberation,” Aug. 28, 2024, arXiv: arXiv:2402.11216. doi: 10.48550/arXiv.2402.11216.
        if (fdn_order == 4)
        {
            initial_config_.delays = {1499, 1889, 2381, 2999};
        }
        else if (fdn_order == 6)
        {
            initial_config_.delays = {997, 1153, 1327, 1559, 1801, 2099};
        }
        else if (fdn_order == 8)
        {
            initial_config_.delays = {809, 877, 937, 1049, 1151, 1249, 1373, 1499};
        }
        else
        {
            initial_config_.delays =
                sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Random, kRandomSeed);
        }
    }

    bool optimize_filters = false;
    for (const auto& type : param_types_)
    {
        if (type == OptimizationParamType::AttenuationFilters || type == OptimizationParamType::TonecorrectionFilters)
        {
            optimize_filters = true;
            break;
        }
    }

    if (!optimize_filters)
    {
        initial_config_.attenuation_t60s = {1.f};
        initial_config_.tc_gains.clear();
    }
    current_config_ = initial_config_;

    // Check that we only have one type of matrix parameterization
    size_t matrix_param_count = 0;
    for (const auto& type : param_types_)
    {
        if (type == OptimizationParamType::Matrix || type == OptimizationParamType::Matrix_Householder ||
            type == OptimizationParamType::Matrix_Circulant)
        {
            matrix_param_count++;
        }
    }
    if (matrix_param_count > 1)
    {
        throw std::runtime_error("FDNModel only supports one type of matrix parameterization at a time.");
    }
}

void FDNModel::SetLossFunctions(const std::vector<LossFunction>& loss_functions)
{
    loss_functions_ = loss_functions;
}

uint32_t FDNModel::GetParamCount() const
{
    uint32_t count = 0;
    const uint32_t fdn_order = initial_config_.N;
    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case OptimizationParamType::Gains:
            count += 2 * fdn_order;
            break;
        case OptimizationParamType::Matrix:
            count += fdn_order * fdn_order;
            break;
        case OptimizationParamType::Delays:
            count += fdn_order;
            break;
        case OptimizationParamType::Matrix_Householder:
            [[fallthrough]];
        case OptimizationParamType::Matrix_Circulant:
            count += fdn_order;
            break;
        case OptimizationParamType::AttenuationFilters:
            count += kNBands;
            break;
        case OptimizationParamType::TonecorrectionFilters:
            count += kNBands;
            break;
        case OptimizationParamType::OverallGain:
            count += 1;
            break;
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return count;
}

void FDNModel::SetT60Estimates(std::span<const float> t60_estimates)
{
    t60_estimates_.assign(t60_estimates.begin(), t60_estimates.end());
}

arma::mat FDNModel::GetInitialParams() const
{
    // arma::arma_rng::set_seed_random();
    // arma::mat params = GetRandomInitialParams(initial_config_.N, param_types_);
    arma::mat params = GetInitialParamsFromConfig(initial_config_, param_types_);

    assert(params.n_cols == GetParamCount());

    return params;
}

void FDNModel::Setup(const arma::mat& params)
{
    current_config_ = GetFDNConfig(params);
}

std::span<const float> FDNModel::GenerateIR()
{
    if (response_buffer_.size() < ir_size_)
    {
        response_buffer_.resize(ir_size_);
    }

    if (impulse_buffer_.size() < ir_size_)
    {
        impulse_buffer_.resize(ir_size_);
    }

    std::ranges::fill(impulse_buffer_, 0.0f);
    std::ranges::fill(response_buffer_, 0.0f);

    impulse_buffer_[0] = 1.0f; // Delta impulse

    sfFDN::AudioBuffer in_buffer(impulse_buffer_);
    sfFDN::AudioBuffer out_buffer(response_buffer_);
    auto fdn = sfFDN::CreateFDNFromConfig(current_config_, kSampleRate);
    fdn->Process(in_buffer, out_buffer);

    return std::span<const float>(response_buffer_);
}

double FDNModel::Evaluate(const arma::mat& params)
{
    Setup(params);
    last_losses_.clear();
    std::span<const float> ir = GenerateIR();

    double total_loss = 0.0;
    for (const auto& loss_function : loss_functions_)
    {
        double loss = loss_function.func(ir);
        last_losses_.push_back(loss_function.weight * loss);
        total_loss += loss_function.weight * loss;
    }

    return total_loss;
}

double FDNModel::Evaluate(const arma::mat& params, const size_t i, const size_t batch_size)
{
    assert(i == 0 && batch_size == 1);
    (void)i;
    (void)batch_size;
    return Evaluate(params);
}

double FDNModel::EvaluateWithGradient(const arma::mat& x, arma::mat& g)
{
    double loss = Evaluate(x);
    switch (gradient_method_)
    {
    case GradientMethod::CentralDifferences:
        GradientCentralDifferences(x, g);
        break;
    case GradientMethod::ForwardDifferences:
        GradientForwardDifferences(x, g, loss);
        break;
    default:
        throw std::runtime_error("Unknown GradientMethod in EvaluateWithGradient");
    }

    return loss;
}

double FDNModel::EvaluateWithGradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize)
{
    assert(i == 0 && batchSize == 1);
    (void)i;
    (void)batchSize;
    return EvaluateWithGradient(x, g);
}

sfFDN::FDNConfig FDNModel::GetFDNConfig(const arma::mat& params) const
{
    arma::mat params_to_process = params;
    sfFDN::FDNConfig config = initial_config_;

    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case OptimizationParamType::Gains:
        {
            params_to_process = ParamToGains(config, params_to_process);
        }
        break;
        case OptimizationParamType::Matrix:
        {
            params_to_process = ParamToMatrix(config, params_to_process);
        }
        break;
        case OptimizationParamType::Matrix_Householder:
        {
            params_to_process = ParamToHouseholderMatrix(config, params_to_process);
        }
        break;
        case OptimizationParamType::Matrix_Circulant:
        {
            params_to_process = ParamToCirculantMatrix(config, params_to_process);
        }
        break;
        case OptimizationParamType::Delays:
        {
            params_to_process = ParamsToDelays(config, params_to_process);
        }
        break;
        case OptimizationParamType::AttenuationFilters:
        {
            params_to_process = ParamsToAttenuationFilters(config, params_to_process);
        }
        break;
        case OptimizationParamType::TonecorrectionFilters:
        {
            params_to_process = ParamsToTonecorrectionFilters(config, params_to_process);
        }
        break;
        case OptimizationParamType::OverallGain:
        {
            params_to_process = ParamsToOverallGain(config, params_to_process);
        }
        break;
        default:
            throw std::runtime_error("Unknown OptimizationParamType in Setup");
        }
    }

    return config;
}

void FDNModel::PrintFDNConfig(const arma::mat& params) const
{
    sfFDN::FDNConfig config = GetFDNConfig(params);

    arma::fvec input_gains_arma(config.input_gains.data(), config.N);
    arma::fvec output_gains_arma(config.output_gains.data(), config.N);

    std::cout << "FDN Configuration:----------------------" << std::endl;
    input_gains_arma.t().print("Input Gains:");
    output_gains_arma.t().print("Output Gains:");
    std::cout << "Delays: [";
    for (const auto& delay : config.delays)
    {
        std::cout << delay << " ";
    }
    std::cout << "]" << std::endl;

    std::vector<float> matrix_data = std::get<std::vector<float>>(config.matrix_info);
    arma::fmat matrix_data_arma(matrix_data.data(), config.N, config.N);

    matrix_data_arma.print("Feedback Matrix:");
    std::cout << "----------------------------------------" << std::endl;
}

void FDNModel::GradientCentralDifferences(const arma::mat& x, arma::mat& g)
{
    g.zeros(x.n_rows, x.n_cols);

#pragma omp parallel for
    for (int col = 0; col < static_cast<int>(x.n_cols); ++col)
    {
        arma::mat x_plus = x;
        x_plus(0, col) += gradient_delta_;
        // Creating a whole new model to avoid threading issues
        FDNModel grad_model(*this);
        double plus_value = grad_model.Evaluate(x_plus);

        arma::mat x_minus = x;
        x_minus(0, col) -= gradient_delta_;
        double minus_value = grad_model.Evaluate(x_minus);
        g(0, col) = (plus_value - minus_value) / (2 * gradient_delta_);
    }
}

void FDNModel::GradientForwardDifferences(const arma::mat& x, arma::mat& g, double current_loss)
{
    g.zeros(x.n_rows, x.n_cols);
#pragma omp parallel for
    for (int col = 0; col < static_cast<int>(x.n_cols); ++col)
    {
        arma::mat x_plus = x;
        x_plus(0, col) += gradient_delta_;
        // Creating a whole new model to avoid threading issues
        FDNModel grad_model(*this);
        double plus_value = grad_model.Evaluate(x_plus);
        g(0, col) = (plus_value - current_loss) / gradient_delta_;
    }
}

} // namespace fdn_optimization