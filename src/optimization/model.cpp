#include "model.h"

#include <armadillo>
#include <sffdn/sffdn.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <omp.h>

namespace
{
constexpr uint32_t kDefaultFdnBlockSize = 128;
constexpr uint32_t kSampleRate = 48000;

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

} // namespace

namespace fdn_optimization
{
FDNModel::FDNModel(uint32_t fdn_order, uint32_t ir_size, std::span<const uint32_t> delays,
                   std::span<const OptimizationParamType> param_types, GradientMethod gradient_method)
    : fdn_(std::make_unique<sfFDN::FDN>(fdn_order, kDefaultFdnBlockSize, false))
    , ir_size_(ir_size)
    , param_types_(param_types.begin(), param_types.end())
    , delays_(delays.begin(), delays.end())
    , gradient_method_(gradient_method)
{
    constexpr uint32_t kRandomSeed = 42;
    matrix_coeffs_ = sfFDN::GenerateMatrix(fdn_order, sfFDN::ScalarMatrixType::Random, kRandomSeed);
    auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_order);
    feedback_matrix->SetMatrix(matrix_coeffs_);
    fdn_->SetFeedbackMatrix(std::move(feedback_matrix));
    fdn_->SetDirectGain(0.0f);

    if (delays_.size() != fdn_order)
    {
        // Following delays are from [1]
        // [1] G. D. Santo, K. Prawda, S. J. Schlecht, and V. Välimäki, “Efficient Optimization of Feedback Delay
        // Networks for Smooth Reverberation,” Aug. 28, 2024, arXiv: arXiv:2402.11216. doi: 10.48550/arXiv.2402.11216.
        if (fdn_order == 4)
        {
            delays_ = {1499, 1889, 2381, 2999};
        }
        else if (fdn_order == 6)
        {
            delays_ = {997, 1153, 1327, 1559, 1801, 2099};
        }
        else if (fdn_order == 8)
        {
            delays_ = {809, 877, 937, 1049, 1151, 1249, 1373, 1499};
        }
        else
        {
            delays_ = sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Random, kRandomSeed);
        }
    }
    fdn_->SetDelays(delays_);

    constexpr std::array<float, 1> t60s = {{1.0f}};
    auto attenuation_filter = sfFDN::CreateAttenuationFilterBank(t60s, delays_, kSampleRate);
    fdn_->SetFilterBank(std::move(attenuation_filter));

    // Default loss functions
    std::vector<LossFunction> default_loss_functions;
    LossFunction spectral_flatness_loss;
    spectral_flatness_loss.func = [&](std::span<const float> signal) -> double {
        double spectral_flatness = SpectralFlatnessLoss(signal);
        return std::abs(0.5575f - spectral_flatness);
    };
    spectral_flatness_loss.weight = 1.0f;
    spectral_flatness_loss.name = "Spectral Flatness Loss";
    default_loss_functions.push_back(spectral_flatness_loss);

    // LossFunction mixing_time_loss;
    // mixing_time_loss.func = [&](std::span<const float> signal) -> double {
    //     double mixing_time = MixingTimeLoss(signal, kSampleRate, 0.1f);
    //     return std::clamp(mixing_time, 0.0, 1.0);
    // };
    // mixing_time_loss.weight = 0.5f;
    // mixing_time_loss.name = "Mixing Time Loss";
    // default_loss_functions.push_back(mixing_time_loss);

    LossFunction rms_loss;
    rms_loss.func = [&](std::span<const float> signal) -> double {
        return PowerEnvelopeLoss(signal, 1024, 512, kSampleRate);
    };
    rms_loss.weight = 1.0f;
    rms_loss.name = "Power Envelope Loss";
    default_loss_functions.push_back(rms_loss);

    LossFunction sparsity_loss;
    sparsity_loss.func = [&](std::span<const float> signal) -> double {
        double sparsity = SparsityLoss(signal.subspan(0, 4096));
        return sparsity;
    };
    sparsity_loss.weight = 10.f;
    sparsity_loss.name = "Sparsity Loss";
    default_loss_functions.push_back(sparsity_loss);

    SetLossFunctions(default_loss_functions);

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

FDNModel::FDNModel(const FDNModel& other)
    : fdn_(other.fdn_->CloneFDN())
    , ir_size_(other.ir_size_)
    , impulse_buffer_(other.impulse_buffer_)
    , response_buffer_(other.response_buffer_)
    , loss_functions_(other.loss_functions_)
    , matrix_coeffs_(other.matrix_coeffs_)
    , param_types_(other.param_types_)
    , delays_(other.delays_)
    , gradient_delta_(other.gradient_delta_)
    , gradient_method_(other.gradient_method_)
{
}

FDNModel& FDNModel::operator=(const FDNModel& other)
{
    if (this != &other)
    {
        fdn_ = other.fdn_->CloneFDN();
        ir_size_ = other.ir_size_;
        impulse_buffer_ = other.impulse_buffer_;
        response_buffer_ = other.response_buffer_;
        loss_functions_ = other.loss_functions_;
        matrix_coeffs_ = other.matrix_coeffs_;
        param_types_ = other.param_types_;
        delays_ = other.delays_;
        gradient_delta_ = other.gradient_delta_;
        gradient_method_ = other.gradient_method_;
    }
    return *this;
}

void FDNModel::SetLossFunctions(const std::vector<LossFunction>& loss_functions)
{
    loss_functions_ = loss_functions;
}

uint32_t FDNModel::GetParamCount() const
{
    uint32_t count = 0;
    const uint32_t fdn_order = fdn_->GetOrder();
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
        default:
            throw std::runtime_error("Unknown ParamType in GetParamCount");
        }
    }
    return count;
}

arma::mat FDNModel::GetInitialParams() const
{
    arma::arma_rng::set_seed_random();
    arma::mat params(1, GetParamCount(), arma::fill::randn);

    if (param_types_.back() == OptimizationParamType::Delays)
    {
        const uint32_t fdn_order = fdn_->GetOrder();
        const size_t delay_param_start = params.n_cols - fdn_order;
        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            // Initialize delay params to zero adjustment
            params(0, delay_param_start + i) = 0.0;
        }
    }

    return params;
}

void FDNModel::Setup(const arma::mat& params)
{
    arma::mat params_to_process = params;
    sfFDN::FDNConfig config;
    config.N = fdn_->GetOrder();
    config.delays = delays_;
    for (const auto& type : param_types_)
    {
        switch (type)
        {
        case OptimizationParamType::Gains:
        {
            params_to_process = ParamToGains(config, params_to_process);
            fdn_->SetInputGains(config.input_gains);
            fdn_->SetOutputGains(config.output_gains);
        }
        break;
        case OptimizationParamType::Matrix:
        {
            params_to_process = ParamToMatrix(config, params_to_process);
            auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_->GetOrder());
            feedback_matrix->SetMatrix(std::get<std::vector<float>>(config.matrix_info));
            fdn_->SetFeedbackMatrix(std::move(feedback_matrix));
        }
        break;
        case OptimizationParamType::Matrix_Householder:
        {
            params_to_process = ParamToHouseholderMatrix(config, params_to_process);
            auto feedback_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_->GetOrder());
            feedback_matrix->SetMatrix(std::get<std::vector<float>>(config.matrix_info));
            fdn_->SetFeedbackMatrix(std::move(feedback_matrix));
        }
        break;
        case OptimizationParamType::Delays:
        {
            params_to_process = ParamsToDelays(config, params_to_process);
            fdn_->SetDelays(config.delays);
        }
        break;
        default:
            throw std::runtime_error("Unknown ParamType in Setup");
        }
    }
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
    fdn_->Clear();
    fdn_->Process(in_buffer, out_buffer);

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
        last_losses_.push_back(loss);
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
    sfFDN::FDNConfig config;
    config.N = fdn_->GetOrder();
    config.delays = delays_;
    config.matrix_info = matrix_coeffs_;
    config.attenuation_t60s = {1.f};

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
        case OptimizationParamType::Delays:
        {
            params_to_process = ParamsToDelays(config, params_to_process);
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
        FDNModel grad_model(fdn_->GetOrder(), ir_size_, delays_, param_types_);
        grad_model.SetGradientDelta(gradient_delta_);
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
        FDNModel grad_model(fdn_->GetOrder(), ir_size_, delays_, param_types_);
        grad_model.SetGradientDelta(gradient_delta_);
        double plus_value = grad_model.Evaluate(x_plus);
        g(0, col) = (plus_value - current_loss) / gradient_delta_;
    }
}

} // namespace fdn_optimization