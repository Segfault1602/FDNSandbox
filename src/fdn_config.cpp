#include "fdn_config.h"

#include "settings.h"
#include "sffdn/audio_processor.h"
#include "sffdn/filter_design.h"
#include "sffdn/sffdn.h"

#include <fstream>

#include <nlohmann/json.hpp>

namespace
{
class MatrixVisitor
{
  public:
    MatrixVisitor(sfFDN::FDN* fdn)
        : fdn_(fdn)
    {
    }

    void operator()(const sfFDN::CascadedFeedbackMatrixInfo& matrix_info) const
    {
        auto filter_matrix = sfFDN::MakeFilterFeedbackMatrix(matrix_info);
        fdn_->SetFeedbackMatrix(std::move(filter_matrix));
    }

    void operator()(const std::vector<float>& matrix_info) const
    {
        auto scalar_matrix = std::make_unique<sfFDN::ScalarFeedbackMatrix>(fdn_->GetN(), matrix_info);
        fdn_->SetFeedbackMatrix(std::move(scalar_matrix));
    }

  private:
    sfFDN::FDN* fdn_;
};

std::unique_ptr<sfFDN::AudioProcessor> CreateInputGainsFromConfig(const FDNConfig& config)
{
    auto input_gains =
        std::make_unique<sfFDN::ParallelGains>(sfFDN::ParallelGainsMode::Multiplexed, config.input_gains);

    if (!config.use_extra_delays && config.schroeder_allpass_delays.empty())
    {
        return input_gains;
    }
    auto chain_processor = std::make_unique<sfFDN::AudioProcessorChain>(Settings::Instance().BlockSize());
    chain_processor->AddProcessor(std::move(input_gains));

    if (config.use_extra_delays && config.input_stage_delays.size() > 0)
    {
        assert(config.input_stage_delays.size() == config.N);

        auto delaybank = std::make_unique<sfFDN::DelayBank>();
        delaybank->SetDelays(config.input_stage_delays, 128);

        chain_processor->AddProcessor(std::move(delaybank));
    }

    if (config.schroeder_allpass_delays.size() > 0)
    {
        assert(config.schroeder_allpass_delays.size() % config.N == 0);
        assert(config.schroeder_allpass_gains.size() == config.N);

        const uint32_t order = config.schroeder_allpass_delays.size() / config.N;

        auto schroeder_allpass = std::make_unique<sfFDN::ParallelSchroederAllpassSection>(config.N, order);
        schroeder_allpass->SetDelays(config.schroeder_allpass_delays);
        schroeder_allpass->SetGains(config.schroeder_allpass_gains);

        chain_processor->AddProcessor(std::move(schroeder_allpass));
    }

    return chain_processor;
}

} // namespace

void to_json(nlohmann::json& j, const FDNConfig& p)
{
    j = nlohmann::json{
        {"N", p.N},
        {"transposed", p.transposed},
        {"input_gains", p.input_gains},
        {"output_gains", p.output_gains},
        {"delays", p.delays},
        //    {"matrix_info", p.matrix_info},
        {"attenuation_t60s", p.attenuation_t60s},
    };

    std::visit(
        [&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::vector<float>>)
            {
                j["scalar_matrix"] = arg;
            }
            else if constexpr (std::is_same_v<T, sfFDN::CascadedFeedbackMatrixInfo>)
            {
                j["filter_matrix"] = {
                    {"N", arg.N},
                    {"num_stages", arg.K},
                    {"delays", arg.delays},
                    {"matrices", arg.matrices},
                };
            }
        },
        p.matrix_info);

    if (p.tc_gains.size() > 0 && p.tc_frequencies.size() > 0)
    {
        assert(p.tc_gains.size() == p.tc_frequencies.size());
        j["tc_gains"] = p.tc_gains;
        j["tc_frequencies"] = p.tc_frequencies;
    }

    if (p.input_stage_delays.size() > 0)
    {
        j["input_stage_delays"] = p.input_stage_delays;
    }

    if (p.schroeder_allpass_delays.size() > 0 && p.schroeder_allpass_gains.size() > 0)
    {
        assert(p.schroeder_allpass_delays.size() == p.schroeder_allpass_gains.size());
        j["schroeder_allpass_delays"] = p.schroeder_allpass_delays;
        j["schroeder_allpass_gains"] = p.schroeder_allpass_gains;
    }
}

void from_json(const nlohmann::json& j, FDNConfig& p)
{
    j.at("N").get_to(p.N);
    j.at("transposed").get_to(p.transposed);
    j.at("input_gains").get_to(p.input_gains);
    j.at("output_gains").get_to(p.output_gains);
    j.at("delays").get_to(p.delays);
    j.at("attenuation_t60s").get_to(p.attenuation_t60s);

    if (j.contains("scalar_matrix"))
    {
        std::vector<float> matrix;
        j.at("scalar_matrix").get_to(matrix);
        p.matrix_info = std::move(matrix);
    }
    else if (j.contains("filter_matrix"))
    {
        sfFDN::CascadedFeedbackMatrixInfo matrix_info;
        j.at("filter_matrix").at("N").get_to(matrix_info.N);
        j.at("filter_matrix").at("num_stages").get_to(matrix_info.K);
        j.at("filter_matrix").at("delays").get_to(matrix_info.delays);
        j.at("filter_matrix").at("matrices").get_to(matrix_info.matrices);

        p.matrix_info = matrix_info;
    }
    else
    {
        throw std::runtime_error("No valid matrix info found in JSON");
    }

    if (j.contains("tc_gains"))
    {
        assert(j.contains("tc_frequencies"));
        j.at("tc_gains").get_to(p.tc_gains);
        j.at("tc_frequencies").get_to(p.tc_frequencies);

        assert(p.tc_gains.size() == p.tc_frequencies.size());
    }

    if (j.contains("input_stage_delays"))
    {
        j.at("input_stage_delays").get_to(p.input_stage_delays);
        assert(p.input_stage_delays.size() == p.N);
    }

    if (j.contains("schroeder_allpass_delays"))
    {
        assert(j.contains("schroeder_allpass_gains"));
        j.at("schroeder_allpass_delays").get_to(p.schroeder_allpass_delays);
        j.at("schroeder_allpass_gains").get_to(p.schroeder_allpass_gains);
        assert(p.schroeder_allpass_delays.size() == p.schroeder_allpass_gains.size());
    }
}

FDNConfig FDNConfig::LoadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Failed to open file");
    }

    nlohmann::json j;
    file >> j;

    return j.template get<FDNConfig>();
}

void FDNConfig::SaveToFile(const std::string& filename, const FDNConfig& config)
{
    nlohmann::json j;
    to_json(j, config);

    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Failed to open file for writing");
    }

    file << j.dump(4); // Pretty print with 4 spaces indentation
}

std::unique_ptr<sfFDN::FDN> CreateFDNFromConfig(const FDNConfig& config, uint32_t samplerate)
{
    auto fdn = std::make_unique<sfFDN::FDN>(config.N, Settings::Instance().BlockSize());

    fdn->SetTranspose(config.transposed);
    fdn->SetInputGains(CreateInputGainsFromConfig(config));
    fdn->SetOutputGains(config.output_gains);
    fdn->SetDelays(config.delays);

    std::visit(MatrixVisitor(fdn.get()), config.matrix_info);

    auto filter_bank = sfFDN::CreateAttenuationFilterBank(config.attenuation_t60s, config.delays, samplerate);

    fdn->SetFilterBank(std::move(filter_bank));

    if (config.tc_gains.size() > 0)
    {
        assert(config.tc_gains.size() == 10);
        std::vector<float> tc_sos = sfFDN::DesignGraphicEQ(config.tc_gains, config.tc_frequencies, samplerate);
        std::unique_ptr<sfFDN::CascadedBiquads> tc_filter = std::make_unique<sfFDN::CascadedBiquads>();
        const size_t num_stages = tc_sos.size() / 6;
        tc_filter->SetCoefficients(num_stages, tc_sos);
        fdn->SetTCFilter(std::move(tc_filter));
    }

    return fdn;
}