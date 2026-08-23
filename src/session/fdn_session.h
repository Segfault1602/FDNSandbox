#pragma once

#include <sffdn/sffdn.h>

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>

namespace fdn_sandbox::session
{

enum class FdnSlot : std::uint8_t
{
    A,
    B
};

struct FdnBuildError
{
    std::string message;
};

struct FdnCommit
{
    std::uint64_t generation;
    std::unique_ptr<sfFDN::FDN> analysis_fdn;
    std::unique_ptr<sfFDN::FDN> audio_fdn;
};

class FdnSession final
{
  public:
    explicit FdnSession(sfFDN::FDNConfig initial_config);

    sfFDN::FDNConfig& Draft() noexcept;
    const sfFDN::FDNConfig& Draft() const noexcept;

    sfFDN::FDN* Preview() noexcept;
    const sfFDN::FDN* Preview() const noexcept;

    std::expected<FdnCommit, FdnBuildError> Commit(float sample_rate);
    std::expected<FdnCommit, FdnBuildError> SwitchTo(FdnSlot slot, float sample_rate);

    void SaveTo(FdnSlot slot);
    const sfFDN::FDNConfig& ConfigForSlot(FdnSlot slot) const noexcept;
    FdnSlot ActiveSlot() const noexcept;

    bool MarkAccepted(std::uint64_t generation) noexcept;
    bool ObserveActiveGeneration(std::uint64_t generation) noexcept;
    std::uint64_t DesiredGeneration() const noexcept;
    std::uint64_t AcceptedGeneration() const noexcept;
    std::uint64_t ActiveGeneration() const noexcept;
    bool ShouldReportInstability(std::uint64_t generation) const noexcept;

  private:
    static std::size_t SlotIndex(FdnSlot slot) noexcept;

    sfFDN::FDNConfig draft_;
    std::array<sfFDN::FDNConfig, 2> slots_;
    std::unique_ptr<sfFDN::FDN> preview_;
    FdnSlot active_slot_ = FdnSlot::A;
    std::uint64_t desired_generation_ = 0;
    std::uint64_t accepted_generation_ = 0;
    std::uint64_t active_generation_ = 0;
};

} // namespace fdn_sandbox::session
