#include "session/fdn_session.h"

#include <exception>
#include <limits>
#include <utility>

namespace fdn_sandbox::session
{

FdnSession::FdnSession(sfFDN::FDNConfig initial_config)
    : draft_(initial_config)
    , slots_{initial_config, std::move(initial_config)}
{
}

sfFDN::FDNConfig& FdnSession::Draft() noexcept
{
    return draft_;
}

const sfFDN::FDNConfig& FdnSession::Draft() const noexcept
{
    return draft_;
}

sfFDN::FDN* FdnSession::Preview() noexcept
{
    return preview_.get();
}

const sfFDN::FDN* FdnSession::Preview() const noexcept
{
    return preview_.get();
}

std::expected<FdnCommit, FdnBuildError> FdnSession::Commit(float sample_rate)
{
    if (desired_generation_ == std::numeric_limits<std::uint64_t>::max())
    {
        return std::unexpected(FdnBuildError{.message = "FDN generation counter exhausted"});
    }

    try
    {
        auto realized_config = draft_;
        realized_config.sample_rate = sample_rate;
        auto realized = sfFDN::CreateFDNFromConfig(realized_config);
        if (realized == nullptr)
        {
            return std::unexpected(FdnBuildError{.message = "FDN construction returned no instance"});
        }

        realized->SetDirectGain(0.0f);
        auto analysis_fdn = realized->CloneFDN();
        auto audio_fdn = realized->CloneFDN();
        if (analysis_fdn == nullptr || audio_fdn == nullptr)
        {
            return std::unexpected(FdnBuildError{.message = "FDN cloning returned no instance"});
        }

        draft_ = std::move(realized_config);
        preview_ = std::move(realized);
        ++desired_generation_;
        return FdnCommit{
            .generation = desired_generation_,
            .analysis_fdn = std::move(analysis_fdn),
            .audio_fdn = std::move(audio_fdn),
        };
    }
    catch (const std::exception& error)
    {
        return std::unexpected(FdnBuildError{.message = error.what()});
    }
}

std::expected<FdnCommit, FdnBuildError> FdnSession::SwitchTo(FdnSlot slot, float sample_rate)
{
    if (slot == active_slot_)
    {
        return Commit(sample_rate);
    }

    auto previous_draft = draft_;
    auto previous_slots = slots_;
    const FdnSlot previous_slot = active_slot_;
    slots_[SlotIndex(active_slot_)] = draft_;
    draft_ = slots_[SlotIndex(slot)];
    active_slot_ = slot;

    auto commit = Commit(sample_rate);
    if (!commit)
    {
        draft_ = std::move(previous_draft);
        slots_ = std::move(previous_slots);
        active_slot_ = previous_slot;
    }
    return commit;
}

void FdnSession::SaveTo(FdnSlot slot)
{
    slots_[SlotIndex(slot)] = draft_;
}

const sfFDN::FDNConfig& FdnSession::ConfigForSlot(FdnSlot slot) const noexcept
{
    return slots_[SlotIndex(slot)];
}

FdnSlot FdnSession::ActiveSlot() const noexcept
{
    return active_slot_;
}

bool FdnSession::MarkAccepted(std::uint64_t generation) noexcept
{
    if (generation == 0 || generation <= accepted_generation_ || generation > desired_generation_)
    {
        return false;
    }
    accepted_generation_ = generation;
    return true;
}

bool FdnSession::ObserveActiveGeneration(std::uint64_t generation) noexcept
{
    if (generation < active_generation_ || generation > accepted_generation_)
    {
        return false;
    }
    active_generation_ = generation;
    return true;
}

std::uint64_t FdnSession::DesiredGeneration() const noexcept
{
    return desired_generation_;
}

std::uint64_t FdnSession::AcceptedGeneration() const noexcept
{
    return accepted_generation_;
}

std::uint64_t FdnSession::ActiveGeneration() const noexcept
{
    return active_generation_;
}

bool FdnSession::ShouldReportInstability(std::uint64_t generation) const noexcept
{
    return generation != 0 && accepted_generation_ != 0 && generation >= accepted_generation_;
}

std::size_t FdnSession::SlotIndex(FdnSlot slot) noexcept
{
    return slot == FdnSlot::A ? 0U : 1U;
}

} // namespace fdn_sandbox::session
