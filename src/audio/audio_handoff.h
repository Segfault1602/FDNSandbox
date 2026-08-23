#pragma once

#include "audio/audio_commands.h"
#include "audio/audio_events.h"

#include <readerwriterqueue.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace fdn_sandbox::audio
{

using RetiredAudioObject =
    std::variant<std::unique_ptr<sfFDN::FDN>, std::unique_ptr<sfFDN::PartitionedConvolver>, std::unique_ptr<AudioClip>>;

class AudioHandoff final
{
  public:
    explicit AudioHandoff(std::size_t capacity)
        : command_queue_(capacity)
        , event_queue_(capacity)
        , retirement_queue_(capacity)
    {
    }

    bool SubmitFdn(std::uint64_t generation, std::unique_ptr<sfFDN::FDN> value)
    {
        if (!accepting_commands_.load(std::memory_order_relaxed) || value == nullptr)
        {
            return false;
        }
        if (pending_fdn_)
        {
            coalesced_command_count_.fetch_add(1, std::memory_order_relaxed);
        }
        pending_fdn_ = InstallFdn{.generation = generation, .value = std::move(value)};
        FlushPendingCommands();
        return true;
    }

    bool SubmitConvolver(std::unique_ptr<sfFDN::PartitionedConvolver> value)
    {
        if (!accepting_commands_.load(std::memory_order_relaxed) || value == nullptr)
        {
            return false;
        }
        if (pending_convolver_)
        {
            coalesced_command_count_.fetch_add(1, std::memory_order_relaxed);
        }
        pending_convolver_ = InstallConvolver{.value = std::move(value)};
        FlushPendingCommands();
        return true;
    }

    bool SubmitImpulse()
    {
        if (!accepting_commands_.load(std::memory_order_relaxed))
        {
            return false;
        }
        if (pending_impulse_)
        {
            coalesced_command_count_.fetch_add(1, std::memory_order_relaxed);
        }
        pending_impulse_ = true;
        FlushPendingCommands();
        return true;
    }

    bool SubmitClip(std::unique_ptr<AudioClip> value)
    {
        if (!accepting_commands_.load(std::memory_order_relaxed) || value == nullptr || value->samples.empty())
        {
            return false;
        }
        if (pending_clip_)
        {
            coalesced_command_count_.fetch_add(1, std::memory_order_relaxed);
        }
        pending_clip_ = InstallClip{.value = std::move(value)};
        FlushPendingCommands();
        return true;
    }

    bool SubmitPlay(bool loop)
    {
        return SubmitTransport(AudioCommand{PlayClip{.loop = loop}});
    }

    bool SubmitStop()
    {
        return SubmitTransport(AudioCommand{StopClip{}});
    }

    void FlushPendingCommands()
    {
        Flush(pending_fdn_);
        Flush(pending_convolver_);
        Flush(pending_clip_);
        FlushCommand(pending_transport_);
        if (!pending_impulse_)
        {
            return;
        }

        AudioCommand command{TriggerImpulse{}};
        if (command_queue_.try_enqueue(std::move(command)))
        {
            pending_impulse_ = false;
        }
        else
        {
            command_queue_full_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    bool TryDequeueCommand(AudioCommand& command) noexcept
    {
        return command_queue_.try_dequeue(command);
    }

    bool TryRetire(std::unique_ptr<sfFDN::FDN>& value) noexcept
    {
        return TryRetireImpl(value);
    }

    bool TryRetire(std::unique_ptr<sfFDN::PartitionedConvolver>& value) noexcept
    {
        return TryRetireImpl(value);
    }

    bool TryRetire(std::unique_ptr<AudioClip>& value) noexcept
    {
        return TryRetireImpl(value);
    }

    bool TryPublishEvent(AudioEvent event) noexcept
    {
        return event_queue_.try_enqueue(event);
    }

    bool TryPopEvent(AudioEvent& event) noexcept
    {
        return event_queue_.try_dequeue(event);
    }

    std::size_t CollectRetiredObjects()
    {
        std::size_t count = 0;
        RetiredAudioObject retired;
        while (retirement_queue_.try_dequeue(retired))
        {
            retired = std::unique_ptr<sfFDN::FDN>{};
            ++count;
        }
        return count;
    }

    void StopAcceptingCommands() noexcept
    {
        accepting_commands_.store(false, std::memory_order_relaxed);
    }

    void DrainOffRealtime()
    {
        pending_fdn_.reset();
        pending_convolver_.reset();
        pending_clip_.reset();
        pending_transport_.reset();
        pending_impulse_ = false;

        AudioCommand command{TriggerImpulse{}};
        while (command_queue_.try_dequeue(command))
        {
            command = TriggerImpulse{};
        }
        CollectRetiredObjects();
    }

    std::uint64_t CommandQueueFullCount() const noexcept
    {
        return command_queue_full_count_.load(std::memory_order_relaxed);
    }

    std::uint64_t CoalescedCommandCount() const noexcept
    {
        return coalesced_command_count_.load(std::memory_order_relaxed);
    }

  private:
    bool SubmitTransport(AudioCommand command)
    {
        if (!accepting_commands_.load(std::memory_order_relaxed))
        {
            return false;
        }
        if (pending_transport_)
        {
            coalesced_command_count_.fetch_add(1, std::memory_order_relaxed);
        }
        pending_transport_ = std::move(command);
        FlushPendingCommands();
        return true;
    }

    void FlushCommand(std::optional<AudioCommand>& pending)
    {
        if (!pending)
        {
            return;
        }
        if (command_queue_.try_enqueue(std::move(*pending)))
        {
            pending.reset();
            return;
        }
        command_queue_full_count_.fetch_add(1, std::memory_order_relaxed);
    }

    template <typename Command>
    void Flush(std::optional<Command>& pending)
    {
        if (!pending)
        {
            return;
        }

        AudioCommand command{std::move(*pending)};
        if (command_queue_.try_enqueue(std::move(command)))
        {
            pending.reset();
            return;
        }

        pending = std::move(std::get<Command>(command));
        command_queue_full_count_.fetch_add(1, std::memory_order_relaxed);
    }

    template <typename Value>
    bool TryRetireImpl(std::unique_ptr<Value>& value) noexcept
    {
        if (value == nullptr)
        {
            return true;
        }

        RetiredAudioObject retired{std::move(value)};
        if (retirement_queue_.try_enqueue(std::move(retired)))
        {
            return true;
        }

        value = std::move(std::get<std::unique_ptr<Value>>(retired));
        return false;
    }

    moodycamel::ReaderWriterQueue<AudioCommand> command_queue_;
    moodycamel::ReaderWriterQueue<AudioEvent> event_queue_;
    moodycamel::ReaderWriterQueue<RetiredAudioObject> retirement_queue_;

    std::optional<InstallFdn> pending_fdn_;
    std::optional<InstallConvolver> pending_convolver_;
    std::optional<InstallClip> pending_clip_;
    std::optional<AudioCommand> pending_transport_;

    bool pending_impulse_ = false;

    std::atomic<bool> accepting_commands_ = true;
    std::atomic<std::uint64_t> command_queue_full_count_ = 0;
    std::atomic<std::uint64_t> coalesced_command_count_ = 0;
};

} // namespace fdn_sandbox::audio
