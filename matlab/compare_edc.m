clearvars;close all;
addpath(genpath("../../FDNToolbox"));

[target_ir, target_fs] = audioread("../../rirs/py_rirs/rir_dining_room.wav");
[init_ir, init_fs] = audioread('../optim_output/initial_ir.wav');
[opt_ir, opt_fs] = audioread('../optim_output/optimized_ir.wav');


figure(1);

plot(target_ir + 1, DisplayName="Target");
hold on;
plot(opt_ir, DisplayName="Opt");
hold off;



target_edc = EDC(target_ir);
init_edc = EDC(init_ir);
opt_edc = EDC(opt_ir);

figure(2);
plot(init_edc, DisplayName="Init");
hold on;
plot(opt_edc, DisplayName="Optimized");
plot(target_edc, DisplayName="Target");
hold off;
title("Energy Decay Curve");
legend;

oct_bank = octaveFilterBank("1 octave", SampleRate=target_fs);

target_filtered = oct_bank(target_ir);
oct_bank.reset();
opt_filtered = oct_bank(opt_ir);

filter_freqs = round(oct_bank.getCenterFrequencies);

target_filtered = RemoveBeginningSilence(target_filtered);
opt_filtered = RemoveBeginningSilence(opt_filtered);

target_edr = EDC(target_filtered);
opt_edr = EDC(opt_filtered);

% target_edr = target_edr - max(target_edr);
% opt_edr = opt_edr - max(opt_edr);

figure(3);

cmap = lines(size(target_edr,2));

for n = 1:9
    subplot(3,3,n);
    target_label = sprintf("Target - %d", filter_freqs(n));
    plot(target_edr(:,n),"--",  Color=cmap(n,:), DisplayName=target_label);
    hold on;
    plot(opt_edr(:,n), Color=cmap(n,:), DisplayName=sprintf("Opt - %d", filter_freqs(n)))
    hold off;
    legend;
    ylim([-100 10]);
end

hold on;
plot(target_edr(:,10),"--",  Color=cmap(10,:), DisplayName=sprintf("Target - %d", filter_freqs(10)));
plot(opt_edr(:,10), Color=cmap(10,:), DisplayName=sprintf("Opt - %d", filter_freqs(10)))
hold off;

legend;



figure(5);
subplot(211);
pspectrum(target_ir, target_fs, "spectrogram");
title("Target");

subplot(212);
pspectrum(opt_ir, opt_fs, "spectrogram");
title("Optimized Impulse Response Spectrogram");

figure(6);
losses = readtable("../optim_output/loss_history.txt", "VariableNamingRule","preserve");
total_loss = losses{:,1};
plot(total_loss, DisplayName=losses.Properties.VariableNames{1});
hold on;
for n = 2:size(losses,2)
    plot(losses{:, n}, DisplayName=losses.Properties.VariableNames{n});
end
hold off;
grid on;
legend();


function [processed_irs] = RemoveBeginningSilence(signals)
    processed_irs = zeros(size(signals));

    for n = 1:size(signals, 2)
        % Find the first non-silent sample
        max_sample = max(abs(signals(:,n)));
        direct_index = find(signals(:,n) >= max_sample*0.25, 1);
        % Remove silence from the signal
        processed_size = size(signals,1) - direct_index+1;
        processed_irs(1:processed_size,n) = signals(direct_index:end,n);
    end

end