clearvars;clf;


[init_ir, init_fs] = audioread('../optim_output/initial_ir.wav');
[opt_ir, opt_fs] = audioread('../optim_output/optimized_ir.wav');
opt_ir = opt_ir / max(abs(opt_ir));
init_ir = init_ir / max(abs(init_ir));

% start_idx = find(opt_ir ~= 0, 1);
% opt_ir = opt_ir(start_idx:end); % Trim the impulse response to start from the first non-zero sample

win_len = 1024;
ovl = 512;
ir_buf = buffer(opt_ir, win_len, ovl, 'nodelay');

win = hann(win_len);

ir_rms = rms(ir_buf .* win);
hop = win_len - ovl;
frame_index = 0:size(ir_buf,2)-1;
frame_index = frame_index*hop;

y = zeros(size(opt_ir));

ir_init_buf = buffer(init_ir, win_len ,ovl, 'nodelay');
ir_init_rms = rms(ir_init_buf .* win);


atk = 0.85;
rel = 0.999;

for n = 2:length(opt_ir)
    x = opt_ir(n)^2;
    
    if x > y(n-1)
        y(n) = atk * (y(n-1) - x) + x;
    else
        y(n) = rel * (y(n-1) - x) + x;
    end
end


figure(1);
clf;

plot(opt_ir);

hold on;

plot(frame_index, sqrt(ir_rms), "*-");
xlabel('Frame Index');
ylabel('RMS Value');
title('RMS of Impulse Response');

plot(sqrt(y));


plot(init_ir-2);
plot(frame_index, sqrt(ir_init_rms)-2, "*-");

hold off;
