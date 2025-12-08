function [losses, overall_loss] = SparsityLoss(ir, window_length, ovl)

overall_loss = norm(ir,2) / norm(ir,1);

buf = buffer(ir, window_length, ovl, 'nodelay');

losses = zeros(size(buf,2),1);
for i = 1:length(losses)
    segment = buf(:,i);
    losses(i) = norm(segment, 2) / norm(segment, 1);

end