% ============================================
% Plot L_q penalty for q = 0.1, 0.5, 1
% ============================================

theta = linspace(0, 5, 1000);

q_vals = [0.1, 0.5, 1];

P = zeros(length(q_vals), length(theta));

for k = 1:length(q_vals)
    q = q_vals(k);
    P(k,:) = theta.^q;
end

% -------- Plot --------
figure;
plot(theta, P(1,:), 'LineWidth', 2); hold on;
plot(theta, P(2,:), 'LineWidth', 2);
plot(theta, P(3,:), 'LineWidth', 2);

xlabel('\theta', 'FontSize', 12);
ylabel('Penalty  \theta^q', 'FontSize', 12);
title('L_q Penalty for Different q', 'FontSize', 13);

legend('q = 0.1', 'q = 0.5', 'q = 1 (L1)', ...
       'Location', 'NorthWest');

grid on;
box on;

% -------- Save as JPG --------
set(gcf, 'Position', [100 100 800 600]);   % optional
print('Lq_penalty_plot', '-djpeg', '-r300');
