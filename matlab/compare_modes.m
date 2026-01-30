clearvars;
close all;
addpath(genpath("../../FDNToolbox"));

fs = 48000;
irLen = 48000;

direct = zeros(1,1);
[A_init, B_init, C_init, m_init] = ReadFDNConfig("../optim_output/initial_fdn_config.txt");
init_ir_gen = dss2impz(irLen, m_init, A_init, B_init', C_init, zeros(1,1));

[A_opt, B_opt, C_opt, m_opt] = ReadFDNConfig("../optim_output/colorless_fdn_config.txt");

N = length(B_init);

gainPerSample = db2mag(RT602slope(10, 48000));
gainPerSamples = (gainPerSample.^m_init).';

absorption.a = zeros(N,1,2);
absorption.a(:,1,1) = 1;
absorption.b = zeros(N,1,2);
absorption.b(:,1,1) = gainPerSamples;
absorption.b(:,1,2) = 0;
zAbsorption = zTF(absorption.b,absorption.a,'isDiagonal',true);

% compute
% [MinCurve,MaxCurve,w] = poleBoundaries(m_init, absorption, A_init);
[res, pol, directTerm, isConjugatePolePair] = dss2pr(m_init, A_init, B_init, C_init, direct, 'absorptionFilters', zAbsorption);

% [MinCurve_opt,MaxCurve_opt,w_opt] = poleBoundaries(m_opt, absorption, A_opt);
[res_opt, pol_opt, directTerm_opt, isConjugatePolePair_opt] = dss2pr(m_opt, A_opt, B_opt, C_opt, direct, 'absorptionFilters', zAbsorption);

% plot
figure(1); grid on;

plot(angle(pol), mag2db(abs(pol)),'.', DisplayName="Init");
% plot(w,slope2RT60(mag2db(MinCurve),fs),'LineWidth',3, DisplayName="Init min curve");
% plot(w,slope2RT60(mag2db(MaxCurve),fs),'LineWidth',3, DisplayName="Init max curve");
hold on;
plot(angle(pol_opt), mag2db(abs(pol_opt)),'.', DisplayName="Optim");
% plot(w,slope2RT60(mag2db(MinCurve_opt),fs),'LineWidth',3, DisplayName="Optim min curve");
% plot(w,slope2RT60(mag2db(MaxCurve_opt),fs),'LineWidth',3, DisplayName="Optim max curve");

hold off;
legend();
% legend({'Poles','Minimum Boundary', 'Maximum Boundary'});
xlabel('Pole Angle [rad]')
ylabel('Pole mag [db]')

figure(2); grid on;

plot(angle(pol), mag2db(abs(res)),'.', DisplayName="Init");
% plot(w,slope2RT60(mag2db(MinCurve),fs),'LineWidth',3, DisplayName="Init min curve");
% plot(w,slope2RT60(mag2db(MaxCurve),fs),'LineWidth',3, DisplayName="Init max curve");

hold on;
plot(angle(pol_opt), mag2db(abs(res_opt)),'.', DisplayName="Optim");
% plot(w,slope2RT60(mag2db(MinCurve_opt),fs),'LineWidth',3, DisplayName="Optim min curve");
% plot(w,slope2RT60(mag2db(MaxCurve_opt),fs),'LineWidth',3, DisplayName="Optim max curve");

hold off;
legend();
% legend({'Poles','Minimum Boundary', 'Maximum Boundary'});
xlabel('Res Angle [rad]')
ylabel('Res mag [db]')

figure(3);
h1 = histogram(mag2db(abs(res)), DisplayName="Initial FDN");
title("Residue");

hold on;
h2 = histogram(mag2db(abs(res_opt)), DisplayName="Optimized FDN");
hold off;
legend();

% h1.Normalization = 'probability';
h1.BinWidth = 1;
% h2.Normalization = 'probability';
h2.BinWidth = 1;