%% Parameter Initialization
clc; clear; %close all;
rng('default')
%% Simulation main function of GMAMP and GVAMP (State Evolution)
% -----------------------------------------------------------------------
% If you use this code, please cite the paper below. Thank you.
%
% [1] F. Tian, L. Liu, and X. Chen, "Generalized Memory approximate message 
%     passing," arXiv preprint arXiv:2110.06069, Oct. 2021, [Online]
%     Available: https://arxiv.org/pdf/2110.06069.pdf
%
% [2] L. Liu, S. Huang, and B. M. Kurkoski, "Memory approximate message 
%     passing," arXiv preprint arXiv:2012.10861, Dec. 2020, [Online]
%     Available: https://arxiv.org/abs/2012.10861 
%
%                                --by Feiyan Tian and Lei Liu, 2021    
% -----------------------------------------------------------------------
% Problem Model: y = clip(Ax) + n (We use FFT for A in simultaion.)
% In this program, we suppse that N >= M.
% M              -- length of vector y
% N              -- length of vector x
% x_true         -- true x to be estimated
%                -- x(i) = 0 (1-P) or x(i) ~ N(u_g, v_g) (P)
% P              -- Non-zero probability of x(i)  
% L              -- length of damping vector (min(L, t))
% u_g            -- mean of the Gaussian distribution
% v_g            -- variance of the Gaussian distribution
% v_n            -- variance of the Gaussian noise
% it             -- maximum times of iterations
% dia            -- singular value of A*AH
% index_ev1      -- index of M elements (of x) for FFT
% index_ev2      -- index of N elements (of x) for FFT
% ------------------------------------------------------------------------

P = 0.1;
iter_limit_M = 80;
iter_limit = 60;
simulation_times = 1;
kappa = 30; 
N = 8192;
delta = 0.5;

M = round(delta * N);
SNR_DB = 40;
L = 3;
u_g = 0;
v_g = 1 / P;
v_x = (P - P^2) * u_g + P * v_g;
sigma_n_square = v_x ./ (10.^(0.1.*SNR_DB));
V_SE = zeros(1, iter_limit);
V_M_SE = zeros(1, iter_limit_M);
T = min(M, N);
dia = kappa.^(-[0:T-1]' / T);
dia = sqrt(N) * dia / norm(dia);
     
%% Monte Carlo 
S = 80000;
b = binornd(1, P, S, 1);
g = normrnd(u_g , sqrt(v_g), [S, 1]);
x = b .* g; 
z = normrnd(0 , sqrt(mean(dia.^2 * v_x)), [S, 1]);

clip = 2;
clip_z = zeros(S,1);
for i=1:S
    if z(i)<-clip
        clip_z(i) = -clip;
    elseif z(i)>clip
        clip_z(i) = clip;
    else
        clip_z(i)=z(i);
    end
end
n = normrnd(0, sqrt(sigma_n_square), [S, 1]);
y = clip_z + n;
%% Main Program
for r = 1 : simulation_times
    % GOAMP
    V_SE_r = GOAMP_SE(x, z, S, N, dia, P, u_g, v_g, sigma_n_square, iter_limit, y, clip);
    V_SE = V_SE + V_SE_r;
    % GMAMP
    [VM_SE_r, vx_reg_SE, vz_reg_SE] = GMAMP_SE(x, z, y, S, P, L, u_g, v_g, sigma_n_square, iter_limit_M, dia, M, N, clip);
    V_M_SE = V_M_SE + VM_SE_r;
end
V_SE = V_SE ./ simulation_times;
V_M_SE = V_M_SE ./ simulation_times;
save ('SE_vari.mat','vx_reg_SE','vz_reg_SE');% copy this .mat into file 'GMAMP'
%% plot
v0 = (P - P^2) * u_g + P * v_g;
figure;
% GMAMP
VSE_M_plot = V_M_SE;
semilogy(0:iter_limit_M-1, VSE_M_plot, "r*");
hold on;
% GOAMP/GVAMP
len = iter_limit;
if iter_limit_M > iter_limit
    len = iter_limit_M;
    diff = iter_limit_M - iter_limit;
    V_SE = [V_SE V_SE(iter_limit)*ones(1,diff)];
end
V_SE_plot = [v0 V_SE];
semilogy(0:iter_limit_M, V_SE_plot, 'bo');
title(['[GMAMP] kappa=', num2str(kappa), ';M=', num2str(M), ';N=', num2str(N), ';SNR(dB)=', num2str(SNR_DB), ';delta=', num2str(delta)]);
legend('SE for GMAMP', 'SE for GOAMP');
xlabel('Iteration Times', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);