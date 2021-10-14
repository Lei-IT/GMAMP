%% Simulation main function of GMAMP and GVAMP 
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
%% Parameter Initialization
clc; clear; %close all;
rng('default')
    
P = 0.1;
iter_limit_M = 80;     % maximum iteration number of GMAMP
iter_limit = 60;       % maximum iteration number of GVAMP
simulation_times = 1;
kappa = 30;            % condition number  
N = 8192;
delta = 0.5;           % delta = M / N

M = round(delta * N);
SNR_DB = 40;
L = 3;      
u_g = 0;
v_g = 1 / P;
mu_x = P * u_g;                  % mean of x
va_x = (P - P^2) * u_g + P * v_g;% variance of x
mu_n = zeros(M, 1);
sigma_n_square = va_x ./ (10.^(0.1.*SNR_DB)); 
MSE = zeros(1, iter_limit);      % mean-square error of GVAMP
V_D = zeros(1, iter_limit);      % theoretical variance of GVAMP
MSE_M = zeros(1, iter_limit_M);  % mean-square error of GMAMP
V_M = zeros(1, iter_limit_M);    % theoretical variance of GMAMP

%% Simulation
for sim = 1 : simulation_times
    %% source
    b = binornd(1, P, N, 1);
    g = normrnd(u_g , sqrt(v_g), [N, 1]);
    x = b .* g; 
    %% Channel
    cov_mat = sigma_n_square * eye(M);       
    n = mvnrnd(mu_n', cov_mat, 1)';  % AWGN

    T = min(M,N);
    dia = kappa.^(-[0:T-1]' / T);
    dia = sqrt(N) * dia / norm(dia);
    index_ev2 = randperm(N);
    index_ev2 = index_ev2(1:T);
    index_ev2 = index_ev2';
    x_f = dct(x);
    z_tem = dia .* x_f(index_ev2); 
    index_ev1 = randperm(M);
    index_ev1 = index_ev1(1:T);
    index_ev1 = index_ev1';
    z_tem_f = z_tem(index_ev1);
    z = dct(z_tem_f);
    
    r=zeros(M,1);
    %% clip
    clip = 2;
    for i=1:M
        if z(i)<-clip
            r(i) = -clip;
        elseif z(i)>clip
            r(i) = clip;
        else
            r(i)=z(i);
        end
    end

    y = r + n; 
    %% GVAMP
    u0 = ones(N, 1) * mu_x;
    v0 = va_x;
    [MSE_r, V_D_r] = GOAMP_Sparse(x, z, u0, v0, dia, y, cov_mat, clip, P, u_g, v_g, iter_limit, index_ev1, index_ev2);
    MSE = MSE + MSE_r; 
    V_D = V_D + V_D_r;
    %% GMAMP 
    [MSE_M_r, V_M_r] = GMAMP(P, L, u_g, v_g, sigma_n_square, iter_limit_M, x, u0, v0, y, dia, index_ev1, index_ev2, clip, z);
    MSE_M = MSE_M + MSE_M_r;
    V_M = V_M + V_M_r;
end

MSE = MSE ./ simulation_times;
MSE_M = MSE_M ./ simulation_times;
V_D = V_D ./ simulation_times;
V_M = V_M ./ simulation_times;
 
save data;

%% plot
load data;
% figure;
semilogy(0:iter_limit_M-1, MSE_M, "r-"); % GMAMP       
hold on
semilogy(0:iter_limit_M-1, V_M, "ro");   % GMAMP 
hold on; 
if iter_limit_M > iter_limit
    diff = iter_limit_M - iter_limit;        % GVAMP(LMMSE)
end
semilogy(0:iter_limit_M, [va_x MSE MSE(iter_limit)*ones(1,diff)], 'b--');
hold on
semilogy(0:iter_limit_M, [va_x V_D V_D(iter_limit)*ones(1,diff)], 'bo');
%title(['[GMAMP] kappa=', num2str(kappa), ';M=', num2str(M), ';N=', num2str(N), ';SNR(dB)=', num2str(SNR_DB),';L=', num2str(L),';delta=', num2str(delta)]);
title(['delta=', num2str(delta)]);
legend('BO-GMAMP','BO-GMAMP(SE)', 'GOAMP/GVAMP','GOAMP/GVAMP(SE)');
xlabel('Number of iterations', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);