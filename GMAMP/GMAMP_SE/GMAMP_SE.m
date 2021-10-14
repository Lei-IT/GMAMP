%% GMAMP(SE)
% ----------------------------------------------------------
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
function [VSE_M, vx_reg_SE, vz_reg_SE] = GMAMP_SE(x, z, y, S, P, L, u_g, v_g, v_n, it, dia, M, N, clip)
    %% Initialization
    delta = M / N;
    lamda = dia.^2;                             % eigenvalue of AAH
    lamda_star = 0.5 * (max(lamda) + min(lamda)); 
    B = lamda_star * ones(M, 1) - lamda;        % eigenvalue of B
    b = zeros(1, it*2+2);                       % b_bar
    for i = 1 : it*2+2
        b(i) = 1 / N * sum(B.^(i-1));
    end
    w = N/M * (lamda_star * b(1: it*2+1) - b(2: it*2+2));  % trace of W
    w_bar = zeros(it, it);
    for i = 1 : it
        for j = 1 : it
            w_bar(i, j) = lamda_star * w(i+j-1) - w(i+j) - w(i) * w(j);
        end
    end
    w_tilde = zeros(it, it);
    for i = 1 : it
        for j = 1 : it
            w_tilde(i, j) = (lamda_star * w(i+j-1) - w(i+j))/delta - w(i) * w(j);
        end
    end
    w_bar2 = zeros(it, it);
    for i =1:it
        for j = 1 : it
            w_bar2(i, j) =  lamda_star^2 * w(i+j-1) - 2 * lamda_star * w(i+j) + w(i+j+1);
        end
    end
    
    % Initialization
    v_x = zeros(it, it);
    v_z = zeros(it,it);
    v_gamma = zeros(it, it);
    v_gamma(1,1) = 1e6;
    vz_gamma = zeros(it, it);
    vz_gamma(1,1) = mean(dia.^2 * 1);
    z_bb = zeros(S,1);
    
    theta_ = ones(it, it);
    epsilon = zeros(1, it); % c_bb_x
    VSE_M = zeros(1, it);
    xi = zeros(1, it);
    theta = zeros(1, it);
    beta = zeros(1, it);
    alpha = zeros(1, it);   % c_bb_z
    vs_tilde = zeros(it, it);
   
    x_hat = zeros(S, it);
    z_hat = zeros(S, it);
    ETA_x = zeros(S, it);
    ETA_z = zeros(S, it);
    
    vx_reg_SE = zeros(it,it);
    vz_reg_SE = zeros(it,it);
    
   %% iteration
    for t = 1 : it
       %% ------------ NLE psi ------------
        % Demodulation_clip    post z and vz
        [~, v_z_tp1, ETA_z, z_hat(:, t)] = ...
        NLEz_GMAMP_SE(t, v_n, z_bb, vz_gamma(1:t, 1:t), z, y, S, z_hat(:, 1:t-1), ETA_z, clip);
        v_z(1:t, t) = v_z_tp1;
        v_z(t, 1:t) = v_z(1:t, t)';
        
        vz_reg_SE(t,1:t) = v_z(t,1:t); % store before damping
        vz_reg_SE(1:t,t) = vz_reg_SE(t,1:t)';
       %% damping
        [z_hat, v_z] = Damping_SE(z_hat, v_z, L, t);

        
       %% ------------ NLE phi ------------
        % Demodulation_x     post x and v
        [v_hat, v_x_tp1, ETA_x, x_hat(:, t)] = ...
        NLEx_GMAMP_SE(t, P, u_g, v_g, v_gamma(1:t, 1:t), x, S, x_hat(:, 1:t-1), ETA_x);
        v_x(1:t, t) = v_x_tp1;
        v_x(t, 1:t) = v_x(1:t, t)';
        VSE_M(t) = v_hat;
        
        ct = 2;
        if t == it
            break
        elseif t > ct
            thres = 10^(-7);    % when we stop the algorithm
            comp = max(abs(v_hat - VSE_M(t-ct:t-1)));
            if comp <= thres
                VSE_M(t+1:it) = v_hat;
                vx_reg_SE(t:it,t:it) = vx_reg_SE(t-1,t-1);
                for i=t:it
                vx_reg_SE(i,1:t-1) = abs(vx_reg_SE(t-1,1:t-1));
                end
                vx_reg_SE(1:t-1,t:it) = vx_reg_SE(t:it,1:t-1)';
                vz_reg_SE(t+1:it,t+1:it) = vz_reg_SE(t,t);
                for i=t+1:it
                vz_reg_SE(i,1:t) = vz_reg_SE(t,1:t);
                end
                vz_reg_SE(1:t,t+1:it) = vz_reg_SE(t+1:it,1:t)';
                break
            end
        end
        
       vx_reg_SE(t,1:t) = v_x(t,1:t); % store before damping
       vx_reg_SE(1:t,t) = vx_reg_SE(t,1:t)';
       %% damping
       [x_hat, v_x] = Damping_SE(x_hat, v_x, L, t);

       
       %% ------------- MLE(SE) --------------
        [xi, theta, beta, alpha, theta_, epsilon, vs_tilde, v_gamma, vz_gamma, ETA_z, z_bb] = MLE_GMAMP_SE(v_gamma, vz_gamma, v_x, v_z, theta_, w, ...
            w_bar, w_tilde, w_bar2, xi, theta, beta, alpha, epsilon, vs_tilde, lamda_star, t, delta, ETA_z, z, S); 
    end
end

%% Damping (SE)
function [xz_hat, v_xz] = Damping_SE(xz_hat, v_xz, L, t)
    l = min(L, t);
    v_tmp = v_xz(t+1-l:t, t+1-l:t);
    tmp = (v_tmp)^(-1);
    v_ = sum(sum(tmp));
    zeta = sum(tmp, 2) / v_;
    xz_hat(:, t) = sum(zeta'.*xz_hat(:, t+1-l:t), 2);
    v_xz(t, t) = 1 / v_;
    for t_ = 1 : t-1
        v_xz(t_, t) = sum(zeta'.*v_xz(t_, t+1-l:t));
        v_xz(t, t_) = v_xz(t_, t);
    end
end

