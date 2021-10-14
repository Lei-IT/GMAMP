%% GMAMP 
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
function [MSE_M, V_M] = GMAMP(P, L, u_g, v_g, v_n, it, x_true, u0, v0, y, dia, index_ev1,index_ev2, clip, z_true)
    %% Initialization
    M = length(y);
    N = length(x_true);
    delta = M / N;
    lamda = dia.^2;                             % eigenvalue of AAH
    lamda_star = 0.5 * (max(lamda) + min(lamda)); 
    B = lamda_star * ones(M, 1) - lamda;        % eigenvalue of B
    b = zeros(1, it*2+2);                       
    for i = 1 : it*2+2
        b(i) = 1 / N * sum(B.^(i-1));
    end
    w = N/M *(lamda_star * b(1: it*2+1) - b(2: it*2+2));  % trace of W
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
    x_bb = u0; 
    v_x_bb = 10e6;

    x_phi = zeros(N, it);
    v_x = zeros(it, it);
    v_x(1,1) = v0;
    
    z_psi = zeros(M, it);
    v_z = zeros(it, it);
     
    u_f = dct(u0);
    uz_dem_tem = dia.* u_f(index_ev2);
    uz_tem_f = uz_dem_tem(index_ev1);
    z_bb = dct(uz_tem_f);
    v_z_bb = mean(dia.^2 * v0); 
    
    x_bb_hat = zeros(M, 1);
    theta_ = ones(1, it);

    MSE_M = zeros(1, it);
    V_M = zeros(1, it);
    damping_flag = true;
    
    load SE_vari.mat;    %load the data saved in file 'SE for GMAMP'
    %% Iteration
    for t = 1 : it
       %% ----------- NLE psi ------------------
        [z_hat, v_z_hat] = Demodulation_clip_erfc(z_bb, v_z_bb, y, v_n, clip);         % Demodulation_clip
        z_psi(:, t) = (z_hat / v_z_hat - z_bb / v_z_bb) / (1 / v_z_hat - 1 / v_z_bb);  % Orthogonalization 
        
        % load register
        v_z(t,1:t) = vz_reg_SE(t,1:t);
        v_z(1:t,t) = v_z(t,1:t)';
        
       %% damping z
         [z_psi, v_z] = Damping_NLE(z_psi, v_z, L, t);  
         

       %% ----------- NLE phi ------------------
        [x_hat, v_x_hat] = Demodulation(x_bb, v_x_bb, P, u_g, v_g);    % Demodulation_x
        MSE_M(t) = sum((x_hat - x_true).^2) / N;
        V_M(t) = v_x_hat;
        ct = 2;
        if t == it
            break
        elseif t > ct
            thres = 10^(-7);    % when we stop the algorithm
            comp = max(abs(v_x_hat - V_M(t-ct:t-1)));
            if comp <= thres
                MSE_M(t+1:it) = MSE_M(t);
                V_M(t+1:it) = v_x_hat;
                break
            end
        end
        x_phi(:, t) = (x_hat / v_x_hat - x_bb / v_x_bb) / (1 / v_x_hat - 1 / v_x_bb);  % Orthogonalization

         
        % load register
        v_x(t,1:t) = vx_reg_SE(t,1:t);
        v_x(1:t,t) = v_x(t,1:t)';
        
       %% damping x
         [x_phi, v_x] = Damping_NLE(x_phi, v_x, L, t);


       %% ----------- MLE ------------------
        [theta_, x_bb_hat, x_bb, v_x_bb, z_bb, v_z_bb] = MLE_GMAMP(theta_, x_bb_hat, x_phi, v_x, z_psi, v_z, delta, w, ...
            w_bar, w_tilde, w_bar2, index_ev1, index_ev2, dia, lamda_star, t, M, N, x_true, z_true); 

    end
    
end

%% Damping (NLE) 
function [esti, vari] = Damping_NLE(esti, vari, L, t)
    l = min(L, t);
    
    if t > 1 && vari(t,t) > vari(t-1,t-1)
        esti(:,t) = esti(:,t-1);
        vari(t, t) = vari(t-1, t-1);
        vari(1:t-1, t) = vari(1:t-1, t-1);
        vari(t, 1:t-1) = vari(t-1, 1:t-1);
    end
    
    v_temp = vari(t+1-l:t, t+1-l:t);
    
    if min(eig(v_temp)) <= 0
        return
    else
        f = 0;
        while rcond(v_temp) < 1e-15
            f = f + 1;
            v_temp(1:l-1, 1:l-1) = vari(t+1-l-f:t-1-f, t+1-l-f:t-1-f);
            v_temp(:, l) = vari(t+1-l:t, t);
            v_temp(l, :) = v_temp(:, l)';
        end
    end
    
    temp = (v_temp)^(-1);
    v_ = sum(sum(temp));
    damping_vector = sum(temp, 2) / v_;
    vari(t, t) = 1 / v_;
    
    esti(:, t) = sum(damping_vector'.*esti(:, [t+1-l-f:t-1-f,t]), 2);
%     esti(:, t) = sum(damping_vector'.*esti(:, t+1-l:t), 2);
    for t_ = 1 : t-1
        vari(t_, t) = sum(damping_vector'.*vari(t_, [t+1-l-f:t-1-f,t]));
%         vari(t_, t) = sum(damping_vector'.*vari(t_, t+1-l:t));
        vari(t, t_) = vari(t_, t);
    end
end

%% NLE_Post_Demodulation
function [u_post, v_post] = Demodulation(u, v, P, u_g, v_g)
    EXP_MAX = 40;
    EXP_MIN = -40;
    N = length(u);
    ug = u_g * ones(N, 1);
    vg = v_g;
    %% p1
    a = sqrt((v + vg) / v);
    b = 0.5 * ((u - ug).^2 / (v + vg) - (u.^2) / v);
    %% set threshold
    b(b > EXP_MAX) = EXP_MAX;
    b(b < EXP_MIN) = EXP_MIN;
    c = (1 - P) / P;
    p1 = 1 ./ (1 + a * exp(b) * c);
    %% Gaussian addition
    v1 = (vg^(-1) + v^(-1))^(-1);
    u1 = v1 * (vg^(-1) * ug + v^(-1) * u);
    %% post u and v
    u_post = p1 .* u1;
    v_post = mean(((p1 - p1.^2) .* (u1.^2) + p1 * v1));
end

%% Ax
function Ax = A_times_x(x, index_ev1, index_ev2, dia)
    x_f = dct(x);
    Ax_tem = dia .* x_f(index_ev2);
    Ax_tem_f = Ax_tem(index_ev1);
    Ax = dct(Ax_tem_f);
end