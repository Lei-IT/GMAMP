%% MLE for GMAMP
% ------------------------------------------------------------------------
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
function [theta_, z_bb_hat, x_bb, v_x_bb, z_bb, v_z_bb] = MLE_GMAMP(theta_, z_bb_hat, x_phi, v_x, z_psi, v_z, delta, w, ...
            w_bar, w_tilde, w_bar2, index_ev1, index_ev2, dia, lamda_star, t, M, N, x_true, z_true)
    p_bar = zeros(1, t);
    theta = 1 / (lamda_star + v_z(t,t) / v_x(t, t));
    theta_(1:t-1) = theta_(1:t-1) .* theta;
    p_bar(1:t-1) = theta_(1:t-1) .* fliplr(w(2:t));
    c0 = sum(p_bar(1:t-1)) / w(1);
    c1 = v_z(t,t) * w(1) + v_x(t, t) * w_tilde(1, 1) * delta;
    c2 = 0;
    for i = 1 : t-1
        c2 = c2 - theta_(i) * (v_z(t,i) * w(t+1-i) + v_x(t, i) * w_tilde(1, t+1-i) * delta);
    end
    c3 = 0;
    for i = 1 : t-1
        for j = 1 : t-1
            c3 = c3 + theta_(i) * theta_(j) * (v_z(i,j) * w(2*t+1-i-j) + ...
            v_x(i, j) * w_tilde(t+1-i, t+1-j) * delta);
        end
    end
    if t >= 2
        tmp = c1 * c0 + c2;
        xi = (c2 * c0 + c3) / tmp; 
    else
        xi = 1;
    end
    theta_(t) = xi;
    p_bar(t) = xi * w(1);
    epsilon = (xi + c0) * w(1);  % c_bb_x

    beta = xi/theta - epsilon;
    
    %-----v_x_bb
    v_x_bb = (c1 * xi^2 - 2 * c2 * xi + c3) / (delta * epsilon^2);
    
    %-----v_z_bb
    v_s_tilde = 0;
    for i = 1 : t
        for j = 1 : t
            v_s_tilde = v_s_tilde + theta_(i) * theta_(j) * (v_x(i,j) * w_bar2(t+1-i,t+1-j) + v_z(i,j) * w_bar(t+1-i,t+1-j));
        end
    end
    s_tilde_tmp = 0;
    for i = 1 : t
        s_tilde_tmp = s_tilde_tmp + theta_(i) * (lamda_star * w(t-i+1) - w(t-i+2)) * v_x(i,t);
    end
    v_s_tilde = v_s_tilde - 2 * (xi/theta) * s_tilde_tmp + xi^2/(theta^2) * w(1) * v_x(t,t);
    v_z_bb = 1 / (w(1)^(-1) + beta^2/v_s_tilde);


    %-----z_bb_hat, x_bb_hat, x_bb and z_bb
    AHx_ = AH_times_x(z_bb_hat, index_ev1, index_ev2, dia, M, N); 
    A_bracket = A_times_x(xi *x_phi(:,t) + theta * AHx_, index_ev1, index_ev2, dia);      
    z_bb_hat = theta * lamda_star * z_bb_hat + xi * z_psi(:, t) -  A_bracket;
     
    x_bb_hat = AH_times_x(z_bb_hat, index_ev1, index_ev2, dia, M, N);
    temp = 0;
    for i = 1 : t
        temp = temp + p_bar(i) * x_phi(:, i);
    end
    x_bb = 1 / epsilon * ( N/M * x_bb_hat + temp);  

    
    temp = 0;
    for i = 1 : t
        temp = temp -  p_bar(i) * z_psi(:, i);
    end
    z_bb = A_times_x(x_bb_hat + xi/theta * x_phi(:,t), index_ev1, index_ev2, dia) + temp; 

    alpha = beta * w(1) / ( beta^2 * w(1) + v_s_tilde);  % c_bb_z
    z_bb = alpha * z_bb; 
    
end

%% Ax
function Ax = A_times_x(x, index_ev1, index_ev2, dia)
    x_f = dct(x);
    Ax_tem = dia .* x_f(index_ev2);
    Ax_tem_f = Ax_tem(index_ev1);
    Ax = dct(Ax_tem_f);
end

%% AHx
function AHx = AH_times_x(x, index_ev1, index_ev2, dia, M, N)
    tmp = zeros(M, 1);
    tmp(index_ev1) = idct(x);
    tmp_f = zeros(N,1);
    tmp_f(index_ev2) = dia .* tmp;
    AHx = idct(tmp_f);
end