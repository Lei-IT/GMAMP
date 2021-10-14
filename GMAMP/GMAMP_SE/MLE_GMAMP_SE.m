%% MLE for GMAMP(SE)
% ----------------------------------------------------------
% If you use this code, please quote our paper. Thank you.
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
% ----------------------------------------------------------
function [xi, theta, beta, alpha, theta_, epsilon, vs_tilde, v_gamma, vz_gamma, ETA, z_bb] = MLE_GMAMP_SE(v_gamma, vz_gamma, v_x, v_z, theta_, w, ...
        w_bar, w_tilde, w_bar2, xi, theta, beta, alpha, epsilon, vs_tilde, lamda_star, t, delta, ETA, z, S)
    theta(t) = 1 / (lamda_star + v_z(t,t) / v_x(t, t));
    if t >= 2
        theta_(t, 1:t-1) = theta_(t-1, 1:t-1) .* theta(t);
    end
    tmp = theta_(t, 1:t-1) .* fliplr(w(2:t));
    c0 = sum(tmp) / w(1);
    c1 = v_z(t,t) * w(1) + v_x(t, t) * w_tilde(1, 1) * delta;
    c2 = 0;
    for i = 1 : t-1
        c2 = c2 - theta_(t, i) * (v_z(t,i) * w(t+1-i) + v_x(t, i) * w_tilde(1, t+1-i) * delta);
    end
    c3 = 0;
    for i = 1 : t-1
        for j = 1 : t-1
            c3 = c3 + theta_(t, i) * theta_(t, j) * (v_z(i,j) * w(2*t+1-i-j) + ...
            v_x(i, j) * w_tilde(t+1-i, t+1-j) * delta);
        end
    end
    if t >= 2
        tmp = c1 * c0 + c2;
        xi(t) = (c2 * c0 + c3) / tmp;
    else
        xi(t) = 1;
    end
    theta_(t, t) = xi(t);
    epsilon(t) = (xi(t) + c0) * w(1); % c_bb_x
    
    beta(t) = xi(t)/theta(t) - epsilon(t);
    
    %-----v_x_bb
    v_gamma(t+1, t+1) = (c1 * xi(t)^2 - 2 * c2 * xi(t) + c3) / (delta * epsilon(t)^2);
    for t_ = 1 : t-1 
        tmp = 0;
        for i = 1 : t
            for j = 1 : t_
                tmp = tmp + theta_(t, i) * theta_(t_, j) * (v_z(i,j) * w(t+t_-i-j+1) / delta ...
                     + v_x(i, j) * w_tilde(t+1-i, t_+1-j));
            end
        end
        v_gamma(t+1, t_+1) = tmp / (epsilon(t) * epsilon(t_));
        v_gamma(t_+1, t+1) = v_gamma(t+1, t_+1);
    end
    
    %-----vs_tilde
    for t_ = 1 : t
        for i = 1 : t
            for j = 1 : t_
                vs_tilde(t, t_) = vs_tilde(t, t_) + theta_(t, i) * theta_(t_, j) * ( v_x(i,j) * w_bar2(t+1-i,t_+1-j) + v_z(i,j) * w_bar(t+1-i,t_+1-j) );
            end
        end
        s_tmp1 = 0;
        for i = 1 : t
            s_tmp1 = s_tmp1 + theta_(t, i) * (lamda_star * w(t-i+1) - w(t-i+2)) * v_x(i,t_);
        end
        s_tmp2 = 0;
        for j = 1 : t_
            s_tmp2 = s_tmp2 + theta_(t_, j) * (lamda_star * w(t_-j+1) - w(t_-j+2)) * v_x(t,j);
        end
        vs_tilde(t, t_) = vs_tilde(t, t_) - xi(t_)/theta(t_) * s_tmp1 - xi(t)/theta(t) * s_tmp2 + xi(t) * xi(t_) / ( theta(t)*theta(t_) ) * w(1) * v_x(t,t_);
        vs_tilde(t_, t) = vs_tilde(t, t_);
    end  
    
    %-----v_z_bb
    vz_gamma(t+1,t+1) = 1 / (w(1)^(-1) + beta(t)^2 / vs_tilde(t, t));
        
    alpha(t) = beta(t) * w(1) / ( beta(t)^2 * w(1) + vs_tilde(t, t) ); % c_bb_z
    
    for t_ = 1 : t-1
        vz_gamma(t+1, t_+1) = (alpha(t) * beta(t) - 1) * (alpha(t_) * beta(t_) - 1) * w(1) + alpha(t) * alpha(t_) * vs_tilde(t, t_);
        vz_gamma(t_+1, t+1) = vz_gamma(t+1, t_+1);
    end
    
    %-----s_tilde
    if t == 1
        s_tilde = normrnd(0, sqrt(vs_tilde(t,t)), [S, 1]);
    else
        alpha_z = vs_tilde(1:t-1, 1:t-1) \ vs_tilde(1:t-1, t);
        v_gt = vs_tilde(t, t) - vs_tilde(t, 1:t-1) * alpha_z;
        g_t = normrnd(0, sqrt(v_gt), [S, 1]);
        s_tilde = (alpha_z' * ETA(:,1:t-1)')' + g_t;
    end
    ETA(:,t) = s_tilde;
    %------z_bb 
    z_bb = alpha(t) * ( beta(t) * z + s_tilde );
end
