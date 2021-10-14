%% NLE_x for GMAMP(SE)
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
function [v_post, v, ETA, x_hat_tp1] = NLEx_GMAMP_SE(t, P, u_g, v_g, v_gamma, x, S, x_hat, ETA)
    v = zeros(t, 1); 
    vx_bb = v_gamma(t, t);
    if t == 1 || t == 2
        n = normrnd(0, sqrt(vx_bb), [S, 1]);
    else
        alpha_x = v_gamma(2:t-1, 2:t-1) \ v_gamma(2:t-1, t);
        v_gt = v_gamma(t, t) - v_gamma(t, 2:t-1) * alpha_x;
        g_t = normrnd(0, sqrt(v_gt), [S, 1]);
        n = (alpha_x' * ETA(:,1:t-2)')' + g_t;
    end
    if t >= 2
        ETA(:,t-1) = n;
    end
    x_bb = x + n;     
    [x_tp1, v_p] = Demodulation(x_bb, vx_bb, P, u_g, v_g); % Demodulation_x
    v_post = v_p;
    x_hat_tp1 = (x_tp1 / v_p - x_bb / vx_bb) ./ (1 / v_p - 1 / vx_bb);
    x_hat = [x_hat, x_hat_tp1];
    for k = 1 : t 
        v(k) = sum((x_hat(:, k) - x) .* (x_hat(:, t) - x)) / S;
    end
end
%% DEM for Sparse
function [u_post, v_post] = Demodulation(u, v, P, u_g, v_g)
    EXP_MAX = 40;
    EXP_MIN = -40;
    N = length(u);
    ug = u_g * ones(N, 1);
    vg = v_g;
    % p1
    a = sqrt((v + vg) / v);
    b = 0.5 * ((u - ug).^2 / (v + vg) - (u.^2) / v);
    b(b > EXP_MAX) = EXP_MAX;
    b(b < EXP_MIN) = EXP_MIN;
    c = (1 - P) / P;
    p1 = 1 ./ (1 + a * exp(b) * c);
    v1 = (vg^(-1) + v^(-1))^(-1);
    u1 = v1 * (vg^(-1) * ug + v^(-1) * u);
    u_post = p1 .* u1;
    v_post = mean(((p1 - p1.^2) .* (u1.^2) + p1 * v1));
end