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
function [u_post, v_post] = Demodulation_Sparse(u, v, p, u_g, v_g)
    if v < 10^(-15)
        u_post = u;
        v_post = 0;
        return
    end
    N = length(u);
    u_post = zeros(N, 1);
    v_post = 0;
    for i = 1 : N
        p_post = (sqrt(v) * p) / (sqrt(v) * p + (1 - p) * sqrt(v_g + v) * exp((u_g^2 * v - 2 * u_g * u(i) * v - v_g * u(i)^2) / (2 * v * (v_g + v))));
        v_g_post = (v_g * v) / (v_g + v);
        u_g_post = (v_g * u(i) + u_g * v) / (v_g + v);
        u_post(i) = p_post * u_g_post;
        v_post = v_post + (p_post * (v_g_post + u_g_post^2) - u_post(i)^2);
    end
    v_post = v_post / N;
end
