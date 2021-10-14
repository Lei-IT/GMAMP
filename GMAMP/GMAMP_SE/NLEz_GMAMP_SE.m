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
function [v_post, v, ETA, z_hat_tp1] = NLEz_GMAMP_SE(t, v_n, z_bb, vz_gamma, z, y, S, z_hat, ETA, clip)
    v = zeros(t, 1);
    vz_bb = vz_gamma(t, t);
        
    [z_tp1, vz_p] = Declip_GMAMP_SE(y, z_bb, vz_bb, v_n, clip); % Demodulation_clip

    v_post = vz_p;
    z_hat_tp1 = (z_tp1 / vz_p - z_bb / vz_bb) ./ (1 / vz_p - 1 / vz_bb); % Orthogonalization
    z_hat = [z_hat, z_hat_tp1];
    for k = 1 : t
        v(k) = sum((z_hat(:, k) - z) .* (z_hat(:, t) - z)) / S;
    end
end