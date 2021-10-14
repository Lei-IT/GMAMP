%% GOAMP(SE)
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
function VSE_m_r = GOAMP_SE(x, z, S, N, dia, P, u_g, v_g, v_n, iter_limit, y, clip)
    VSE_m_r = zeros(1, iter_limit);
    v_dem = (P - P^2) * u_g + P * v_g;
    vz_dem = mean(dia.^2 * v_dem);
    for i = 1 : iter_limit
        % NLD--psi
        vz_nle_post = GOAMP_Declip_SE(clip, z, vz_dem, y, v_n); % clip
        vz_mmse = 1 / (1 / vz_nle_post - 1 / vz_dem);
        % LD--gamma
        [v_le_post,vz_le_post] = GOAMP_LE_SE(N, v_dem, dia, vz_mmse);
        v_mmse = 1 / (1 / v_le_post - 1 / v_dem);
        vz_dem = 1 / (1 / vz_le_post - 1 / vz_mmse);
        % NLD--phi
        v_nle_post = GOAMP_Demodulation_SE(x, S, v_mmse, P, u_g, v_g);
        VSE_m_r(i) = v_nle_post;
        v_dem = 1 / (1 / v_nle_post - 1 / v_mmse);
    end
end

%% LE_SE
function [v_post, vz_post] = GOAMP_LE_SE(N, v, dia, v_z)
    Dia = 1 ./ (v_z ./ dia.^2 + v);
    v_post = v - v^2 * sum(Dia) / N;
    Dia = 1 ./ (dia.^2 /v_z  + 1/v);
    vz_post = mean(dia.^2 .* Dia);
end

%% DEM_SE
function v_post = GOAMP_Demodulation_SE(x, S, v, P, u_g, v_g)
    EXP_MAX = 50;
    EXP_MIN = -50;
    % Monte Carlo
    n = normrnd(0 , sqrt(v), [S, 1]);
    u = x + n;
    % Demodulator:
    ug = u_g * ones(S, 1);
    vg = v_g;
    a = sqrt((v + vg) / v);
    b = 0.5 * ((u - ug).^2 / (v + vg) - (u.^2) / v);
    b(b > EXP_MAX) = EXP_MAX;
    b(b < EXP_MIN) = EXP_MIN;
    c = (1 - P) / P;
    p1 = 1 ./ (1 + a * exp(b) * c);
    v1 = (vg^(-1) + v^(-1))^(-1);
    u1 = v1 * (vg^(-1) * ug + v^(-1) * u);
    x_post = p1 .* u1;
    v_post = mean((x_post - x).^2);
end
