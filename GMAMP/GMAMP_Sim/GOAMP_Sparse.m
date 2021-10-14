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
function [MSE_m_r, VD_m_r] = GOAMP_Sparse(x, z, u0, v0, dia, y, cov_mat, clip, p, u_g, v_g, iter_N, index_ev1, index_ev2) 
    % Main process of GOAMP
    MSE_m_r = zeros(1, iter_N);
    VD_m_r = zeros(1, iter_N);
    u_dem = u0;     % x_pri
    v_dem = v0;     % vx_pri
    N = length(u0);
    M = length(y);

    % initialization of uz and vz        
    u_f = dct(u_dem);
    uz_dem_tem = dia.* u_f(index_ev2);
    uz_tem_f = uz_dem_tem(index_ev1);
    uz_dem = dct(uz_tem_f);
    vz_dem = mean(dia.^2 * v_dem);
    
    for i = 1 : iter_N
        % ======NLE==eta=====
        % Demodulator_clip
        [uz_post_d, vz_post_d] = Demodulation_clip_erfc(uz_dem, vz_dem, y, cov_mat(1,1), clip);
        % Orthogonalization
        [uz_mmse, vz_mmse] = Ortho(uz_post_d, vz_post_d, uz_dem, vz_dem);
        
        % ======LE==gamma=====
        % LMMSE
        [u_post_m, v_post_m,uz_post_m, vz_post_m] = LMMSE_FFT(u_dem, v_dem, uz_mmse, vz_mmse, dia, index_ev1, index_ev2); % FFT    
        if v_post_m < 10^(-15)
            MSE_m_r(i) = 10^(-15);
            VD_m_r(i) = 10^(-15);
            return
        end
        % Orthogonalization
        [u_mmse, v_mmse] = Ortho(u_post_m, v_post_m, u_dem, v_dem);
        [uz_dem, vz_dem] = Ortho(uz_post_m, vz_post_m, uz_mmse, vz_mmse);     
        
        % ======NLE==phi=====
        % Demodulator_x
        [u_post_d, v_post_d] = Demodulation_Sparse(u_mmse, v_mmse, p, u_g, v_g);
        if v_post_d < 10^(-15)
            MSE_m_r(i) = 10^(-15);
            VD_m_r(i) = 10^(-15);
            return
        end
        MSE_m_r(i) = sum((u_post_d - x).^2)/ N;
        VD_m_r(i) = v_post_d;
        % Orthogonalization
        [u_dem, v_dem] = Ortho(u_post_d, v_post_d, u_mmse, v_mmse);
 
    end

end