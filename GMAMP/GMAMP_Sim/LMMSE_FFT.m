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
function [u_post, v_post, uz_post, vz_post] = LMMSE_FFT(u, v, uz, vz, dia, index_ev1, index_ev2)
    % get u_post and v_post
    if v < 10^(-15)
        u_post = u;
        v_post = 0;
        uz_post = uz;
        vz_post = 0;
        return
    end
    
    M = length(uz);
    N = length(u);
    T = min(M, N);
    Dia = 1 ./ (dia.^2 /vz  + 1/v);
    v_post = (sum(Dia) + (N-T) * v) / N;                
    u_f = dct(u);        
    u_tem_f = [dia .* u_f(index_ev2); zeros(M-N,1)];
    tem_vir = zeros(M, 1);
    tem_vir(index_ev1) = idct( uz - dct(u_tem_f(index_ev1)) );
    tem_inv = 1 ./ (vz + v * [dia.^2; zeros(M-N,1)]) .* tem_vir;
    tem_inv = tem_inv(1: T); 
    tem_vir2 = zeros(N, 1);
    tem_vir2(index_ev2) = dia .* tem_inv;
    u_post = u + v * idct(tem_vir2);
    
    u_post_f = dct(u_post);
    uz_tem = dia .* u_post_f(index_ev2);
    uz_post = dct(uz_tem(index_ev1));
    vz_post = mean(dia.^2 .* Dia);

end