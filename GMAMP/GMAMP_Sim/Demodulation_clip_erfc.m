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
function [uz_post, vz_post] = Demodulation_clip_erfc(uz, vz, y, vn, lambda)
%% To calculate the posterior mean and variance
% y = Q(z) + n
% Q(z) = z, if |z| < lambda,
% Q(z) = sign(z)*lambda, otherwise.
% z_pri      N(uz, vz)
% n          N(0, vn)
% uz_post    estimate of z
% vz_post    variance of estimate uz_post
%
%                        --by Feiyan Tian 2021
%
    tema = (lambda - y)./sqrt(2.*vn);
    c_a =  exp(-tema.^2) ./ sqrt(2.*pi.*vn);
    temb = (-lambda - y)./sqrt(2.*vn);
    c_b = exp(-temb.^2) ./ sqrt(2.*pi.*vn);
    
    bound_a = (lambda - uz)./sqrt(2.*vz);
    bound_b = (-lambda - uz)./sqrt(2.*vz);
    
    u_star = (y .* vz + uz .* vn) ./ (vz + vn);
    v_star = (vz .* vn) ./ (vz + vn);
    c_star = sqrt(v_star./(2.*pi.*vz.*vn)) .* exp((u_star.^2 - (y.^2 .* vz + uz.^2 .* vn)./(vz + vn)) ./ (2 .* v_star));
    
    bound_c = (lambda - u_star)./sqrt(2.*v_star);
    bound_d = (-u_star - lambda)./sqrt(2.*v_star);

    norm = ( c_a .* erfc(bound_a) + c_b .* (2-erfc(bound_b)) + c_star .* (erfc(bound_d)-erfc(bound_c)) ) ./2 ;

    uz_post = c_a .* ( sqrt(vz) .* exp(-bound_a.^2) ./ sqrt(2*pi) + uz .* erfc(bound_a)./2 )...,
        + c_b .* ( uz .* (2-erfc(bound_b))./2 - sqrt(vz) .* exp(-bound_b.^2) ./ sqrt(2*pi) )...,
        + c_star .* ( sqrt(v_star) .* (exp(-bound_d.^2)-exp(-bound_c.^2)) ./ sqrt(2*pi) + u_star  .* (erfc(bound_d)-erfc(bound_c))./2 );

    uz_post = uz_post ./ norm;
    
    vz_post_vec = c_a .* ( (uz.^2+vz).*erfc(bound_a)./2 - vz.*bound_b.*exp(-bound_a.^2)./sqrt(pi) )...,
         + c_b .* ( (uz.^2+vz).*(2-erfc(bound_b))./2 + vz.*bound_a.*exp(-bound_b.^2)./sqrt(pi) )...,
         + c_star .* (u_star.^2+v_star).*(erfc(bound_d)-erfc(bound_c))./2 + c_star .* v_star.* ( bound_d.*exp(-bound_c.^2) - bound_c.*exp(-bound_d.^2) )./sqrt(pi);

    vz_post = mean(vz_post_vec ./ norm - uz_post.^2);

end