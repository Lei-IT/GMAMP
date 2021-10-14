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
function vz_post = GOAMP_Declip_SE( clip, z, vz, y, v_n )
%% To calculate the posterior mean and variance
% y = Q(z) + n
% Q(z) = z, if |z| < clip,
% Q(z) = sign(z)*clip, otherwise.
%% Monte Carlo
M = length(z);
zn = normrnd(0, sqrt(vz), [M, 1]);
uz = z + zn;
%%
u_star = (y .* vz + uz .* v_n) ./ (vz+v_n);
sigma_star = (vz .* v_n) ./ (vz+v_n);
c_star = exp((u_star.^2 - ((y.^2) .* vz + uz.^2 .* v_n)./(vz+v_n)) ./ (2 .* sigma_star)) .*sqrt(sigma_star./(2.*pi.*v_n.*vz));

%%
alpha_y = (-y - clip)./sqrt(2.*v_n);
beta_y = (clip-y)./sqrt(2.*v_n);

alpha_star = (-u_star - clip)./sqrt(2.*sigma_star);
beta_star = (clip-u_star)./sqrt(2.*sigma_star);

alpha_x = (-uz - clip)./sqrt(2.*vz);
beta_x = (clip-uz)./sqrt(2.*vz);

%%
p_y_gx = (exp(-alpha_y.^2).*(2-erfc(alpha_x))+exp(-beta_y.^2).*erfc(beta_x))./sqrt(8.*pi.*v_n) ...,
    + (c_star.*(erfc(alpha_star)-erfc(beta_star)))./2;
p_y_gx = max(p_y_gx,1e-20);


integral_x = exp(-alpha_y.^2).*(sqrt(pi/2).*uz.*(2-erfc(alpha_x))-exp(-alpha_x.^2).*sqrt(vz))./(2*pi*sqrt(v_n))...,
    + exp(-beta_y.^2).*(sqrt(pi/2).*uz.*erfc(beta_x) + exp(-beta_x.^2).*sqrt(vz))/(2*pi*sqrt(v_n))...,
    + c_star.*(sqrt(sigma_star).*(exp(-alpha_star.^2)-exp(-beta_star.^2)) + sqrt(pi/2).*u_star.*(erfc(alpha_star)-erfc(beta_star)))./sqrt(2*pi);


integral_x2 = exp(-alpha_y.^2).*(sqrt(pi)./2.*(uz.^2+vz).*(2-erfc(alpha_x)) + beta_x.*exp(-alpha_x.^2).*vz)./(pi*sqrt(2*v_n))...,
     + sigma_star.*c_star.*(alpha_star.*exp(-beta_star.^2) - beta_star.*exp(-alpha_star.^2))./sqrt(pi)...,
     + 0.5*c_star.*(u_star.^2+sigma_star).*(erfc(alpha_star)-erfc(beta_star)) ...,
     + exp(-beta_y.^2).*(sqrt(pi)./2.*(uz.^2+vz).*erfc(beta_x) - alpha_x.*exp(-beta_x.^2).*vz)./(pi*sqrt(2*v_n));

%%
uz_post = integral_x./p_y_gx;

vz_post = mean(integral_x2./p_y_gx - uz_post.^2);

end

